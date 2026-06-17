"""Job submission command group.

This module provides commands for submitting and managing SLURM jobs via FirecREST.

Key features:
- SBATCH option override via -- separator: ``fcw job submit --time 12:00:00 -- script.sh``
- Environment variable injection from config or CLI
- Job monitoring, logs, and cancellation

Usage patterns:
    # Simple submission
    fcw job submit train.sh
    
    # With SBATCH overrides (options before -- override script's #SBATCH directives)
    fcw job submit --time 12:00:00 --nodes 4 -- train.sh
    
    # With job dependency
    JOB1=$(fcw job submit preprocess.sh)
    fcw job submit --dependency afterok:$JOB1 -- train.sh
    
    # Using config-defined job with env override
    fcw job submit train --set CONFIG=exp1.yaml
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.table import Table

import firecrest

from fcw.core import (
    load_config,
    get_async_client,
    get_client,
    get_system,
    get_account,
    extract_job_id,
    resolve_context,
    get_error_console,
    get_output_console,
    get_global_sbatch_options,
    SLURM_FAILED_STATES,
    FcwConfig,
)

app = typer.Typer(no_args_is_help=True)
_error = get_error_console
_output = get_output_console

# SLURM states after which a job produces no further output (stop following).
SLURM_TERMINAL_STATES = SLURM_FAILED_STATES | {"COMPLETED"}

# The follow loop reads a growing log via the `view` endpoint's byte-range parameters
# (offset+size), which return exactly `[offset, offset+size)` at any offset — a bounded,
# delta-only read. `view` rejects size >= 5242880 B (5 MiB) with HTTP 400, but in practice large
# view reads also hit transient SSH-channel errors on some deployments (clariden flaked at 4 MiB,
# was reliable at 1 MiB), so we use a small chunk; a failed read just retries next poll. The
# ranged-view read is verified in tests/e2e/test_e2e_view_range.py.
READ_CHUNK_BYTES = 1 * 1024 * 1024


def _log_grace_seconds() -> float:
    """Seconds to keep waiting for a job's output file to become visible/consistent.

    Covers filesystems (e.g. VAST) where the .out file appears on the login/API
    node noticeably later than the job's terminal SLURM state — the file is at the
    SLURM-reported path, just not yet readable via ``tail``/``stat``. On LUSTRE the
    file is visible immediately, so this grace is never consumed. Override with
    FCW_LOG_GRACE (set to 0 to restore the old fail-fast behavior).
    """
    try:
        return max(0.0, float(os.environ.get("FCW_LOG_GRACE", "30")))
    except ValueError:
        return 30.0


class LogStream(str, Enum):
    """Which job output stream(s) `fcw job logs` operates on."""

    stdout = "stdout"
    stderr = "stderr"
    both = "both"


async def _job_is_terminal(client, system: str, job_id: str) -> bool:
    """Return True once the job has reached a terminal SLURM state.

    Lookup failures are treated as non-terminal so following continues.
    """
    try:
        jobs = await client.job_info(system_name=system, jobid=job_id)
        if not jobs:
            return False
        state = jobs[0].get("status", {}).get("state", "")
        if isinstance(state, list):
            state = ",".join(state)
        return any(ts in str(state) for ts in SLURM_TERMINAL_STATES)
    except Exception:
        return False


def _tail_content(result) -> str:
    """Extract text from a FirecREST tail payload (dict) or a plain string."""
    if isinstance(result, dict):
        return result.get("content") or result.get("output") or ""
    return result or ""


def _emit(content: str, prefix: str = "") -> None:
    """Write a log chunk to stdout, optionally prefixing each non-empty line."""
    if not content:
        return
    if prefix:
        # Preserve the chunk's own newlines; prefix only non-empty lines.
        lines = content.split("\n")
        out = "\n".join(f"{prefix}{ln}" if ln else ln for ln in lines)
        print(out, end="")
    else:
        print(content, end="")


async def _view_range(client, system: str, path: str, offset: int, size: int) -> bytes:
    """Read exactly ``[offset, offset+size)`` of a remote file via the ``view`` endpoint.

    ``view`` accepts ``offset``/``size`` (a true byte-range read) but pyfirecrest's ``view()``
    wrapper omits them, so we call the endpoint directly. ``size`` must stay under 5 MiB.
    TODO: drop the private-helper use once pyfirecrest's ``view()`` exposes ``offset``/``size``.
    """
    resp = await client._get_request(
        endpoint=f"/filesystem/{system}/ops/view",
        params={"path": path, "offset": offset, "size": size},
    )
    out = client._check_response(resp, 200)["output"]
    return (out or "").encode("utf-8")


async def _read_full(client, system: str, path: str) -> None:
    """Emit the entire remote file via bounded, exact ranged ``view`` reads.

    Mirrors the drain loop in ``_follow_stream``: walks the file in
    ``READ_CHUNK_BYTES`` slices so each call stays under the ``view`` per-call
    cap. ``stat`` failure (file not yet visible) propagates to the caller's
    grace-window retry — it raises before any emit, so a retry never duplicates.
    """
    size = int((await client.stat(system_name=system, path=path)).get("size", 0))
    offset = 0
    while offset < size:
        try:
            chunk = await _view_range(
                client, system, path, offset, min(READ_CHUNK_BYTES, size - offset))
        except Exception:
            break  # transient read error: stop (no restart/dup), like the follow loop
        if not chunk:
            break
        _emit(chunk.decode("utf-8", errors="replace"))
        offset += len(chunk)


async def _follow_stream(
    client,
    system: str,
    job_id: str,
    path: str,
    *,
    lines: int,
    tail_only: bool,
    interval: float,
    prefix: str = "",
) -> None:
    """Stream a remote file like `tail -f`, until the job reaches a terminal state.

    Tracks an absolute byte offset and each poll drains the appended bytes via ranged ``view``
    reads (``READ_CHUNK_BYTES`` at a time — bounded under the endpoint's per-call cap, yet
    delta-only), advancing by the bytes actually read so nothing is dropped or skipped if the
    file grows between the ``stat`` and the read.
    """
    async def _size() -> int:
        try:
            return int((await client.stat(system_name=system, path=path)).get("size", 0))
        except Exception:
            return -1

    if tail_only:
        try:
            _emit(_tail_content(
                await client.tail(system_name=system, path=path, num_lines=lines)
            ), prefix)
        except Exception:
            pass
        offset = await _size()
        if offset < 0:
            offset = 0
    else:
        offset = 0  # first delta read prints the whole file so far

    seen = False  # whether the file has ever been visible to stat
    draining_polls = 0
    grace = _log_grace_seconds()
    # Safety cap on post-terminal polls: how many fit in the grace window
    # (interval may be 0 in tests, where the per-poll snapshots bound the loop).
    max_drain_polls = max(1, int(grace / interval) + 1) if interval > 0 else max(1, int(grace) + 1)
    while True:
        terminal = await _job_is_terminal(client, system, job_id)
        size = await _size()
        if size >= 0:
            seen = True
        else:
            size = offset
        if size < offset:  # file rotated/truncated
            offset = 0
        grew = size > offset
        if grew:
            # Drain the backlog in bounded, delta-only ranged reads. Each chunk is exactly
            # ``[offset, offset+n)``; advancing by the bytes returned keeps the offset exact and
            # never skips. A read error (e.g. a transient channel hiccup) breaks the inner loop
            # and retries from the same offset on the next poll.
            while offset < size:
                try:
                    chunk = await _view_range(
                        client, system, path, offset, min(READ_CHUNK_BYTES, size - offset))
                except Exception:
                    break
                if not chunk:
                    break
                _emit(chunk.decode("utf-8", errors="replace"), prefix)
                offset += len(chunk)
        if terminal:
            # Output can lag the terminal SLURM state — on VAST the file may not be
            # visible yet (see _log_grace_seconds). Keep draining until it has
            # appeared and shows no new bytes, or the grace window elapses (bounded
            # so a persistent read error / endless-growth report can't hang us).
            draining_polls += 1
            if (seen and not grew) or draining_polls >= max_drain_polls:
                break
        await asyncio.sleep(interval)


def _follow_streams(system: str, job_id: str, selected, *,
                    tail: bool, lines: int, interval: float = 2.0) -> None:
    """Follow one or more log streams (gathered) until the job is terminal.

    ``selected`` is a list of ``(label, path, suffix)``; lines are prefixed with
    ``[label] `` only when more than one stream is followed. Ctrl-C stops cleanly.
    """
    multi = len(selected) > 1

    async def _run():
        async_client = get_async_client()
        await asyncio.gather(*[
            _follow_stream(
                async_client, system, job_id, path,
                lines=lines, tail_only=tail, interval=interval,
                prefix=f"[{label}] " if multi else "",
            )
            for label, path, _ in selected
        ])

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def _report_final_state(client, system: str, job_id: str) -> None:
    """Block until the job finishes; print its state and exit 1 on failure."""
    job_info = client.wait_for_job(system_name=system, job_id=job_id)
    state = job_info[0]["status"]["state"]
    if isinstance(state, list):
        state = ",".join(state)
    if any(fs in state for fs in SLURM_FAILED_STATES):
        _error().print(f"[red]Job {job_id} finished with state: {state}[/red]")
        _error().print(f"[dim]Hint: Run `fcw job logs {job_id}` to see output[/dim]")
        raise typer.Exit(1)
    _error().print(f"[green]Job {job_id} completed ({state})[/green]")


def _job_stream_paths(client, system: str, job_id: str) -> tuple[Optional[str], Optional[str]]:
    """Resolve (stdout_path, stderr_path) from job metadata, %j-expanded.

    Returns (None, None) when no metadata is available. Lets client errors propagate.
    """
    metadata = client.job_metadata(system_name=system, jobid=job_id)
    if isinstance(metadata, list):
        metadata = metadata[0] if metadata else None
    if not metadata:
        return None, None

    # TODO(FirecREST): the job_metadata response returns SLURM filename patterns
    # (%j, %x, %N, ...) unexpanded (FirecREST bug). We expand %j client-side as
    # a temporary workaround and warn when it occurs; remove once fixed upstream.
    def _pick(*keys: str) -> Optional[str]:
        return next((metadata.get(k) for k in keys if metadata.get(k)), None)

    out = _pick("standardOutput", "stdout", "StdOut")
    err = _pick("standardError", "stderr", "StdErr")
    if (out and "%j" in out) or (err and "%j" in err):
        _error().print(
            "[yellow]Warning:[/yellow] FirecREST returned an unexpanded output path "
            "(contains '%j'); expanding it client-side as a temporary workaround "
            "(FirecREST bug)."
        )
    return (out.replace("%j", job_id) if out else None,
            err.replace("%j", job_id) if err else None)


# -----------------------------------------------------------------------------
# Job listing helpers
# -----------------------------------------------------------------------------

def _dig(obj, path: str):
    """Return the value at a dotted path (``a.b.c``) within nested dicts, or None."""
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _fmt_duration(seconds) -> str:
    """Format a duration in seconds as ``[D-]H:MM:SS``; falsy/invalid -> ''."""
    if seconds is None or seconds == "":
        return ""
    try:
        total = int(seconds)
    except (TypeError, ValueError):
        return ""
    if total < 0:
        return ""
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    base = f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{days}-{base}" if days else base


def _fmt_epoch(ts) -> str:
    """Format a Unix timestamp as ``YYYY-MM-DD HH:MM``; falsy/invalid -> ''."""
    try:
        if not ts:
            return ""
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return ""


def _job_state(job: dict) -> str:
    """Normalize ``status.state`` (string or list) to a single string."""
    state = _dig(job, "status.state")
    if isinstance(state, list):
        return ",".join(state)
    return str(state) if state else ""


def _is_terminal_state(state: str) -> bool:
    return any(ts in state for ts in SLURM_TERMINAL_STATES)


def _job_reason(job: dict) -> str:
    reason = _dig(job, "status.stateReason")
    return "" if reason in (None, "None", "") else str(reason)


def _job_time_left(job: dict) -> str:
    """Remaining walltime (limit - elapsed) for active jobs; '' once terminal."""
    if _is_terminal_state(_job_state(job)):
        return ""
    limit, elapsed = _dig(job, "time.limit"), _dig(job, "time.elapsed")
    if limit is None:
        return ""
    try:
        return _fmt_duration(int(limit) - int(elapsed or 0))
    except (TypeError, ValueError):
        return ""


# Column specs: (header, render_fn, optional). Optional columns are hidden when
# empty for every displayed row. `long` columns are only considered with --long.
_CORE_COLUMNS = [
    ("Job ID", lambda j: str(extract_job_id(j) or ""), False),
    ("Name", lambda j: str(j.get("name") or ""), False),
    ("User", lambda j: str(j.get("user") or ""), True),
    ("Partition", lambda j: str(j.get("partition") or ""), True),
    ("State", _job_state, False),
    ("Reason", _job_reason, True),
    ("Nodes", lambda j: str(j.get("allocationNodes") or ""), True),
    ("Elapsed", lambda j: _fmt_duration(_dig(j, "time.elapsed")), True),
    ("Time Left", _job_time_left, True),
]
_LONG_COLUMNS = [
    ("Account", lambda j: str(j.get("account") or ""), True),
    ("Nodelist", lambda j: str(j.get("nodes") or ""), True),
    ("Start", lambda j: _fmt_epoch(_dig(j, "time.start")), True),
    ("End", lambda j: _fmt_epoch(_dig(j, "time.end")), True),
    ("Time Limit", lambda j: _fmt_duration(_dig(j, "time.limit")), True),
    ("Priority", lambda j: str(j.get("priority") or ""), True),
]


def _build_jobs_table(jobs, *, long: bool = False, state: Optional[str] = None,
                      user: Optional[str] = None, partition: Optional[str] = None):
    """Build the `job list` Rich table and a per-state Counter from job dicts.

    Filters client-side by state (substring), user (exact), and partition (exact);
    optional columns with no data across the kept rows are hidden.
    """
    kept = []
    for job in jobs:
        if state and state.lower() not in _job_state(job).lower():
            continue
        if user and str(job.get("user") or "") != user:
            continue
        if partition and str(job.get("partition") or "") != partition:
            continue
        kept.append(job)

    specs = _CORE_COLUMNS + (_LONG_COLUMNS if long else [])
    # Precompute every cell once, then drop optional all-empty columns.
    rendered = [[render(job) for _, render, _ in specs] for job in kept]
    visible = []
    for idx, (header, _, optional) in enumerate(specs):
        if optional and not any(row[idx] for row in rendered):
            continue
        visible.append((idx, header))

    table = Table(title="Jobs")
    for _, header in visible:
        table.add_column(header)
    for row in rendered:
        table.add_row(*(row[idx] for idx, _ in visible))

    counts = Counter(_job_state(job) for job in kept)
    return table, counts


# -----------------------------------------------------------------------------
# SBATCH Script Manipulation Helpers
# -----------------------------------------------------------------------------

def _apply_sbatch_overrides(script_content: str, overrides: dict[str, str]) -> str:
    """Apply SBATCH option overrides to a SLURM script.
    
    For each override, either replaces an existing ``#SBATCH --key`` directive
    or inserts a new one after the existing SBATCH block.
    
    Args:
        script_content: Original script content.
        overrides: Dict of SBATCH options to set (key -> value).
                   Keys should NOT include the leading ``--``. An empty-string
                   value denotes a valueless flag, rendered as a bare
                   ``#SBATCH --key`` (e.g. ``{"exclusive": ""}``).
    
    Returns:
        Modified script content with overrides applied.
    
    Example:
        >>> script = \"\"\"#!/bin/bash
        ... #SBATCH --time=01:00:00
        ... #SBATCH --nodes=1
        ... echo "hello"
        ... \"\"\"
        >>> _apply_sbatch_overrides(script, {"time": "12:00:00", "gpus-per-node": "4"})
        # Returns script with --time changed to 12:00:00 and --gpus-per-node=4 added
    """
    if not overrides:
        return script_content
    
    lines = script_content.split("\n")
    modified_keys = set()
    last_sbatch_idx = -1
    
    # First pass: find and replace existing SBATCH directives
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#SBATCH"):
            last_sbatch_idx = i
            
            # Parse the SBATCH option from this line.
            # Handles: #SBATCH --key=value, #SBATCH --key value, and valueless
            # flags (value == "") rendered as a bare #SBATCH --key.
            # NOTE: multiple options on one line and the short `-k value` form are
            # not reconciled here — SLURM rejects inconsistent overrides.
            for key, value in overrides.items():
                if value == "":
                    # Flag: treat an existing bare `#SBATCH --key` as already set.
                    if re.match(rf'^\s*#SBATCH\s+--{re.escape(key)}\s*$', line):
                        modified_keys.add(key)
                        break
                    continue
                # Match both --key=... and --key ...
                pattern = rf'^\s*#SBATCH\s+--{re.escape(key)}(?:=|\s+)(\S+)'
                match = re.match(pattern, line)
                if match:
                    old_value = match.group(1)
                    if old_value != value:
                        _error().print(
                            f"[yellow]Overriding script #SBATCH "
                            f"--{key} ({old_value} → {value})[/yellow]"
                        )
                    lines[i] = f"#SBATCH --{key}={value}"
                    modified_keys.add(key)
                    break
    
    # Second pass: insert new directives for keys not found
    new_directives = []
    for key, value in overrides.items():
        if key not in modified_keys:
            directive = f"#SBATCH --{key}" if value == "" else f"#SBATCH --{key}={value}"
            new_directives.append(directive)
    
    if new_directives:
        # Insert after last #SBATCH line, or after shebang if no SBATCH found
        if last_sbatch_idx >= 0:
            insert_idx = last_sbatch_idx + 1
        elif lines and lines[0].startswith("#!"):
            insert_idx = 1
        else:
            insert_idx = 0
        
        # Insert in reverse: each insert() at the same index pushes the
        # previous one down, so reversing preserves new_directives' order.
        for directive in reversed(new_directives):
            lines.insert(insert_idx, directive)
    
    return "\n".join(lines)


def _inject_env_vars(script_content: str, env_vars: dict[str, str]) -> str:
    """Inject environment variable exports into SLURM script.
    
    Inserts export statements after the #SBATCH block. Variables use
    shell default syntax (``${VAR:-default}``) so CLI-provided values
    take precedence.
    
    Args:
        script_content: Original script content.
        env_vars: Dict of variable name -> default value.
    
    Returns:
        Modified script with exports inserted.
    """
    if not env_vars:
        return script_content

    exports = "\n".join(f'export {k}="${{{k}:-{v}}}"' for k, v in env_vars.items())
    lines = script_content.split("\n")

    # Insert after container TOML block if present, otherwise after last #SBATCH
    insert_idx = -1
    last_sbatch_idx = -1
    for i, line in enumerate(lines):
        if "export FCW_CONTAINER_TOML=" in line:
            insert_idx = i + 1
            break
        if line.strip().startswith("#SBATCH"):
            last_sbatch_idx = i

    if insert_idx < 0:
        insert_idx = last_sbatch_idx + 1 if last_sbatch_idx >= 0 else (1 if lines[0].startswith("#!") else 0)
    
    # Add blank line and exports
    lines.insert(insert_idx, "")
    lines.insert(insert_idx + 1, "# Environment variables from fcw")
    lines.insert(insert_idx + 2, exports)
    
    return "\n".join(lines)


def _resolve_job_env(config, job_config, overrides: dict[str, str]) -> dict[str, str]:
    """Resolve job environment variables.

    ``env_paths`` values are treated as paths: relative ones are resolved
    against the configured remote workdir. ``env`` values are literals, passed
    through untouched. CLI overrides (``--set KEY=VALUE``) win over both and are
    type-faithful: an override of a declared ``env_paths`` key is resolved as a
    path (unless already absolute/``$``-prefixed); every other override is
    literal.

    Args:
        config: The FcwConfig object.
        job_config: The JobConfig for this job.
        overrides: CLI-provided overrides (--set KEY=VALUE).

    Returns:
        Dict of resolved environment variables.
    """
    def _resolve_path_value(value: str) -> str:
        if not value.startswith("/") and not value.startswith("$"):
            return config.resolve_path(value, remote=True)
        return value

    env = {}

    # Path-valued env vars: resolve relative paths against the remote workdir.
    for key, value in job_config.env_paths.items():
        env[key] = _resolve_path_value(value)

    # Literal env vars: passed through untouched.
    for key, value in job_config.env.items():
        env[key] = value

    # CLI overrides win; an override of a declared path key stays a path.
    for key, value in overrides.items():
        env[key] = _resolve_path_value(value) if key in job_config.env_paths else value

    return env


def _remote_script_name(job_name: str, script_path: str, is_config_job: bool) -> str:
    """Remote filename for a ``--remote-script`` upload.

    Config jobs keep the stable, human-readable ``<job_name>.sh``. A raw script
    path uses the script's basename, so an absolute/relative path isn't appended
    to ``.fcw/scripts/`` verbatim (which produced a nested, double-``.sh`` path
    and a 404).
    """
    if is_config_job:
        return f"{job_name}.sh"
    return os.path.basename(script_path)


def _build_container_toml(config: FcwConfig, container_name: str) -> str:
    """Build resolved container TOML content for injection into a SLURM script.

    If the container config has a ``toml`` path, reads that file and overrides
    the ``image`` field with the resolved sqsh path.  Otherwise generates a
    minimal TOML with the resolved image path.

    Args:
        config: The FcwConfig object.
        container_name: Name of the container in ``config.containers``.

    Returns:
        TOML content string with resolved image path.

    Raises:
        typer.Exit: If the container name is unknown or the TOML file is missing.
    """
    if container_name not in config.containers:
        _error().print(f"[red]Unknown container: {container_name}[/red]")
        _error().print(
            f"[dim]Available containers: {', '.join(config.containers)}[/dim]"
        )
        raise typer.Exit(1)

    cont = config.containers[container_name]
    image_path = config.resolve_container_image(cont)

    if cont.toml:
        toml_path = Path(cont.toml)
        if not toml_path.exists():
            _error().print(f"[red]Container TOML not found: {cont.toml}[/red]")
            raise typer.Exit(1)
        toml_content = toml_path.read_text()
        # Override the image field with the resolved absolute path
        toml_content = re.sub(
            r'^image\s*=\s*"[^"]*"',
            f'image = "{image_path}"',
            toml_content,
            count=1,
            flags=re.MULTILINE,
        )
        return toml_content
    else:
        return f'image = "{image_path}"\nwritable = true\n'


def _inject_container_toml(script_content: str, toml_content: str) -> str:
    """Inject container TOML heredoc into a SLURM script.

    Writes the TOML to ``/dev/shm/fcw-container-${SLURM_JOB_ID}.toml`` at
    runtime and exports ``FCW_CONTAINER_TOML`` pointing to it.

    The heredoc uses a quoted delimiter (``<< 'FCWEOF'``) so that shell
    variables inside the TOML (e.g. ``${SCRATCH}``) are written literally
    for pyxis/enroot to expand at runtime.

    Inserted after #SBATCH directives, before any other content.

    Args:
        script_content: Original script content.
        toml_content: Resolved TOML content to inline.

    Returns:
        Modified script with heredoc block inserted.
    """
    block = (
        "\n# Container environment from fcw\n"
        "cat > /dev/shm/fcw-container-${SLURM_JOB_ID}.toml << 'FCWEOF'\n"
        f"{toml_content.rstrip()}\n"
        "FCWEOF\n"
        "export FCW_CONTAINER_TOML=/dev/shm/fcw-container-${SLURM_JOB_ID}.toml\n"
        'csrun() { srun --environment="$FCW_CONTAINER_TOML" "$@"; }\n'
        "export -f csrun"
    )

    lines = script_content.split("\n")

    # Find the last #SBATCH line
    last_sbatch_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("#SBATCH"):
            last_sbatch_idx = i

    insert_idx = (
        last_sbatch_idx + 1
        if last_sbatch_idx >= 0
        else (1 if lines and lines[0].startswith("#!") else 0)
    )

    lines.insert(insert_idx, block)
    return "\n".join(lines)


def _norm_toml_path(p: str) -> str:
    """Normalize a TOML path for comparison (strip a single leading ``./``)."""
    return p[2:] if p.startswith("./") else p


def _environment_tokens(script_content: str) -> List[str]:
    """Return every ``<X>`` appearing as ``--environment <X>`` / ``--environment=<X>``.

    Surrounding quotes are stripped. Used to detect srun calls that would bypass
    the fcw-managed container environment.
    """
    tokens = re.findall(r"--environment(?:=|\s+)(\S+)", script_content)
    return [t.strip("'\"") for t in tokens]


def _rewrite_environment_path(script_content: str, bound_path: str) -> str:
    """Point hardcoded ``--environment <bound_path>`` at the injected TOML.

    Replaces ``--environment`` tokens whose value matches the bound container's
    configured TOML path (modulo a leading ``./``) with ``${FCW_CONTAINER_TOML}``,
    so a script can keep its natural ``--environment ./env/foo.toml`` and still run
    under the fcw-resolved env. The injected heredoc/export lines reference the
    ``/dev/shm`` path, not ``bound_path``, so they are untouched.
    """
    target = _norm_toml_path(bound_path)

    def _sub(m: "re.Match[str]") -> str:
        value = m.group(2).strip("'\"")
        if _norm_toml_path(value) == target:
            return f"{m.group(1)}${{FCW_CONTAINER_TOML}}"
        return m.group(0)

    return re.sub(r"(--environment(?:=|\s+))(\S+)", _sub, script_content)


def _warn_env_bindings(script_content: str, bound_path: Optional[str], bound: bool) -> None:
    """Warn when an ``srun --environment`` would bypass the fcw-managed env.

    A token is "managed" if it references ``FCW_CONTAINER_TOML`` or matches the
    bound container's TOML path (which gets rewritten). Anything else runs outside
    the resolved env; a binding with no managed reference has no effect; and a
    reference with no binding is not fcw-managed at all.

    Args:
        bound: whether any container env is bound (via flag or configured job).
        bound_path: the bound container's configured TOML path, if any (used to
            recognize a hardcoded path that will be rewritten). May be None even
            when ``bound`` is True (container without a ``toml`` file).
    """
    tokens = _environment_tokens(script_content)
    target = _norm_toml_path(bound_path) if bound_path else None

    def _is_managed(tok: str) -> bool:
        return "FCW_CONTAINER_TOML" in tok or (target is not None and _norm_toml_path(tok) == target)

    for tok in tokens:
        if not _is_managed(tok):
            _error().print(
                f"[yellow]Warning:[/yellow] srun --environment {tok} runs outside the "
                "fcw-managed container environment (image path not resolved by fcw)."
            )

    if bound:
        if not any(_is_managed(t) for t in tokens):
            _error().print(
                "[yellow]Warning:[/yellow] a container is bound but no "
                "srun --environment references it (binding has no effect)."
            )
    elif tokens or "FCW_CONTAINER_TOML" in script_content:
        _error().print(
            "[yellow]Warning:[/yellow] the script references a container environment "
            "but none is bound; pass --container/--environment or use a configured job."
        )


def _parse_sbatch_args(args: List[str]) -> tuple[dict[str, str], List[str]]:
    """Parse SBATCH options from argument list.
    
    Separates SBATCH options (before ``--``) from remaining arguments.
    
    Args:
        args: Raw argument list, potentially containing SBATCH options,
              a ``--`` separator, and remaining positional args.
    
    Returns:
        Tuple of (sbatch_overrides, remaining_args).
    
    Example:
        >>> _parse_sbatch_args(["--time", "12:00:00", "--nodes", "4", "--", "train.sh"])
        ({"time": "12:00:00", "nodes": "4"}, ["train.sh"])
        
        >>> _parse_sbatch_args(["train.sh"])  # No SBATCH options
        ({}, ["train.sh"])
    """
    sbatch_opts = {}
    remaining = []
    
    # Check if there's a -- separator
    if "--" in args:
        sep_idx = args.index("--")
        sbatch_args = args[:sep_idx]
        remaining = args[sep_idx + 1:]
        
        # Parse SBATCH options (--key value or --key=value)
        i = 0
        while i < len(sbatch_args):
            arg = sbatch_args[i]
            if arg.startswith("--"):
                key = arg[2:]  # Remove leading --
                if "=" in key:
                    key, value = key.split("=", 1)
                    sbatch_opts[key] = value
                elif i + 1 < len(sbatch_args) and not sbatch_args[i + 1].startswith("--"):
                    sbatch_opts[key] = sbatch_args[i + 1]
                    i += 1
                else:
                    # Flag without value (rare for SBATCH but handle it)
                    sbatch_opts[key] = ""
            i += 1
    else:
        # No separator - all args are positional/remaining
        remaining = args
    
    return sbatch_opts, remaining


# -----------------------------------------------------------------------------
# Job Commands
# -----------------------------------------------------------------------------

@app.command(
    "submit",
    context_settings={
        "allow_interspersed_args": False,
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def submit_job(
    ctx: typer.Context,
    set_vars: Optional[List[str]] = typer.Option(
        None, "--set",
        help="Override env var: KEY=VALUE"
    ),
    container: Optional[str] = typer.Option(
        None, "--container",
        help="Container name from fcw.yaml to bind (overrides the job's configured container)"
    ),
    environment: Optional[str] = typer.Option(
        None, "--environment",
        help="Path to a TOML file to inline as the container env "
             "(mutually exclusive with --container)"
    ),
    wait: bool = typer.Option(False, "--wait/--no-wait", "-w", help="Wait for job completion"),
    follow: bool = typer.Option(False, "--follow", "-f",
                                help="Stream job output until it finishes (implies --wait)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print job ID"),
    remote_script: bool = typer.Option(
        False, "--remote-script",
        help="Upload script to remote before submitting (workaround for slurmrestd/pyxis "
             "segfault). Uses a fixed remote filename and is not safe for concurrent "
             "submissions; submit serially. Scripts are not cleaned up remotely."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show modified script without submitting"),
):
    """Submit a job with optional SBATCH overrides.

    SBATCH options placed BEFORE the ``--`` separator override directives in the script.
    This allows runtime customization without editing the script file.

    Examples:
        # Simple submission (script path or config job name)
        fcw job submit train.sh
        fcw job submit train  # uses jobs.train.script from fcw.yaml

        # Override SBATCH options
        fcw job submit --time 12:00:00 --nodes 4 -- train.sh

        # Chain jobs with dependency
        JOB1=$(fcw job submit preprocess.sh)
        fcw job submit --dependency afterok:$JOB1 -- train.sh

        # Bind a container env explicitly (overrides the job's configured one)
        fcw job submit --container app -- train.sh

        # Stream output live until the job finishes (like srun)
        fcw job submit --follow -- train.sh

        # Set environment variables
        fcw job submit train --set CONFIG=exp1.yaml --set EPOCHS=100
    """
    config, system, account = resolve_context(ctx)

    # With ignore_unknown_options + allow_extra_args, Click passes unknown
    # options (SBATCH overrides) through ctx.args instead of rejecting them.
    # With allow_interspersed_args=False, Click also stops binding known options
    # after the first positional arg, so fcw flags after the job name end up here too.
    args = ctx.args
    sbatch_overrides, remaining = _parse_sbatch_args(args or [])

    # Extract fcw options from remaining args (after -- or after job name).
    _FCW_BOOL_FLAGS = {
        "--wait": "wait", "-w": "wait",
        "--no-wait": "no-wait",
        "--follow": "follow", "-f": "follow",
        "--quiet": "quiet", "-q": "quiet",
        "--dry-run": "dry_run",
        "--remote-script": "remote_script",
    }
    filtered_remaining = []
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg in _FCW_BOOL_FLAGS:
            flag = _FCW_BOOL_FLAGS[arg]
            if flag == "wait":
                wait = True
            elif flag == "no-wait":
                wait = False
            elif flag == "follow":
                follow = True
            elif flag == "quiet":
                quiet = True
            elif flag == "dry_run":
                dry_run = True
            elif flag == "remote_script":
                remote_script = True
        elif arg == "--set" and i + 1 < len(remaining):
            set_vars = (set_vars or []) + [remaining[i + 1]]
            i += 1
        elif arg == "--container" and i + 1 < len(remaining):
            container = remaining[i + 1]
            i += 1
        elif arg == "--environment" and i + 1 < len(remaining):
            environment = remaining[i + 1]
            i += 1
        else:
            filtered_remaining.append(arg)
        i += 1
    remaining = filtered_remaining

    # Error if first remaining arg looks like an SBATCH option (missing -- separator)
    if remaining and remaining[0].startswith("--"):
        _error().print(
            f"[red]Error: Expected a script path or job name but got "
            f"'{remaining[0]}', which looks like an SBATCH option.[/red]"
        )
        _error().print(
            "[dim]SBATCH options must be placed before a -- separator:\n"
            "  fcw job submit --time 12:00:00 --nodes 4 -- train.sh[/dim]"
        )
        raise typer.Exit(1)

    # Error if SBATCH-style options appear after the job name
    stray = [a for a in remaining[1:] if a.startswith("--")]
    if stray:
        _error().print(
            f"[red]Error: SBATCH-style options found after the script/job name: "
            f"{', '.join(stray)}[/red]"
        )
        _error().print(
            "[dim]Place SBATCH options before the -- separator:\n"
            f"  fcw job submit {' '.join(stray)} -- {remaining[0]}[/dim]"
        )
        raise typer.Exit(1)

    if not remaining:
        _error().print("[red]Error: No script or job name provided[/red]")
        _error().print("[dim]Usage: fcw job submit [SBATCH_OPTS]... -- <script|job_name> [--set KEY=VALUE]...[/dim]")
        raise typer.Exit(1)

    job_name = remaining[0]

    if container and environment:
        _error().print(
            "[red]Error: --container and --environment are mutually exclusive[/red]"
        )
        raise typer.Exit(1)
    if environment and not Path(environment).exists():
        _error().print(f"[red]Environment TOML not found: {environment}[/red]")
        raise typer.Exit(1)

    # Parse --set overrides
    overrides = {}
    if set_vars:
        for s in set_vars:
            if "=" not in s:
                _error().print(f"[red]Invalid --set format: {s} (expected KEY=VALUE)[/red]")
                raise typer.Exit(1)
            k, v = s.split("=", 1)
            overrides[k] = v

    # Determine if job_name is a config job or a script path. An explicit
    # --container/--environment overrides the job's configured container.
    container_name = None
    config_sbatch: dict[str, str] = {}
    if job_name in config.jobs:
        job_config = config.jobs[job_name]
        script_path = job_config.script
        container_name = job_config.container
        config_sbatch = job_config.sbatch_options()

        # Resolve environment variables from config
        env_vars = _resolve_job_env(config, job_config, overrides)
    else:
        # Treat as script path
        script_path = job_name
        env_vars = overrides
    if container:
        container_name = container

    # Merge SBATCH options: CLI args > job config > global env > script directives
    sbatch_overrides = {**get_global_sbatch_options(), **config_sbatch, **sbatch_overrides}

    # Read and modify script
    if not os.path.exists(script_path):
        _error().print(f"[red]Script not found: {script_path}[/red]")
        raise typer.Exit(1)

    script_content = Path(script_path).read_text()
    script_content = _apply_sbatch_overrides(script_content, sbatch_overrides)

    # Resolve the container env binding (explicit --environment/--container, else
    # the named job's container). Raw scripts bind only via these — no inference.
    toml_content = None
    bound_path: Optional[str] = None
    if environment:
        toml_content = Path(environment).read_text()
        bound_path = environment
    elif container_name:
        from fcw.commands.container import _resync_container_patches
        _resync_container_patches(config, container_name, system, account)
        toml_content = _build_container_toml(config, container_name)
        if container_name in config.containers:
            bound_path = config.containers[container_name].toml

    # Warn about srun --environment calls that bypass the managed env (before
    # injection rewrites them), then inject + rewrite the hardcoded path.
    _warn_env_bindings(script_content, bound_path, bound=toml_content is not None)
    if toml_content is not None:
        script_content = _inject_container_toml(script_content, toml_content)
        if bound_path:
            script_content = _rewrite_environment_path(script_content, bound_path)

    script_content = _inject_env_vars(script_content, env_vars)

    if remote_script:
        _error().print(
            "[yellow]Warning:[/yellow] --remote-script uploads a fixed-name remote script "
            "(.fcw/scripts/<name>.sh) and is not safe for concurrent submissions — submit "
            "serially. Scripts are not cleaned up remotely."
        )

    if dry_run:
        # Header is a label (stderr); the script body is the artifact (stdout),
        # so `fcw job submit --dry-run … > script.sh` yields a clean script.
        _error().print(f"[bold]Modified script ({script_path}):[/bold]")
        _output().print(script_content)
        return

    # Submit job
    client = get_client()
    working_dir = config.workdir.remote

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        modified_script_path = f.name

    try:
        # TODO: remove --remote-script workaround when slurmrestd inline script
        # + pyxis segfault is fixed (inline scripts don't initialize the pyxis
        # SPANK plugin properly, causing srun --environment to segfault)
        if remote_script:
            remote_scripts_dir = f"{working_dir}/.fcw/scripts"
            remote_filename = _remote_script_name(job_name, script_path, job_name in config.jobs)
            client.mkdir(system_name=system, path=remote_scripts_dir, create_parents=True)
            client.upload(
                system_name=system,
                local_file=modified_script_path,
                directory=remote_scripts_dir,
                filename=remote_filename,
                account=account,
                transfer_method="s3",
            )
            result = client.submit(
                system_name=system,
                account=account,
                working_dir=working_dir,
                script_remote_path=f"{remote_scripts_dir}/{remote_filename}",
            )
        else:
            result = client.submit(
                system_name=system,
                account=account,
                working_dir=working_dir,
                script_local_path=modified_script_path,
            )

        job_id = extract_job_id(result)

        # Print job ID to stdout for scripting
        print(job_id)

        # Print details to stderr (unless quiet)
        if not quiet:
            _error().print(f"[green]Submitted job {job_id}[/green]", highlight=False)
            if sbatch_overrides:
                override_str = ", ".join(f"{k}={v}" for k, v in sbatch_overrides.items())
                _error().print(f"[dim]SBATCH overrides: {override_str}[/dim]")
    finally:
        os.unlink(modified_script_path)

    if follow:
        stdout_path = None
        try:
            stdout_path, _ = _job_stream_paths(client, system, job_id)
        except Exception:
            pass
        if stdout_path:
            if not quiet:
                _error().print(f"[dim]Following job {job_id} (Ctrl-C to stop)...[/dim]")
            _follow_streams(system, job_id, [("stdout", stdout_path, "out")],
                            tail=False, lines=50)
        else:
            _error().print(
                "[yellow]Could not resolve stdout path from metadata; "
                "waiting for completion instead.[/yellow]"
            )
        _report_final_state(client, system, job_id)
    elif wait:
        _error().print(f"[dim]Waiting for job {job_id}...[/dim]")
        _report_final_state(client, system, job_id)


@app.command(
    "run",
    context_settings={
        "allow_interspersed_args": False,
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def run_command(
    ctx: typer.Context,
    time: str = typer.Option("00:30:00", "--time", "-t", help="Default time limit"),
    nodes: int = typer.Option(1, "--nodes", "-N", help="Default number of nodes"),
    wait: bool = typer.Option(False, "--wait/--no-wait", "-w", help="Wait for job completion"),
    follow: bool = typer.Option(False, "--follow", "-f",
                                help="Stream job output until it finishes (implies --wait)"),
    container: Optional[str] = typer.Option(
        None, "--container",
        help="Container name from fcw.yaml to run the command in (defines csrun)"
    ),
    environment: Optional[str] = typer.Option(
        None, "--environment",
        help="Path to a TOML file to inline as the container env (mutually exclusive with --container)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the final script to stdout and exit without submitting"
    ),
    remote_script: bool = typer.Option(
        False, "--remote-script",
        help="Upload script to remote before submitting (workaround for slurmrestd/pyxis "
             "segfault). Uses a fixed remote filename and is not safe for concurrent "
             "submissions; submit serially. Scripts are not cleaned up remotely."
    ),
):
    """Run an ad-hoc command as a SLURM job.

    Similar to ``sbatch --wrap``, submitting a batch job via FirecREST.
    SBATCH options before ``--`` override the defaults.

    Examples:
        # Simple command
        fcw job run 'echo "Hello from $(hostname)"'

        # With SBATCH overrides
        fcw job run --time 01:00:00 --nodes 2 -- 'nvidia-smi'

        # In a container (csrun = srun --environment=$FCW_CONTAINER_TOML)
        fcw job run --container mycont -- 'csrun python analyze.py'

        # Stream output live until the job finishes (like srun)
        fcw job run --follow -- 'python train.py'
    """
    config, system, account = resolve_context(ctx)

    if container and environment:
        _error().print(
            "[red]Error: --container and --environment are mutually exclusive[/red]"
        )
        raise typer.Exit(1)

    if environment and not Path(environment).exists():
        _error().print(f"[red]Environment TOML not found: {environment}[/red]")
        raise typer.Exit(1)

    args = ctx.args
    sbatch_overrides, remaining = _parse_sbatch_args(args or [])

    if not remaining:
        _error().print("[red]Error: No command provided[/red]")
        _error().print("[dim]Usage: fcw job run [SBATCH_OPTS]... -- <command>[/dim]")
        raise typer.Exit(1)

    command = " ".join(remaining)

    # Apply defaults, then overrides
    sbatch_defaults = {
        "job-name": "fcw-run",
        "time": time,
        "nodes": str(nodes),
        "output": "fcw-run-%j.out",
    }
    sbatch_final = {**sbatch_defaults, **get_global_sbatch_options(), **sbatch_overrides}

    # Build script
    sbatch_lines = "\n".join(f"#SBATCH --{k}={v}" for k, v in sbatch_final.items())

    script_content = f"""#!/bin/bash -l
{sbatch_lines}

{command}
"""

    if container:
        from fcw.commands.container import _resync_container_patches
        _resync_container_patches(config, container, system, account)
        toml_content = _build_container_toml(config, container)
        script_content = _inject_container_toml(script_content, toml_content)
    elif environment:
        toml_content = Path(environment).read_text()
        script_content = _inject_container_toml(script_content, toml_content)

    if remote_script:
        _error().print(
            "[yellow]Warning:[/yellow] --remote-script uploads a fixed-name remote script "
            "(.fcw/scripts/<name>.sh) and is not safe for concurrent submissions — submit "
            "serially. Scripts are not cleaned up remotely."
        )

    if dry_run:
        _output().print(script_content)
        return

    client = get_client()
    working_dir = config.workdir.remote

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # TODO: remove --remote-script workaround when slurmrestd inline script
        # + pyxis segfault is fixed
        if remote_script:
            remote_scripts_dir = f"{working_dir}/.fcw/scripts"
            remote_filename = "fcw-run.sh"  # FIXME: make script name unique to avoid collisions if multiple fcw-run commands are submitted in a short time frame?
            client.mkdir(system_name=system, path=remote_scripts_dir, create_parents=True)
            client.upload(
                system_name=system,
                local_file=script_path,
                directory=remote_scripts_dir,
                filename=remote_filename,
                account=account,
                transfer_method="s3",
            )
            result = client.submit(
                system_name=system,
                account=account,
                working_dir=working_dir,
                script_remote_path=f"{remote_scripts_dir}/{remote_filename}",
            )
        else:
            result = client.submit(
                system_name=system,
                account=account,
                working_dir=working_dir,
                script_local_path=script_path,
            )

        job_id = extract_job_id(result)
        print(job_id)
        _error().print(f"[green]Submitted job {job_id}[/green]")
    finally:
        os.unlink(script_path)

    if follow:
        stdout_path = None
        try:
            stdout_path, _ = _job_stream_paths(client, system, job_id)
        except Exception:
            pass
        if stdout_path:
            _error().print(f"[dim]Following job {job_id} (Ctrl-C to stop)...[/dim]")
            _follow_streams(system, job_id, [("stdout", stdout_path, "out")],
                            tail=False, lines=50)
        else:
            _error().print(
                "[yellow]Could not resolve stdout path from metadata; "
                "waiting for completion instead.[/yellow]"
            )
        _report_final_state(client, system, job_id)
    elif wait:
        _error().print(f"[dim]Waiting for job {job_id}...[/dim]")
        _report_final_state(client, system, job_id)


@app.command("status")
def job_status(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID"),
):
    """Get status of a job."""
    system = get_system((ctx.obj or {}).get("system"))

    client = get_client()
    
    try:
        jobs = client.job_info(system_name=system, jobid=job_id)
        if not jobs:
            _error().print(f"[yellow]No info found for job {job_id}[/yellow]")
            raise typer.Exit(1)

        job = jobs[0]
        table = Table(title=f"Job {job_id}")
        table.add_column("Field")
        table.add_column("Value")

        for key, value in job.items():
            table.add_row(str(key), str(value))
        
        _output().print(table)
    except Exception as e:
        _error().print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("logs")
def job_logs(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID"),
    stream: LogStream = typer.Option(
        LogStream.stdout, "--stream",
        help="Which stream to operate on: stdout, stderr, or both",
    ),
    follow: bool = typer.Option(False, "--follow", "-f",
                                help="Follow output until the job finishes (like tail -f)"),
    download: bool = typer.Option(
        False, "--download", "-d",
        help="Download log file(s), keeping the remote filename; by default "
             "mirrors the remote path under workdir.local"),
    dest: Optional[str] = typer.Option(
        None, "--dest",
        help="Directory to download log(s) into (overrides the workdir.local mirror)"),
    lines: Optional[int] = typer.Option(
        None, "--lines", "-n",
        help="Show only the last N lines (default: entire file)"),
):
    """View job stdout/stderr logs.

    Examples:
        fcw job logs ID            # entire file
        fcw job logs ID -n 50      # last 50 lines
        fcw job logs ID -f         # entire file, then follow
        fcw job logs ID -f -n 50   # last 50 lines, then follow
    """
    config, system, account = resolve_context(ctx)

    client = get_client()

    # Get job metadata to find output file(s)
    try:
        stdout_path, stderr_path = _job_stream_paths(client, system, job_id)
    except Exception as e:
        _error().print(f"[red]Error getting job metadata: {e}[/red]")
        raise typer.Exit(1)

    if not stdout_path:
        _error().print("[red]Could not determine stdout path from job metadata[/red]")
        raise typer.Exit(1)

    # Build the selected (label, path, suffix) streams. Collapse to a single
    # unprefixed stdout follower when stderr is absent or combined into stdout.
    combined = (not stderr_path) or (stderr_path == stdout_path)
    if stream is LogStream.stderr and not combined:
        selected = [("stderr", stderr_path, "err")]
    elif stream is LogStream.both and not combined:
        selected = [("stdout", stdout_path, "out"), ("stderr", stderr_path, "err")]
    else:
        if stream is not LogStream.stdout and combined:
            _error().print("[dim]stdout and stderr are combined; showing the single log.[/dim]")
        selected = [("stdout", stdout_path, "out")]

    multi = len(selected) > 1

    if download:
        async def do_download():
            async_client = get_async_client()
            for label, path, _ in selected:
                if dest is not None:
                    local_path = os.path.join(dest, os.path.basename(path))
                else:
                    # Mirror the remote path's position under workdir.remote
                    # into the corresponding location under workdir.local.
                    rel = os.path.relpath(path, config.workdir.remote)
                    if rel.startswith("..") or os.path.isabs(rel):
                        rel = os.path.basename(path)  # log lives outside workdir.remote
                    local_path = os.path.join(config.workdir.local, rel)
                parent = os.path.dirname(local_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                await async_client.download(
                    system_name=system,
                    source_path=path,
                    target_path=local_path,
                    account=account,
                    blocking=True,
                    transfer_method="s3",
                )
                _error().print(f"[green]Downloaded {label} to {local_path}[/green]")

        asyncio.run(do_download())
        return

    if follow:
        _follow_streams(system, job_id, selected,
                        tail=(lines is not None), lines=lines or 0)
        return

    # One-shot read. The output file can lag the job's terminal state on some
    # filesystems (e.g. VAST), so retry a not-yet-readable file within the grace
    # window (see _log_grace_seconds) before reporting an error.
    async def do_read():
        async_client = get_async_client()
        interval = 2.0
        attempts = max(1, int(_log_grace_seconds() / interval) + 1)
        for label, path, _ in selected:
            if multi:
                print(f"==> {label} <==")
            for attempt in range(attempts):
                try:
                    if lines is None:
                        await _read_full(async_client, system, path)
                    else:
                        _emit(_tail_content(await async_client.tail(
                            system_name=system, path=path, num_lines=lines)))
                    break
                except Exception as e:
                    if attempt + 1 >= attempts:
                        _error().print(f"[red]Error reading {label}: {e}[/red]")
                    else:
                        await asyncio.sleep(interval)

    asyncio.run(do_read())


@app.command("wait")
def wait_for_jobs(
    ctx: typer.Context,
    job_ids: List[str] = typer.Argument(..., help="Job IDs to wait for"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="Timeout in seconds"),
):
    """Wait for one or more jobs to complete."""
    system = get_system((ctx.obj or {}).get("system"))

    client = get_client()
    
    for job_id in job_ids:
        _error().print(f"[dim]Waiting for job {job_id}...[/dim]")
        try:
            job_info = client.wait_for_job(system_name=system, job_id=job_id)
            state = job_info[0]["status"]["state"]
            if isinstance(state, list):
                state = ",".join(state)
            if any(fs in state for fs in SLURM_FAILED_STATES):
                _error().print(f"[red]Job {job_id} finished with state: {state}[/red]")
                _error().print(f"[dim]Hint: Run `fcw job logs {job_id}` to see output[/dim]")
                raise typer.Exit(1)
            _error().print(f"[green]Job {job_id} completed ({state})[/green]")
        except typer.Exit:
            raise
        except Exception as e:
            _error().print(f"[red]Job {job_id} failed: {e}[/red]")
            raise typer.Exit(1)


@app.command("cancel")
def cancel_jobs(
    ctx: typer.Context,
    job_ids: List[str] = typer.Argument(..., help="Job IDs to cancel"),
):
    """Cancel one or more jobs."""
    system = get_system((ctx.obj or {}).get("system"))

    client = get_client()
    
    for job_id in job_ids:
        try:
            client.cancel_job(system_name=system, jobid=job_id)
            _error().print(f"[green]Cancelled job {job_id}[/green]")
        except Exception as e:
            _error().print(f"[red]Failed to cancel {job_id}: {e}[/red]")


@app.command("list")
def list_jobs(
    ctx: typer.Context,
    state: Optional[str] = typer.Option(None, "--state", "-s", help="Filter by state"),
    user: Optional[str] = typer.Option(None, "--user", "-u",
                                        help="Filter by user (implies --all-users; "
                                             "may be slow/unsupported on busy systems)"),
    all_users: bool = typer.Option(False, "--all-users", "-a",
                                   help="Show all users' jobs (runs a cluster-wide sacct; "
                                        "may be slow or unsupported on busy systems)"),
    account: Optional[str] = typer.Option(None, "--account", help="Filter by account"),
    partition: Optional[str] = typer.Option(None, "--partition", "-p",
                                            help="Filter by partition"),
    long: bool = typer.Option(False, "--long", "-l", help="Show additional columns"),
):
    """List jobs on the cluster."""
    system = get_system((ctx.obj or {}).get("system"))

    client = get_client()

    try:
        want_all = all_users or bool(user)
        kwargs = {"allusers": want_all}
        if account:
            kwargs["account"] = account
        try:
            jobs = client.job_info(system_name=system, **kwargs)
        except firecrest.NotImplementedOnAPIversion as e:
            _error().print(f"[yellow]Warning: {e} Ignoring --all-users/--account.[/yellow]")
            jobs = client.job_info(system_name=system)
        except Exception as e:
            # The only multi-user mechanism is allusers=True, which makes FirecREST
            # run a cluster-wide `sacct` — often very slow / 500s on busy systems.
            if not want_all:
                raise
            _error().print(
                "[red]Listing all users' jobs failed.[/red] The FirecREST all-users query "
                "runs a cluster-wide `sacct` that is often very slow and can time out on "
                "busy systems. Retry without --all-users/--user, or narrow with --account."
            )
            _error().print(f"[dim]Underlying error: {e}[/dim]")
            raise typer.Exit(1)

        table, counts = _build_jobs_table(
            jobs, long=long, state=state, user=user, partition=partition,
        )
        _output().print(table)

        if counts:
            total = sum(counts.values())
            parts = "  ".join(f"{s}: {n}" for s, n in sorted(counts.items()))
            _output().print(f"[dim]Total: {total} · {parts}[/dim]", highlight=False)
    except typer.Exit:
        raise
    except Exception as e:
        _error().print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
