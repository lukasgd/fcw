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
    get_console,
    get_global_sbatch_options,
    SLURM_FAILED_STATES,
    FcwConfig,
)

app = typer.Typer(no_args_is_help=True)
_console = get_console


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
                   Keys should NOT include the leading ``--``.
    
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
            
            # Parse the SBATCH option from this line
            # Handles: #SBATCH --key=value, #SBATCH --key value, #SBATCH -k value
            for key, value in overrides.items(): # FIXME: does this handle (or error on) multiple SBATCH options on the same line? Furthermore, where is #SBATCH -k value handled? Or should we let SLURM just report an error if the user uses inconsistent overrides (maybe easier)?
                # Match both --key=... and --key ...
                pattern = rf'^\s*#SBATCH\s+--{re.escape(key)}(?:=|\s+)(\S+)'
                match = re.match(pattern, line)
                if match:
                    old_value = match.group(1)
                    if old_value != value:
                        _console().print(
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
            new_directives.append(f"#SBATCH --{key}={value}")
    
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
    
    Expands relative paths against the configured remote workdir and
    applies any CLI overrides.
    
    Args:
        config: The FcwConfig object.
        job_config: The JobConfig for this job.
        overrides: CLI-provided overrides (--set KEY=VALUE).
    
    Returns:
        Dict of resolved environment variables.
    """
    env = {}
    
    for key, value in job_config.env.items():
        # Expand relative paths
        if not value.startswith("/") and not value.startswith("$"):  # FIXME: probably we don't need all environment variables to be resolved as remote paths
            value = config.resolve_path(value, remote=True)
        env[key] = value
    
    # Apply overrides (resolve relative paths the same way as config values)
    for key, value in overrides.items():
        if not value.startswith("/") and not value.startswith("$"):  #FIXME: probably we don't need all environment variables to be resolved as remote paths
            value = config.resolve_path(value, remote=True)
        env[key] = value

    return env


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
        _console().print(f"[red]Unknown container: {container_name}[/red]")
        _console().print(
            f"[dim]Available containers: {', '.join(config.containers)}[/dim]"
        )
        raise typer.Exit(1)

    cont = config.containers[container_name]
    image_path = config.resolve_container_image(cont)

    if cont.toml:
        toml_path = Path(cont.toml)
        if not toml_path.exists():
            _console().print(f"[red]Container TOML not found: {cont.toml}[/red]")
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
        None, "--set", "-e",
        help="Override env var: KEY=VALUE"
    ),
    wait: bool = typer.Option(False, "--wait/--no-wait", "-w", help="Wait for job completion"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print job ID"),
    remote_script: bool = typer.Option(
        False, "--remote-script",
        help="Upload script to remote before submitting (workaround for slurmrestd/pyxis segfault)"
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
            elif flag == "quiet":
                quiet = True
            elif flag == "dry_run":
                dry_run = True
            elif flag == "remote_script":
                remote_script = True
        elif arg in ("--set", "-e") and i + 1 < len(remaining):
            set_vars = (set_vars or []) + [remaining[i + 1]]
            i += 1
        else:
            filtered_remaining.append(arg)
        i += 1
    remaining = filtered_remaining

    # Error if first remaining arg looks like an SBATCH option (missing -- separator)
    if remaining and remaining[0].startswith("--"):
        _console().print(
            f"[red]Error: Expected a script path or job name but got "
            f"'{remaining[0]}', which looks like an SBATCH option.[/red]"
        )
        _console().print(
            "[dim]SBATCH options must be placed before a -- separator:\n"
            "  fcw job submit --time 12:00:00 --nodes 4 -- train.sh[/dim]"
        )
        raise typer.Exit(1)

    # Error if SBATCH-style options appear after the job name
    stray = [a for a in remaining[1:] if a.startswith("--")]
    if stray:
        _console().print(
            f"[red]Error: SBATCH-style options found after the script/job name: "
            f"{', '.join(stray)}[/red]"
        )
        _console().print(
            "[dim]Place SBATCH options before the -- separator:\n"
            f"  fcw job submit {' '.join(stray)} -- {remaining[0]}[/dim]"
        )
        raise typer.Exit(1)

    if not remaining:
        _console().print("[red]Error: No script or job name provided[/red]")
        _console().print("[dim]Usage: fcw job submit [SBATCH_OPTS]... -- <script|job_name> [--set KEY=VALUE]...[/dim]")
        raise typer.Exit(1)

    job_name = remaining[0]

    # Parse --set overrides
    overrides = {}
    if set_vars:
        for s in set_vars:
            if "=" not in s:
                _console().print(f"[red]Invalid --set format: {s} (expected KEY=VALUE)[/red]")
                raise typer.Exit(1)
            k, v = s.split("=", 1)
            overrides[k] = v

    # Determine if job_name is a config job or a script path
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

    # Merge SBATCH options: CLI args > job config > global env > script directives
    sbatch_overrides = {**get_global_sbatch_options(), **config_sbatch, **sbatch_overrides}

    # Read and modify script
    if not os.path.exists(script_path):
        _console().print(f"[red]Script not found: {script_path}[/red]")
        raise typer.Exit(1)

    script_content = Path(script_path).read_text()
    script_content = _apply_sbatch_overrides(script_content, sbatch_overrides)

    # Inject container TOML before env vars (ordering matters).
    # When submitting a raw script (not a config job), infer the container
    # if the script references FCW_CONTAINER_TOML.
    if not container_name and "FCW_CONTAINER_TOML" in script_content:  # FIXME: Why not requiring the user to explicitly specify the container with --container when submitting a script instead of a named job, if FCW_CONTAINER_TOML is present? One could also offer the option to override the toml by having an extra option --environment (both for named jobs and script submissions). If FCW_CONTAINER_TOML is absent, should check that --environment is not being used, otherwise inform user that container environment is not managed by fcw. Probably any srun with a --environment option that doesn't evaluate to an FCW_CONTAINER_TOML injection should be logged to the user so they don't accidentally bypass the fcw-managed container environment without realizing it.
        # Try to find a job config that uses this script
        for jname, jcfg in config.jobs.items():
            if jcfg.script == script_path and jcfg.container:
                container_name = jcfg.container
                break
        # Fall back to the sole container with a TOML
        if not container_name:
            toml_containers = [
                name for name, c in config.containers.items() if c.toml
            ]
            if len(toml_containers) == 1:
                container_name = toml_containers[0]
    if container_name:
        from fcw.commands.container import _resync_container_patches
        _resync_container_patches(config, container_name, system, account)
        toml_content = _build_container_toml(config, container_name)
        script_content = _inject_container_toml(script_content, toml_content)

    script_content = _inject_env_vars(script_content, env_vars)

    if dry_run:
        _console().print(f"[bold]Modified script ({script_path}):[/bold]")
        _console().print(script_content)
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
            remote_filename = f"{job_name}.sh"
            client.mkdir(system_name=system, path=remote_scripts_dir, create_parents=True)
            client.upload(
                system_name=system,
                local_file=modified_script_path,
                directory=remote_scripts_dir,
                filename=remote_filename,
                account=account,
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
            _console().print(f"[green]Submitted job {job_id}[/green]", highlight=False)
            if sbatch_overrides:
                override_str = ", ".join(f"{k}={v}" for k, v in sbatch_overrides.items())
                _console().print(f"[dim]SBATCH overrides: {override_str}[/dim]")
    finally:
        os.unlink(modified_script_path)

    if wait:
        _console().print(f"[dim]Waiting for job {job_id}...[/dim]")
        job_info = client.wait_for_job(system_name=system, job_id=job_id)
        state = job_info[0]["status"]["state"]
        if isinstance(state, list):
            state = ",".join(state)
        if any(fs in state for fs in SLURM_FAILED_STATES):
            _console().print(f"[red]Job {job_id} finished with state: {state}[/red]")
            _console().print(f"[dim]Hint: Run `fcw job logs {job_id}` to see output[/dim]")
            raise typer.Exit(1)
        _console().print(f"[green]Job {job_id} completed ({state})[/green]")


@app.command(
    "run",
    context_settings={
        "allow_interspersed_args": False,
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def run_command(  # TODO: add --wait option similar to submit, additionally add an option that follows job logs in real time until completion, similar to running `srun`
    ctx: typer.Context,
    time: str = typer.Option("00:30:00", "--time", "-t", help="Default time limit"),
    nodes: int = typer.Option(1, "--nodes", "-N", help="Default number of nodes"),
    container: Optional[str] = typer.Option(
        None, "--container", "-c",
        help="Container name from fcw.yaml to run the command in (defines csrun)"
    ),
    environment: Optional[str] = typer.Option(
        None, "--environment", "-e",
        help="Path to a TOML file to inline as the container env (mutually exclusive with --container)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the final script to stdout and exit without submitting"
    ),
    remote_script: bool = typer.Option(
        False, "--remote-script",
        help="Upload script to remote before submitting (workaround for slurmrestd/pyxis segfault)"
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
        fcw job run -c mycont -- 'csrun python analyze.py'
    """
    config, system, account = resolve_context(ctx)

    if container and environment:
        _console().print(
            "[red]Error: --container and --environment are mutually exclusive[/red]"
        )
        raise typer.Exit(1)

    if environment and not Path(environment).exists():
        _console().print(f"[red]Environment TOML not found: {environment}[/red]")
        raise typer.Exit(1)

    args = ctx.args
    sbatch_overrides, remaining = _parse_sbatch_args(args or [])

    if not remaining:
        _console().print("[red]Error: No command provided[/red]")
        _console().print("[dim]Usage: fcw job run [SBATCH_OPTS]... -- <command>[/dim]")
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

    if dry_run:
        _console().print(script_content)
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
        _console().print(f"[green]Submitted job {job_id}[/green]")
    finally:
        os.unlink(script_path)


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
            _console().print(f"[yellow]No info found for job {job_id}[/yellow]")
            raise typer.Exit(1)

        job = jobs[0]
        table = Table(title=f"Job {job_id}")
        table.add_column("Field")
        table.add_column("Value")

        for key, value in job.items():
            table.add_row(str(key), str(value))
        
        _console().print(table)
    except Exception as e:
        _console().print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("logs")
def job_logs( # FIXME: this currently follows only stdout. Consider also supporting stderr (e.g. by checking job metadata for separate stderr path, or by allowing the user to specify which stream to follow). Furthermore, consider supporting the case where stdout and stderr are combined into a single file (e.g. by checking job metadata for a single output path and documenting that behavior).
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID"),
    tail: bool = typer.Option(False, "--tail", help="Show last lines only"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow output (poll)"),
    download: bool = typer.Option(False, "--download", "-d", help="Download log file"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines for --tail"),
):
    """View job stdout/stderr logs."""
    config, system, account = resolve_context(ctx)

    client = get_client()
    
    # Get job metadata to find output file
    try:
        metadata_list = client.job_metadata(system_name=system, jobid=job_id)
        if not metadata_list:
            _console().print(f"[red]No metadata found for job {job_id}[/red]")
            raise typer.Exit(1)
        metadata = metadata_list[0]
        stdout_path = (
            metadata.get("standardOutput")
            or metadata.get("stdout")
            or metadata.get("StdOut")
        )
        
        if not stdout_path:
            _console().print("[red]Could not determine stdout path from job metadata[/red]")
            raise typer.Exit(1)
        
        # Resolve %j to job_id if present
        stdout_path = stdout_path.replace("%j", job_id)
        
    except Exception as e:
        _console().print(f"[red]Error getting job metadata: {e}[/red]")
        raise typer.Exit(1)
    
    if download:
        local_path = f"job-{job_id}.out"
        
        async def do_download():
            async_client = get_async_client()
            await async_client.download(
                system_name=system,
                source_path=stdout_path,
                target_path=local_path,
                account=account,
                blocking=True,
            )
        
        asyncio.run(do_download())
        _console().print(f"[green]Downloaded to {local_path}[/green]")
        return
    
    async def do_tail():
        async_client = get_async_client()
        seen_lines = 0
        
        while True:
            try:
                result = await async_client.tail(
                    system_name=system,
                    path=stdout_path,
                    num_lines=lines if tail else 1000,
                )
                
                output = result if isinstance(result, str) else (
                    result.get("content") or result.get("output") or ""
                )
                output_lines = output.split("\n")
                
                if follow:  # FIXME: this is faulty because seen_lines points into the current tail output, which gets reset every 2 seconds. To properly implement --follow (which should mimick tail -f in a shell on the remote system), would need to keep track of the file offset and use that for subsequent reads (polling on tail endpoint might not be suitable for this).
                    # Only print new lines
                    new_lines = output_lines[seen_lines:]
                    if new_lines:
                        for line in new_lines:
                            print(line)
                        seen_lines = len(output_lines)
                    await asyncio.sleep(2)
                else:
                    print(output)
                    break
                    
            except Exception as e:
                if follow:
                    await asyncio.sleep(2)
                    continue
                _console().print(f"[red]Error: {e}[/red]")
                break
    
    asyncio.run(do_tail())


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
        _console().print(f"[dim]Waiting for job {job_id}...[/dim]")
        try:
            job_info = client.wait_for_job(system_name=system, job_id=job_id)
            state = job_info[0]["status"]["state"]
            if isinstance(state, list):
                state = ",".join(state)
            if any(fs in state for fs in SLURM_FAILED_STATES):
                _console().print(f"[red]Job {job_id} finished with state: {state}[/red]")
                _console().print(f"[dim]Hint: Run `fcw job logs {job_id}` to see output[/dim]")
                raise typer.Exit(1)
            _console().print(f"[green]Job {job_id} completed ({state})[/green]")
        except typer.Exit:
            raise
        except Exception as e:
            _console().print(f"[red]Job {job_id} failed: {e}[/red]")
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
            _console().print(f"[green]Cancelled job {job_id}[/green]")
        except Exception as e:
            _console().print(f"[red]Failed to cancel {job_id}: {e}[/red]")


@app.command("list")
def list_jobs(
    ctx: typer.Context,
    state: Optional[str] = typer.Option(None, "--state", "-s", help="Filter by state"),
):
    """List jobs on the cluster."""
    system = get_system((ctx.obj or {}).get("system"))

    client = get_client()
    
    try:
        jobs = client.job_info(system_name=system)

        table = Table(title="Jobs")  # FIXME: usually, on the cluster squeue --me displays all of: JOBID, USER, ACCOUNT, PARTITION, NAME,  EXEC_HOST, ST, REASON, START_TIME, END_TIME, TIME_LEFT, NODES, PRIORITY. A --long/-l version displays NODELIST in addition. Consider adding more of these fields to the table, and also supporting filtering by user (e.g. --user) and other criteria (e.g. partition, time range, etc.). Furthermore, consider adding a summary line with counts of jobs in each state (e.g. RUNNING: 5, PENDING: 3, etc.).
        table.add_column("Job ID")
        table.add_column("Name")
        table.add_column("State")
        table.add_column("Time")

        for job in jobs:
            # State may be in status.state (list or string)
            status = job.get("status", {})
            job_state = status.get("state", "") if isinstance(status, dict) else job.get("state", "")
            if isinstance(job_state, list):
                job_state = ",".join(job_state)
            job_state = str(job_state)

            if state and state.lower() not in job_state.lower():
                continue

            # Time limit may be nested under limits
            limits = job.get("limits", {})
            time_str = limits.get("time", "") if isinstance(limits, dict) else job.get("time", "")

            table.add_row(
                str(extract_job_id(job)),
                str(job.get("name", "")),
                job_state,
                str(time_str),
            )

        _console().print(table)
    except Exception as e:
        _console().print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
