"""Job submission command group.

This module provides commands for submitting and managing SLURM jobs via FirecREST.

Key features:
- SBATCH option override via -- separator: ``fcw job submit --time 24:00:00 -- script.sh``
- Environment variable injection from config or CLI
- Job monitoring, logs, and cancellation

Usage patterns:
    # Simple submission
    fcw job submit train.sh
    
    # With SBATCH overrides (options before -- override script's #SBATCH directives)
    fcw job submit --time 24:00:00 --nodes 4 -- train.sh
    
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
import sys
import tempfile
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

import firecrest

from fcw.core import load_config, get_async_client, get_client, get_system, get_account

app = typer.Typer(no_args_is_help=True)
console = Console()


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
        >>> _apply_sbatch_overrides(script, {"time": "24:00:00", "gpus-per-node": "4"})
        # Returns script with --time changed to 24:00:00 and --gpus-per-node=4 added
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
            for key, value in overrides.items():
                # Match both --key=... and --key ...
                pattern = rf'^(\s*#SBATCH\s+--{re.escape(key)})(?:=|\s+)\S*(.*)$'
                match = re.match(pattern, line)
                if match:
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
    
    # Find the last #SBATCH line
    last_sbatch_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("#SBATCH"):
            last_sbatch_idx = i
    
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
        if not value.startswith("/") and not value.startswith("$"):
            value = config.resolve_path(value, remote=True)
        env[key] = value
    
    # Apply overrides
    env.update(overrides)
    
    return env


def _parse_sbatch_args(args: List[str]) -> tuple[dict[str, str], List[str]]:
    """Parse SBATCH options from argument list.
    
    Separates SBATCH options (before ``--``) from remaining arguments.
    
    Args:
        args: Raw argument list, potentially containing SBATCH options,
              a ``--`` separator, and remaining positional args.
    
    Returns:
        Tuple of (sbatch_overrides, remaining_args).
    
    Example:
        >>> _parse_sbatch_args(["--time", "24:00:00", "--nodes", "4", "--", "train.sh"])
        ({"time": "24:00:00", "nodes": "4"}, ["train.sh"])
        
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

@app.command("submit", context_settings={"allow_interspersed_args": False})
def submit_job(
    ctx: typer.Context,
    args: List[str] = typer.Argument(None, help="[SBATCH_OPTS]... -- <script> [--set KEY=VALUE]..."),
    set_vars: Optional[List[str]] = typer.Option(
        None, "--set", "-e",
        help="Override env var: KEY=VALUE"
    ),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for job completion"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print job ID"),
):
    """Submit a job with optional SBATCH overrides.
    
    SBATCH options placed BEFORE the ``--`` separator override directives in the script.
    This allows runtime customization without editing the script file.
    
    Examples:
        # Simple submission (script path or config job name)
        fcw job submit train.sh
        fcw job submit train  # uses jobs.train.script from fcw.yaml
        
        # Override SBATCH options
        fcw job submit --time 24:00:00 --nodes 4 -- train.sh
        
        # Chain jobs with dependency
        JOB1=$(fcw job submit preprocess.sh)
        fcw job submit --dependency afterok:$JOB1 -- train.sh
        
        # Set environment variables
        fcw job submit train --set CONFIG=exp1.yaml --set EPOCHS=100
    """
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)
    
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    account = get_account(ctx.obj.get("account") if ctx.obj else None)
    
    # Parse SBATCH options and remaining arguments
    sbatch_overrides, remaining = _parse_sbatch_args(args or [])
    
    if not remaining:
        console.print("[red]Error: No script or job name provided[/red]")
        console.print("[dim]Usage: fcw job submit [SBATCH_OPTS]... -- <script|job_name> [--set KEY=VALUE]...[/dim]")
        raise typer.Exit(1)
    
    job_name = remaining[0]
    
    # Parse --set overrides
    overrides = {}
    if set_vars:
        for s in set_vars:
            if "=" not in s:
                console.print(f"[red]Invalid --set format: {s} (expected KEY=VALUE)[/red]")
                raise typer.Exit(1)
            k, v = s.split("=", 1)
            overrides[k] = v
    
    # Determine if job_name is a config job or a script path
    if job_name in config.jobs:
        job_config = config.jobs[job_name]
        script_path = job_config.script
        
        # Resolve environment variables from config
        env_vars = _resolve_job_env(config, job_config, overrides)
    else:
        # Treat as script path
        script_path = job_name
        env_vars = overrides
    
    # Read and modify script
    if not os.path.exists(script_path):
        console.print(f"[red]Script not found: {script_path}[/red]")
        raise typer.Exit(1)
    
    script_content = open(script_path).read()
    script_content = _apply_sbatch_overrides(script_content, sbatch_overrides)
    script_content = _inject_env_vars(script_content, env_vars)
    
    # Submit job
    client = get_client()
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        modified_script_path = f.name
    
    try:
        working_dir = config.workdir.remote
        
        result = client.submit(
            system_name=system,
            account=account,
            working_dir=working_dir,
            script_local_path=modified_script_path,
        )
        
        job_id = result.get("jobId") or result.get("jobid") or result.get("job_id")
        
        # Print job ID to stdout for scripting
        print(job_id)
        
        # Print details to stderr (unless quiet)
        if not quiet:
            console.print(f"[green]Submitted job {job_id}[/green]", highlight=False)
            if sbatch_overrides:
                override_str = ", ".join(f"{k}={v}" for k, v in sbatch_overrides.items())
                console.print(f"[dim]SBATCH overrides: {override_str}[/dim]")
    finally:
        os.unlink(modified_script_path)
    
    if wait:
        console.print(f"[dim]Waiting for job {job_id}...[/dim]")
        client.wait_for_job(system_name=system, job_id=job_id)
        console.print(f"[green]Job {job_id} completed[/green]")


@app.command("run", context_settings={"allow_interspersed_args": False})
def run_command(
    ctx: typer.Context,
    args: List[str] = typer.Argument(None, help="[SBATCH_OPTS]... -- <command>"),
    time: str = typer.Option("00:30:00", "--time", "-t", help="Default time limit"),
    nodes: int = typer.Option(1, "--nodes", "-N", help="Default number of nodes"),
):
    """Run an ad-hoc command as a SLURM job.
    
    Similar to ``srun``, but submits as a batch job via FirecREST.
    SBATCH options before ``--`` override the defaults.
    
    Examples:
        # Simple command
        fcw job run 'echo "Hello from $(hostname)"'
        
        # With SBATCH overrides
        fcw job run --time 01:00:00 --nodes 2 -- 'nvidia-smi'
        
        # With dependency
        fcw job run --dependency afterok:12345 -- 'python analyze.py'
    """
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)
    
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    account = get_account(ctx.obj.get("account") if ctx.obj else None)
    
    # Parse SBATCH options and remaining arguments
    sbatch_overrides, remaining = _parse_sbatch_args(args or [])
    
    if not remaining:
        console.print("[red]Error: No command provided[/red]")
        console.print("[dim]Usage: fcw job run [SBATCH_OPTS]... -- <command>[/dim]")
        raise typer.Exit(1)
    
    command = " ".join(remaining)
    
    # Apply defaults, then overrides
    sbatch_defaults = {
        "job-name": "fcw-run",
        "time": time,
        "nodes": str(nodes),
        "output": "fcw-run-%j.out",
    }
    sbatch_final = {**sbatch_defaults, **sbatch_overrides}
    
    # Build script
    sbatch_lines = "\n".join(f"#SBATCH --{k}={v}" for k, v in sbatch_final.items())
    
    script_content = f"""#!/bin/bash -l
{sbatch_lines}

{command}
"""
    
    client = get_client()
    working_dir = config.workdir.remote
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        result = client.submit(
            system_name=system,
            account=account,
            working_dir=working_dir,
            script_local_path=script_path,
        )
        
        job_id = result.get("jobId") or result.get("jobid") or result.get("job_id")
        print(job_id)
        console.print(f"[green]Submitted job {job_id}[/green]")
    finally:
        os.unlink(script_path)


@app.command("status")
def job_status(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID"),
):
    """Get status of a job."""
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    
    client = get_client()
    
    try:
        jobs = client.job_info(system_name=system, jobid=job_id)
        if not jobs:
            console.print(f"[yellow]No info found for job {job_id}[/yellow]")
            raise typer.Exit(1)

        job = jobs[0]
        table = Table(title=f"Job {job_id}")
        table.add_column("Field")
        table.add_column("Value")

        for key, value in job.items():
            table.add_row(str(key), str(value))
        
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("logs")
def job_logs(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID"),
    tail: bool = typer.Option(False, "--tail", "-t", help="Show last lines only"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow output (poll)"),
    download: bool = typer.Option(False, "--download", "-d", help="Download log file"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines for --tail"),
):
    """View job stdout/stderr logs."""
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)
    
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    account = get_account(ctx.obj.get("account") if ctx.obj else None)
    
    client = get_client()
    
    # Get job metadata to find output file
    try:
        metadata_list = client.job_metadata(system_name=system, jobid=job_id)
        if not metadata_list:
            console.print(f"[red]No metadata found for job {job_id}[/red]")
            raise typer.Exit(1)
        metadata = metadata_list[0]
        stdout_path = (
            metadata.get("standardOutput")
            or metadata.get("stdout")
            or metadata.get("StdOut")
        )
        
        if not stdout_path:
            console.print("[red]Could not determine stdout path from job metadata[/red]")
            raise typer.Exit(1)
        
        # Resolve %j to job_id if present
        stdout_path = stdout_path.replace("%j", job_id)
        
    except Exception as e:
        console.print(f"[red]Error getting job metadata: {e}[/red]")
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
        console.print(f"[green]Downloaded to {local_path}[/green]")
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
                
                if follow:
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
                console.print(f"[red]Error: {e}[/red]")
                break
    
    asyncio.run(do_tail())


@app.command("wait")
def wait_for_jobs(
    ctx: typer.Context,
    job_ids: List[str] = typer.Argument(..., help="Job IDs to wait for"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="Timeout in seconds"),
):
    """Wait for one or more jobs to complete."""
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    
    client = get_client()
    
    for job_id in job_ids:
        console.print(f"[dim]Waiting for job {job_id}...[/dim]")
        try:
            client.wait_for_job(system_name=system, job_id=job_id)
            console.print(f"[green]Job {job_id} completed[/green]")
        except Exception as e:
            console.print(f"[red]Job {job_id} failed: {e}[/red]")
            raise typer.Exit(1)


@app.command("cancel")
def cancel_jobs(
    ctx: typer.Context,
    job_ids: List[str] = typer.Argument(..., help="Job IDs to cancel"),
):
    """Cancel one or more jobs."""
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    
    client = get_client()
    
    for job_id in job_ids:
        try:
            client.cancel_job(system_name=system, jobid=job_id)
            console.print(f"[green]Cancelled job {job_id}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to cancel {job_id}: {e}[/red]")


@app.command("list")
def list_jobs(
    ctx: typer.Context,
    state: Optional[str] = typer.Option(None, "--state", "-s", help="Filter by state"),
):
    """List jobs on the cluster."""
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    
    client = get_client()
    
    try:
        jobs = client.job_list(system_name=system)
        
        table = Table(title="Jobs")
        table.add_column("Job ID")
        table.add_column("Name")
        table.add_column("State")
        table.add_column("Time")
        
        for job in jobs:
            job_state = job.get("state", "")
            if state and state.lower() not in job_state.lower():
                continue
            
            table.add_row(
                str(job.get("jobId", job.get("job_id", ""))),
                job.get("name", ""),
                job_state,
                job.get("time", ""),
            )
        
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
