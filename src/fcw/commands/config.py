"""Config command group."""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.table import Table

from fcw.core import (
    ContainerConfig,
    DirectoryType,
    JobConfig,
    add_container_to_config_roundtrip,
    add_directory_to_config,
    add_job_to_config,
    generate_default_config,
    generate_interactive_config,
    get_async_client,
    get_client,
    get_console,
    get_system,
    load_config,
    remove_container_from_config,
    remove_directory_from_config,
    remove_job_from_config,
)

app = typer.Typer(no_args_is_help=True)
_console = get_console

# Nested sub-typers for managing config sections
directory_app = typer.Typer(no_args_is_help=True)
container_app = typer.Typer(no_args_is_help=True)
job_app = typer.Typer(no_args_is_help=True)

app.add_typer(directory_app, name="directory", help="Manage directory entries")
app.add_typer(container_app, name="container", help="Manage container entries")
app.add_typer(job_app, name="job", help="Manage job entries")


def _is_interactive() -> bool:
    """Check if stdin is a TTY (interactive terminal)."""
    return sys.stdin.isatty()


def _resolve_config_path(ctx: typer.Context) -> Path:
    """Resolve config file path from context or default to ./fcw.yaml."""
    custom = (ctx.obj or {}).get("config_file")
    if custom:
        p = Path(custom)
        if not p.exists():
            _console().print(f"[red]Config file not found: {custom}[/red]")
            raise typer.Exit(1)
        return p
    default = Path.cwd() / "fcw.yaml"
    if not default.exists():
        _console().print("[red]No fcw.yaml found. Run 'fcw config init' first.[/red]")
        raise typer.Exit(1)
    return default


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Skip prompts, use static template"
    ),
) -> None:
    """Create fcw.yaml template in current directory."""
    config_path = Path.cwd() / "fcw.yaml"

    if config_path.exists() and not force:
        _console().print(f"[red]Config file already exists: {config_path}[/red]")
        _console().print("Use --force to overwrite.")
        raise typer.Exit(1)

    if non_interactive or not _is_interactive():
        config_path.write_text(generate_default_config())
    else:
        project = typer.prompt("Project name", default=Path.cwd().name)
        remote_workdir = typer.prompt(
            "Remote workdir", default=f"${{FIRECREST_SCRATCH}}/{project}"
        )
        local_workdir = typer.prompt("Local workdir", default=".")
        config_path.write_text(
            generate_interactive_config(project, remote_workdir, local_workdir)
        )

    _console().print(f"[green]Created config file: {config_path}[/green]")
    _console().print("\nEdit the file to configure your project, then run:")
    _console().print("  fcw config validate")


# ---------------------------------------------------------------------------
# Directory sub-commands
# ---------------------------------------------------------------------------


@directory_app.command("add")
def directory_add(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Directory path (e.g., data/raw)"),
    type: DirectoryType = typer.Option(DirectoryType.BOTH, "--type", "-t", help="Directory type"),
) -> None:
    """Add a directory entry to fcw.yaml."""
    config_path = _resolve_config_path(ctx)
    try:
        add_directory_to_config(config_path, path, type)
        _console().print(f"[green]Added directory '{path}' (type: {type.value})[/green]")
    except ValueError as e:
        _console().print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@directory_app.command("remove")
def directory_remove(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Directory path to remove"),
) -> None:
    """Remove a directory entry from fcw.yaml."""
    config_path = _resolve_config_path(ctx)
    try:
        remove_directory_from_config(config_path, path)
        _console().print(f"[green]Removed directory '{path}'[/green]")
    except ValueError as e:
        _console().print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@directory_app.command("list")
def directory_list(ctx: typer.Context) -> None:
    """List configured directories."""
    config_path = _resolve_config_path(ctx)
    config = load_config(str(config_path))
    if not config.directories:
        _console().print("[yellow]No directories configured.[/yellow]")
        return
    table = Table(show_header=True)
    table.add_column("Path")
    table.add_column("Type")
    for path, dir_config in config.directories.items():
        table.add_row(path, dir_config.type.value)
    _console().print(table)


# ---------------------------------------------------------------------------
# Container sub-commands
# ---------------------------------------------------------------------------


@container_app.command("add")  # FIXME: What about optional params build_args, local_stages, remote_stage?
def container_add(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Container name"),
    file: str = typer.Option(..., "--file", "-f", help="Dockerfile path"),
    tag: str = typer.Option(..., "--tag", "-t", help="Image tag"),
    remote_path: Optional[str] = typer.Option(None, "--remote-path", help="Remote image directory"),
    toml: Optional[str] = typer.Option(None, "--toml", help="Container TOML path"),
    platform: Optional[str] = typer.Option(None, "--platform", help="Target platform"),
) -> None:
    """Add a container entry to fcw.yaml."""
    config_path = _resolve_config_path(ctx)
    container = ContainerConfig(
        file=file,
        tag=tag,
        remote_path=remote_path,
        toml=toml,
        platform=platform,
    )
    try:
        add_container_to_config_roundtrip(config_path, name, container)
        _console().print(f"[green]Added container '{name}' (tag: {tag})[/green]")
    except ValueError as e:
        _console().print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@container_app.command("remove")
def container_remove(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Container name to remove"),
) -> None:
    """Remove a container entry from fcw.yaml."""
    config_path = _resolve_config_path(ctx)
    try:
        remove_container_from_config(config_path, name)
        _console().print(f"[green]Removed container '{name}'[/green]")
    except ValueError as e:
        _console().print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@container_app.command("list")  # FIXME: this doesn't list download image(s), just the final remote image
def container_list(ctx: typer.Context) -> None:
    """List configured containers."""
    config_path = _resolve_config_path(ctx)
    config = load_config(str(config_path))
    if not config.containers:
        _console().print("[yellow]No containers configured.[/yellow]")
        return
    table = Table(show_header=True)
    table.add_column("Name")
    table.add_column("Tag")
    table.add_column("Remote Path")
    for name, cont_config in config.containers.items():
        table.add_row(name, cont_config.tag, cont_config.remote_path or "-")
    _console().print(table)


# ---------------------------------------------------------------------------
# Job sub-commands
# ---------------------------------------------------------------------------


@job_app.command("add")
def job_add(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Job name"),
    script: str = typer.Option(..., "--script", "-s", help="SLURM script path"),
    container: Optional[str] = typer.Option(None, "--container", "-c", help="Container name"),
    env: Optional[List[str]] = typer.Option(None, "--env", "-e", help="Env var: KEY=VALUE"),
    time: Optional[str] = typer.Option(None, "--time", help="SBATCH time limit"),
    nodes: Optional[int] = typer.Option(None, "--nodes", help="SBATCH node count"),
    gpus_per_node: Optional[int] = typer.Option(None, "--gpus-per-node", help="SBATCH GPUs/node"),
    cpus_per_task: Optional[int] = typer.Option(None, "--cpus-per-task", help="SBATCH CPUs/task"),
) -> None:
    """Add a job entry to fcw.yaml."""
    config_path = _resolve_config_path(ctx)
    env_dict: dict[str, str] = {}
    if env:
        for item in env:
            if "=" not in item:
                _console().print(f"[red]Invalid --env format: '{item}' (expected KEY=VALUE)[/red]")
                raise typer.Exit(1)
            k, v = item.split("=", 1)
            env_dict[k] = v
    job = JobConfig(
        script=script,
        container=container,
        env=env_dict,
        time=time,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        cpus_per_task=cpus_per_task,
    )
    try:
        add_job_to_config(config_path, name, job)
        _console().print(f"[green]Added job '{name}' (script: {script})[/green]")
    except ValueError as e:
        _console().print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@job_app.command("remove")
def job_remove(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Job name to remove"),
) -> None:
    """Remove a job entry from fcw.yaml."""
    config_path = _resolve_config_path(ctx)
    try:
        remove_job_from_config(config_path, name)
        _console().print(f"[green]Removed job '{name}'[/green]")
    except ValueError as e:
        _console().print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@job_app.command("list")
def job_list(ctx: typer.Context) -> None:
    """List configured jobs."""
    config_path = _resolve_config_path(ctx)
    config = load_config(str(config_path))
    if not config.jobs:
        _console().print("[yellow]No jobs configured.[/yellow]")
        return
    table = Table(show_header=True)
    table.add_column("Name")
    table.add_column("Script")
    table.add_column("Container")
    table.add_column("Time")
    for name, job_config in config.jobs.items():
        table.add_row(
            name,
            job_config.script,
            job_config.container or "-",
            str(job_config.time) if job_config.time is not None else "-",
        )
    _console().print(table)


@app.command()
def show(
    ctx: typer.Context,
):
    """Display resolved configuration."""
    try:
        config = load_config((ctx.obj or {}).get("config_file"))
    except FileNotFoundError:
        _console().print("[yellow]No config file found. Using defaults.[/yellow]")
        _console().print("Run 'fcw config init' to create a config file.")
        return

    _console().print(f"[bold]Project:[/bold] {config.project}")
    if config._config_path:
        _console().print(f"[bold]Config file:[/bold] {config._config_path}")

    _console().print("\n[bold]Workdir:[/bold]")
    _console().print(f"  remote: {config.workdir.remote}")
    _console().print(f"  local:  {config.workdir.local}")

    if config.directories:
        _console().print("\n[bold]Directories:[/bold]")
        table = Table(show_header=True)
        table.add_column("Path")
        table.add_column("Type")
        for path, dir_config in config.directories.items():
            table.add_row(path, dir_config.type.value)
        _console().print(table)

    if config.containers:
        _console().print("\n[bold]Containers:[/bold]")
        table = Table(show_header=True)
        table.add_column("Name")
        table.add_column("Tag")
        table.add_column("Remote Path")
        for name, cont_config in config.containers.items():
            table.add_row(name, cont_config.tag, cont_config.remote_path or "-")
        _console().print(table)

    if config.jobs:
        _console().print("\n[bold]Jobs:[/bold]")
        table = Table(show_header=True)
        table.add_column("Name")
        table.add_column("Script")
        for name, job_config in config.jobs.items():
            table.add_row(name, job_config.script)
        _console().print(table)


# ---------------------------------------------------------------------------
# Validate helpers
# ---------------------------------------------------------------------------

def _ok(msg: str) -> None:
    _console().print(f"  [green]✓[/green] {msg}")


def _warn(warnings: list[str], msg: str) -> None:
    warnings.append(msg)


def _err(errors: list[str], msg: str) -> None:
    errors.append(msg)


def _human_time_ago(ts: float) -> str:
    """Format a timestamp as a human-readable 'X ago' string."""
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    return f"{int(delta / 86400)}d ago"


def _validate_local(ctx: typer.Context, errors: list[str], warnings: list[str]):
    """Phase 1: local-only checks. Returns (config, system_name) or raises Exit."""
    con = _console()
    con.print("\n[bold]── Local checks ──────────────────[/bold]")

    # 1. Config file
    config = None
    try:
        config = load_config((ctx.obj or {}).get("config_file"))
        if config._config_path:
            _ok(f"Config file: {config._config_path}")
        else:
            _warn(warnings, "No config file found (using defaults)")
    except Exception as e:
        _err(errors, f"Config file error: {e}")
        return None

    # 2. Required env vars
    required_vars = [
        "FIRECREST_URL",
        "FIRECREST_CLIENT_ID",
        "FIRECREST_CLIENT_SECRET",
        "AUTH_TOKEN_URL",
    ]
    for var in required_vars:
        if os.environ.get(var):
            _ok(f"{var} is set")
        else:
            _err(errors, f"Missing required: {var}")

    # 3. Optional env vars (informational)
    optional_vars = [
        "FIRECREST_SYSTEM",
        "FIRECREST_ACCOUNT",
        "FIRECREST_USER",
        "FIRECREST_HOME",
        "FIRECREST_SCRATCH",
        "FIRECREST_RESERVATION",
    ]
    for var in optional_vars:
        if os.environ.get(var):
            _ok(f"{var} is set")
        else:
            _warn(warnings, f"Optional not set: {var}")

    # 4. Job script files exist locally
    if config._config_path:
        for name, job_cfg in config.jobs.items():
            if job_cfg.script:
                if os.path.exists(job_cfg.script):
                    _ok(f"Job '{name}' script exists: {job_cfg.script}")
                else:
                    _warn(warnings, f"Job '{name}' script not found: {job_cfg.script}")

    # 5. Container Dockerfiles exist locally
    if config._config_path:
        for name, cont_cfg in config.containers.items():
            if cont_cfg.file:
                if os.path.exists(cont_cfg.file):
                    _ok(f"Container '{name}' Dockerfile exists: {cont_cfg.file}")
                else:
                    _warn(warnings, f"Container '{name}' Dockerfile not found: {cont_cfg.file}")

    # 6. Data directories exist locally
    if config._config_path:
        for dir_path in config.directories:
            local = config.resolve_path(dir_path, remote=False)
            if os.path.isdir(local):
                _ok(f"Directory '{dir_path}' exists locally")
            else:
                _warn(warnings, f"Directory '{dir_path}' not found locally: {local}")

    return config


def _validate_remote(
    ctx: typer.Context,
    config,
    errors: list[str],
    warnings: list[str],
    diff: bool = False,
) -> None:
    """Phase 2: remote checks (connectivity, env var cross-validation, containers, data)."""
    con = _console()
    con.print("\n[bold]── Remote checks ─────────────────[/bold]")

    # Bail early if required env vars are missing
    if any("Missing required" in e for e in errors):
        con.print("  [dim]Skipped (missing required env vars)[/dim]")
        return

    # 1. Connectivity + system check
    try:
        client = get_client()
        system = get_system((ctx.obj or {}).get("system"))
        systems = client.systems()
        if any(s["name"] == system for s in systems):
            _ok(f"Connected to FirecREST (system: {system})")
        else:
            _err(errors, f"System '{system}' not found in FirecREST")
            return
    except Exception as e:
        _err(errors, f"Connection failed: {e}")
        return

    # 2. Cross-validate optional env vars against remote state
    _validate_env_vars_remote(client, system, warnings)

    # 3. Container images on remote
    if config and config.containers:
        _validate_containers_remote(client, system, config, warnings)

    # 4. Data directory existence + sync status
    if config and config.directories:
        _validate_directories_remote(client, system, config, warnings, diff)


def _validate_env_vars_remote(
    client, system: str, warnings: list[str]
) -> None:
    """Cross-validate FIRECREST_* env vars against the remote system."""
    # Fetch userinfo
    userinfo = None
    try:
        userinfo = client.userinfo(system)
    except Exception:
        _warn(warnings, "Could not fetch userinfo from remote")

    # FIRECREST_USER — compare against remote username
    local_user = os.environ.get("FIRECREST_USER")
    if local_user and userinfo:
        # userinfo may have: {"user": {"id": ..., "name": ...}, ...}
        # or flat: {"name": ..., "user": ..., "username": ...}
        user_field = userinfo.get("user")
        if isinstance(user_field, dict):
            remote_user = user_field.get("name") or user_field.get("id")
        else:
            remote_user = userinfo.get("name") or user_field or userinfo.get("username")
        if remote_user and str(local_user) != str(remote_user):
            _warn(warnings, f"FIRECREST_USER='{local_user}' does not match remote user '{remote_user}'")
        elif remote_user:
            _ok(f"FIRECREST_USER matches remote user '{remote_user}'")

    # FIRECREST_ACCOUNT — check against user's groups
    local_account = os.environ.get("FIRECREST_ACCOUNT")
    if local_account and userinfo:
        groups = userinfo.get("groups", [])
        # groups may be list of dicts with "name" key, or list of strings
        group_names = set()
        for g in groups:
            if isinstance(g, dict):
                group_names.add(g.get("name", ""))
                group_names.add(g.get("group", ""))
            else:
                group_names.add(str(g))
        group_names.discard("")
        if group_names and local_account not in group_names:
            _warn(warnings, f"FIRECREST_ACCOUNT='{local_account}' not found in user groups: {', '.join(sorted(group_names))}")
        elif group_names:
            _ok(f"FIRECREST_ACCOUNT '{local_account}' matches user groups")

    # FIRECREST_HOME — check path exists and matches remote
    local_home = os.environ.get("FIRECREST_HOME")
    if local_home:
        # Cross-check against userinfo if available
        if userinfo:
            remote_home = userinfo.get("home") or userinfo.get("homeDirectory")
            if remote_home and local_home != remote_home:
                _warn(warnings, f"FIRECREST_HOME='{local_home}' does not match remote home '{remote_home}'")
            elif remote_home:
                _ok(f"FIRECREST_HOME matches remote home '{remote_home}'")
        # Check path exists
        try:
            async_client = get_async_client()
            asyncio.run(async_client.list_files(system_name=system, path=local_home))
            _ok(f"FIRECREST_HOME path exists on remote: {local_home}")
        except Exception:
            _warn(warnings, f"FIRECREST_HOME path not accessible on remote: {local_home}")

    # FIRECREST_SCRATCH — check path exists and matches remote
    local_scratch = os.environ.get("FIRECREST_SCRATCH")
    if local_scratch:
        # Cross-check against userinfo if available
        if userinfo:
            remote_scratch = userinfo.get("scratch") or userinfo.get("scratchDirectory")
            if remote_scratch and local_scratch != remote_scratch:
                _warn(warnings, f"FIRECREST_SCRATCH='{local_scratch}' does not match remote scratch '{remote_scratch}'")
            elif remote_scratch:
                _ok(f"FIRECREST_SCRATCH matches remote scratch '{remote_scratch}'")
        # Check path exists
        try:
            async_client = get_async_client()
            asyncio.run(async_client.list_files(system_name=system, path=local_scratch))
            _ok(f"FIRECREST_SCRATCH path exists on remote: {local_scratch}")
        except Exception:
            _warn(warnings, f"FIRECREST_SCRATCH path not accessible on remote: {local_scratch}")

    # FIRECREST_RESERVATION — check reservation exists
    local_reservation = os.environ.get("FIRECREST_RESERVATION")
    if local_reservation:
        try:
            reservations = client.reservations(system)
            res_names = {
                r.get("ReservationName") or r.get("reservation_name", "")
                for r in reservations
            }
            if local_reservation in res_names:
                _ok(f"FIRECREST_RESERVATION '{local_reservation}' exists on remote")
            else:
                available = ", ".join(sorted(res_names - {""})) or "(none)"
                _warn(warnings, f"FIRECREST_RESERVATION='{local_reservation}' not found. Available: {available}")
        except Exception:
            _warn(warnings, "Could not check reservations on remote")


def _validate_containers_remote(
    client, system: str, config, warnings: list[str]
) -> None:
    """Check that container sqsh images exist on the remote system."""
    async_client = get_async_client()

    for name, cont_cfg in config.containers.items():
        sqsh_path = config.resolve_container_image(cont_cfg)
        images_dir = config.resolve_container_images_dir(cont_cfg)

        try:
            entries = asyncio.run(
                async_client.list_files(system_name=system, path=images_dir)
            )
            sqsh_name = os.path.basename(sqsh_path)
            found = False
            for entry in entries:
                ename = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
                if ename == sqsh_name:
                    found = True
                    break

            if found:
                _ok(f"Container '{name}' image found: {sqsh_name}")
            else:
                _warn(warnings, f"Container '{name}' image not found on remote: {sqsh_path}")
        except Exception:
            _warn(warnings, f"Container '{name}' images dir not accessible: {images_dir}")


def _validate_directories_remote(
    client, system: str, config, warnings: list[str], diff: bool = False
) -> None:
    """Check remote data directories: existence, sync staleness, optional diff."""
    from fcw.commands.data import _collect_local_files_since, _read_last_sync_timestamp

    async_client = get_async_client()

    for dir_path, dir_cfg in config.directories.items():
        remote_dir = config.resolve_path(dir_path, remote=True)

        # Check remote directory exists
        remote_exists = False
        remote_files = []
        try:
            entries = asyncio.run(
                async_client.list_files(system_name=system, path=remote_dir, recursive=diff)
            )
            remote_exists = True
            if diff:
                remote_files = [
                    e.get("name") if isinstance(e, dict) else getattr(e, "name", "")
                    for e in entries
                    if (e.get("type") if isinstance(e, dict) else getattr(e, "type", "")) != "d"
                ]
        except Exception:
            pass

        if not remote_exists:
            _warn(warnings, f"Directory '{dir_path}' not found on remote: {remote_dir}")
            continue

        # Sync staleness from markers
        push_ts = _read_last_sync_timestamp(dir_path, "push")
        pull_ts = _read_last_sync_timestamp(dir_path, "pull")

        status_parts = ["exists on remote"]
        if push_ts > 0:
            status_parts.append(f"last push {_human_time_ago(push_ts)}")
        if pull_ts > 0:
            status_parts.append(f"last pull {_human_time_ago(pull_ts)}")
        if push_ts == 0 and pull_ts == 0:
            status_parts.append("never synced")

        # Full diff: compare file counts
        if diff:
            local_dir = config.resolve_path(dir_path, remote=False)
            if os.path.isdir(local_dir):
                local_files = _collect_local_files_since(local_dir, 0)
                local_count = len(local_files)
                remote_count = len(remote_files)
                status_parts.append(f"{local_count} local / {remote_count} remote files")
            else:
                status_parts.append(f"{len(remote_files)} remote files (no local dir)")

        _ok(f"Directory '{dir_path}': {', '.join(status_parts)}")


# ---------------------------------------------------------------------------
# Main validate command
# ---------------------------------------------------------------------------

@app.command()
def validate(
    ctx: typer.Context,
    local: bool = typer.Option(False, "--local", help="Run local checks only (skip remote)"),
    diff: bool = typer.Option(False, "--diff", help="Show full file diff for data directories"),
):
    """Validate configuration, credentials, and remote state.

    Runs local checks (config, env vars, file references) first, then
    remote checks (connectivity, env var cross-validation, containers,
    data directory sync status).

    Use --local to skip remote checks (fast, offline).
    Use --diff to compare local vs remote file counts for data directories.
    """
    errors: list[str] = []
    warnings: list[str] = []

    config = _validate_local(ctx, errors, warnings)

    if not local and config:
        _validate_remote(ctx, config, errors, warnings, diff=diff)

    # Print summary
    con = _console()
    if warnings:
        con.print("\n[yellow]Warnings:[/yellow]")
        for w in warnings:
            con.print(f"  [yellow]![/yellow] {w}")

    if errors:
        con.print("\n[red]Errors:[/red]")
        for e in errors:
            con.print(f"  [red]✗[/red] {e}")
        raise typer.Exit(1)
    else:
        con.print("\n[green]All checks passed![/green]")
