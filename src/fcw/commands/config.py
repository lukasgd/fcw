"""Config command group."""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from fcw.core import load_config, generate_default_config, get_client, get_system

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
):
    """Create fcw.yaml template in current directory."""
    config_path = Path.cwd() / "fcw.yaml"
    
    if config_path.exists() and not force:
        console.print(f"[red]Config file already exists: {config_path}[/red]")
        console.print("Use --force to overwrite.")
        raise typer.Exit(1)
    
    config_path.write_text(generate_default_config())
    console.print(f"[green]Created config file: {config_path}[/green]")
    console.print("\nEdit the file to configure your project, then run:")
    console.print("  fcw config validate")


@app.command()
def show(
    ctx: typer.Context,
):
    """Display resolved configuration."""
    try:
        config_file = ctx.obj.get("config_file") if ctx.obj else None
        config = load_config(config_file)
    except FileNotFoundError:
        console.print("[yellow]No config file found. Using defaults.[/yellow]")
        console.print("Run 'fcw config init' to create a config file.")
        return
    
    console.print(f"[bold]Project:[/bold] {config.project}")
    if config._config_path:
        console.print(f"[bold]Config file:[/bold] {config._config_path}")
    
    console.print(f"\n[bold]Workdir:[/bold]")
    console.print(f"  remote: {config.workdir.remote}")
    console.print(f"  local:  {config.workdir.local}")
    
    if config.directories:
        console.print(f"\n[bold]Directories:[/bold]")
        table = Table(show_header=True)
        table.add_column("Path")
        table.add_column("Type")
        for path, dir_config in config.directories.items():
            table.add_row(path, dir_config.type.value)
        console.print(table)
    
    if config.containers:
        console.print(f"\n[bold]Containers:[/bold]")
        table = Table(show_header=True)
        table.add_column("Name")
        table.add_column("Tag")
        table.add_column("Remote Path")
        for name, cont_config in config.containers.items():
            table.add_row(name, cont_config.tag, cont_config.remote_path or "-")
        console.print(table)
    
    if config.jobs:
        console.print(f"\n[bold]Jobs:[/bold]")
        table = Table(show_header=True)
        table.add_column("Name")
        table.add_column("Script")
        table.add_column("After")
        for name, job_config in config.jobs.items():
            after = ", ".join(job_config.after) if job_config.after else "-"
            table.add_row(name, job_config.script, after)
        console.print(table)


@app.command()
def validate(
    ctx: typer.Context,
):
    """Check credentials and remote connectivity."""
    errors = []
    warnings = []
    
    # Check config file
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    try:
        config = load_config(config_file)
        if config._config_path:
            console.print(f"[green]✓[/green] Config file: {config._config_path}")
        else:
            warnings.append("No config file found (using defaults)")
    except Exception as e:
        errors.append(f"Config file error: {e}")
    
    # Check environment variables
    import os
    required_vars = [
        "FIRECREST_URL",
        "FIRECREST_CLIENT_ID", 
        "FIRECREST_CLIENT_SECRET",
        "AUTH_TOKEN_URL",
    ]
    optional_vars = [
        "FIRECREST_SYSTEM",
        "FIRECREST_ACCOUNT",
    ]
    
    for var in required_vars:
        if os.environ.get(var):
            console.print(f"[green]✓[/green] {var} is set")
        else:
            errors.append(f"Missing required: {var}")
    
    for var in optional_vars:
        if os.environ.get(var):
            console.print(f"[green]✓[/green] {var} is set")
        else:
            warnings.append(f"Optional not set: {var}")
    
    # Try to connect
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()
    debugpy.breakpoint()
    
    if not any("FIRECREST" in e for e in errors):
        try:
            client = get_client()
            system = ctx.obj.get("system") if ctx.obj else None
            system = get_system(system)
            
            # Test connection by listing systems or getting parameters
            systems = client.systems()
            if filter(lambda s: s['name'] == system, systems):
                console.print(f"[green]✓[/green] Connected to FirecREST (system: {system})")
            else:
                errors.append(f"System '{system}' not found in FirecREST")
        except Exception as e:
            errors.append(f"Connection failed: {e}")
    
    # Print summary
    if warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in warnings:
            console.print(f"  [yellow]![/yellow] {w}")
    
    if errors:
        console.print("\n[red]Errors:[/red]")
        for e in errors:
            console.print(f"  [red]✗[/red] {e}")
        raise typer.Exit(1)
    else:
        console.print("\n[green]All checks passed![/green]")
