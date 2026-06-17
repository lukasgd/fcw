"""Main CLI application."""

import typer
from typing import Optional

from fcw import __version__
from fcw.commands import config, data, job, container, mount

app = typer.Typer(
    name="fcw",
    help="FirecREST Container Workflows - a CLI for remote HPC job orchestration.",
    no_args_is_help=True,
    epilog="Get started: fcw config init && fcw config validate\nTab completion: fcw --install-completion",
)

# Register command groups
app.add_typer(config.app, name="config", help="Configuration management")
app.add_typer(data.app, name="data", help="Transfer data with type enforcement")
app.add_typer(job.app, name="job", help="Submit jobs and job chains")
app.add_typer(container.app, name="container", help="Build and manage container images")
app.add_typer(mount.app, name="mount", help="FUSE filesystem operations")

# Promote frequently-used job verbs to the top level: `fcw submit` == `fcw job submit`.
# The `job` group remains the canonical home for the full verb set.
_PROMOTED_JOB_VERBS = {"submit", "run", "logs", "wait", "cancel"}
for _cmd in job.app.registered_commands:
    if _cmd.name in _PROMOTED_JOB_VERBS:
        app.registered_commands.append(_cmd)


@app.callback()
def main(
    ctx: typer.Context,
    system: Optional[str] = typer.Option(
        None, "--system", "-s", envvar="FIRECREST_SYSTEM",
        help="Target HPC system"
    ),
    account: Optional[str] = typer.Option(
        None, "--account", "-a", envvar="FIRECREST_ACCOUNT",
        help="SLURM account"
    ),
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Config file path (default: ./fcw.yaml)"
    ),
):
    """FirecREST Container Workflows - a CLI for remote HPC job orchestration."""
    # Store global options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["system"] = system
    ctx.obj["account"] = account
    ctx.obj["config_file"] = config_file


@app.command()
def version():
    """Show version information."""
    typer.echo(f"fcw version {__version__}")


if __name__ == "__main__":
    app()
