"""Core modules."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from rich.console import Console

from fcw.core.client import (
    extract_job_id,
    get_account,
    get_async_client,
    get_client,
    get_system,
)
from fcw.core.logging import configure_logging
from fcw.core.config import (
    ContainerConfig,
    DirectoryConfig,
    DirectoryType,
    FcwConfig,
    JobConfig,
    WorkdirConfig,
    add_container_to_config,
    add_container_to_config_roundtrip,
    add_directory_to_config,
    add_job_to_config,
    generate_default_config,
    generate_interactive_config,
    load_config,
    remove_container_from_config,
    remove_directory_from_config,
    remove_job_from_config,
)

if TYPE_CHECKING:
    import typer


# SLURM job states that indicate failure.
SLURM_FAILED_STATES = frozenset({
    "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
    "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "PREEMPTED",
})


def get_error_console() -> Console:
    """Get a Rich console for diagnostics (errors, warnings, status) on stderr.

    Keeping these off stdout leaves it clean for machine-readable output (e.g.
    a submitted job id), so `JOB=$(fcw job submit ...)` captures only that.
    """
    return Console(stderr=True)


def get_output_console() -> Console:
    """Get a Rich console for primary command output (tables, listings) on stdout.

    Used by commands whose result is meant to be piped/redirected, e.g.
    `fcw job list`, `fcw config show`, `fcw data ls`.
    """
    return Console()


def get_global_sbatch_options() -> dict[str, str]:
    """Get global SBATCH options from environment variables.

    Currently supports:
        FIRECREST_RESERVATION -> --reservation
        FIRECREST_PARTITION -> --partition
        FIRECREST_NODELIST -> --nodelist
        FIRECREST_EXCLUDE -> --exclude
    """
    opts: dict[str, str] = {}
    reservation = os.environ.get("FIRECREST_RESERVATION")
    partition = os.environ.get("FIRECREST_PARTITION")
    nodelist = os.environ.get("FIRECREST_NODELIST")
    exclude = os.environ.get("FIRECREST_EXCLUDE")
    if reservation:
        opts["reservation"] = reservation
    if partition:
        opts["partition"] = partition
    if nodelist:
        opts["nodelist"] = nodelist
    if exclude:
        opts["exclude"] = exclude
    return opts


def format_sbatch_lines(options: dict[str, str]) -> str:
    """Format SBATCH options dict as #SBATCH directive lines.

    Returns a string of #SBATCH lines (with trailing newline per line),
    or empty string if no options.
    """
    return "".join(f"#SBATCH --{k}={v}\n" for k, v in options.items())


def resolve_context(ctx: "typer.Context") -> tuple[FcwConfig, str, str]:
    """Extract config, system, and account from a Typer context.

    Returns:
        Tuple of (config, system, account).
    """
    obj = ctx.obj or {}
    config = load_config(obj.get("config_file"))
    system = get_system(obj.get("system"))
    account = get_account(obj.get("account"))
    return config, system, account


__all__ = [
    "FcwConfig",
    "DirectoryType",
    "DirectoryConfig",
    "ContainerConfig",
    "JobConfig",
    "WorkdirConfig",
    "load_config",
    "add_container_to_config",
    "add_container_to_config_roundtrip",
    "add_directory_to_config",
    "add_job_to_config",
    "generate_default_config",
    "generate_interactive_config",
    "remove_container_from_config",
    "remove_directory_from_config",
    "remove_job_from_config",
    "get_client",
    "get_async_client",
    "get_system",
    "get_account",
    "extract_job_id",
    "resolve_context",
    "SLURM_FAILED_STATES",
    "get_error_console",
    "get_output_console",
    "get_global_sbatch_options",
    "format_sbatch_lines",
    "configure_logging",
]
