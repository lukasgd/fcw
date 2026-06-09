"""FirecREST client initialization."""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Iterator

import firecrest

if TYPE_CHECKING:
    from firecrest.v2 import Firecrest, AsyncFirecrest


# ---------------------------------------------------------------------------
# Poll-cadence override
# ---------------------------------------------------------------------------
# pyfirecrest's wait_for_job() (and the transfer / extract / compress waits that
# call it) polls on a hardcoded exponential backoff (sleep_generator:
# 0.2→0.5→1→2→…→64s), so a job that actually finishes at ~35s isn't noticed until
# ~64s. We replace it with a tight, bounded cadence so completion is detected
# within FCW_POLL_MAX seconds. This speeds every --wait job, staged transfer, and
# extract/compress fallback, in the CLI and the e2e suite alike.

def _fcw_poll_intervals() -> Iterator[float]:
    """Bounded poll cadence replacing pyfirecrest's exponential backoff.

    Yields a short first interval, then a gentle ramp capped at FCW_POLL_MAX
    (default 3s, floored at 0.2s to avoid a busy loop) so the worst-case
    completion-detection overshoot stays small instead of growing to ~64s.
    """
    try:
        poll_max = float(os.environ.get("FCW_POLL_MAX", "3"))
    except ValueError:
        poll_max = 3.0
    poll_max = max(0.2, poll_max)
    yield min(0.5, poll_max)
    value = 1.0
    while True:
        yield min(value, poll_max)
        value *= 1.5


def _install_fast_polling() -> None:
    """Redirect pyfirecrest's module-level ``sleep_generator`` (sync + async).

    Both wait loops reference the bare module global ``sleep_generator``, so
    reassigning the module attribute redirects them in one place. Guarded: if a
    pyfirecrest upgrade moves/renames the symbol, we silently keep the library
    default rather than break.
    """
    for modpath in ("firecrest.v2._sync.Client", "firecrest.v2._async.Client"):
        try:
            module = importlib.import_module(modpath)
            setattr(module, "sleep_generator", _fcw_poll_intervals)
        except Exception:
            pass


_install_fast_polling()


@lru_cache(maxsize=1)  # reuse one auth object so its OAuth token cache is shared
def _get_auth() -> firecrest.ClientCredentialsAuth:
    """Get FirecREST authentication from environment.

    Cached so the async path (which builds a fresh client per command) reuses one
    token instead of fetching a new OAuth token on every data call. ``lru_cache``
    does not cache the ValueError raised when env vars are missing.
    """
    client_id = os.environ.get("FIRECREST_CLIENT_ID")
    client_secret = os.environ.get("FIRECREST_CLIENT_SECRET")
    token_uri = os.environ.get("AUTH_TOKEN_URL")
    
    if not all([client_id, client_secret, token_uri]):
        missing = []
        if not client_id:
            missing.append("FIRECREST_CLIENT_ID")
        if not client_secret:
            missing.append("FIRECREST_CLIENT_SECRET")
        if not token_uri:
            missing.append("AUTH_TOKEN_URL")
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Set these variables or use 'fcw config validate' to check your setup."
        )
    
    return firecrest.ClientCredentialsAuth(
        client_id, client_secret, token_uri, min_token_validity=60
    )


def _get_firecrest_url() -> str:
    """Get FirecREST API URL from environment."""
    url = os.environ.get("FIRECREST_URL")
    if not url:
        raise ValueError(
            "Missing required environment variable: FIRECREST_URL\n"
            "Set this variable or use 'fcw config validate' to check your setup."
        )
    return url


@lru_cache(maxsize=1)  # process-wide singleton: all callers share one client
def get_client() -> "Firecrest":
    """Get a synchronous FirecREST v2 client.
    
    The client is cached for reuse across calls.
    """
    return firecrest.v2.Firecrest(
        firecrest_url=_get_firecrest_url(),
        authorization=_get_auth(),
    )


def get_async_client() -> "AsyncFirecrest":
    """Get an asynchronous FirecREST v2 client.

    Note: A new client is created each time since async clients
    should be used within a single async context.
    """
    client = firecrest.v2.AsyncFirecrest(
        firecrest_url=_get_firecrest_url(),
        authorization=_get_auth(),
    )
    # Increase timeout for large uploads (container images)
    client.timeout = 300
    return client


def get_system(system: str | None = None) -> str:
    """Get the target system name.
    
    Args:
        system: Explicit system name, or None to use environment.
    
    Returns:
        System name.
    
    Raises:
        ValueError: If no system specified.
    """
    system = system or os.environ.get("FIRECREST_SYSTEM")
    if not system:
        raise ValueError(
            "No target system specified.\n"
            "Use --system option or set FIRECREST_SYSTEM environment variable."
        )
    return system


def extract_job_id(result: dict) -> str:
    """Extract job ID from a FirecREST submit response.

    The API returns the ID under different keys depending on version.
    """
    return result.get("jobId") or result.get("jobid") or result.get("job_id") or ""


def get_account(account: str | None = None) -> str:
    """Get the SLURM account.
    
    Args:
        account: Explicit account name, or None to use environment.
    
    Returns:
        Account name.
    
    Raises:
        ValueError: If no account specified.
    """
    account = account or os.environ.get("FIRECREST_ACCOUNT")
    if not account:
        raise ValueError(
            "No SLURM account specified.\n"
            "Use --account option or set FIRECREST_ACCOUNT environment variable."
        )
    return account
