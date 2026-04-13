"""FirecREST client initialization."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

import firecrest

if TYPE_CHECKING:
    from firecrest.v2 import Firecrest, AsyncFirecrest


def _get_auth() -> firecrest.ClientCredentialsAuth:
    """Get FirecREST authentication from environment."""
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
    
    return firecrest.ClientCredentialsAuth(client_id, client_secret, token_uri)


def _get_firecrest_url() -> str:
    """Get FirecREST API URL from environment."""
    url = os.environ.get("FIRECREST_URL")
    if not url:
        raise ValueError(
            "Missing required environment variable: FIRECREST_URL\n"
            "Set this variable or use 'fcw config validate' to check your setup."
        )
    return url


@lru_cache(maxsize=1)  # FIXME: can you explain how the client is reused across calls? 
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
