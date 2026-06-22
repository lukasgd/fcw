"""Logging configuration for the fcw CLI.

Diagnostics go to stderr (consistent with `get_error_console`), leaving stdout
clean for machine-readable output. Verbosity is controlled by the `-v`/`-vv`
count flag on the root callback, or by the `FCW_LOG_LEVEL` env var (overrides).

The configuration owns the *whole* diagnostic stream: the `fcw` logger and
pyfirecrest's `firecrest` logger are driven in lockstep on a single handler, so
one `-v`/`-vv` knob surfaces both fcw's orchestration logs and pyfirecrest's own
request/transfer-job heartbeat (the ground truth when a remote op hangs).
"""

from __future__ import annotations

import logging
import os
import sys

_HANDLER_TAG = "_fcw_handler"

# Loggers driven in lockstep by the verbosity knob. pyfirecrest logs under the
# `firecrest.*` tree, so raising it here surfaces its per-request + wait_for_job
# heartbeat. httpx/httpcore are intentionally left at root WARNING (too noisy;
# the firecrest heartbeat already localizes a hang).
_MANAGED_LOGGERS = ("fcw", "firecrest")

# verbosity count -> level (clamped: 2+ -> DEBUG)
_LEVELS = {0: logging.WARNING, 1: logging.INFO}


def configure_logging(verbosity: int = 0) -> None:
    """Configure fcw + pyfirecrest logging.

    verbosity: 0 -> WARNING, 1 -> INFO, 2+ -> DEBUG. `FCW_LOG_LEVEL` (e.g.
    `INFO`, `DEBUG`) overrides the count when set. Idempotent: repeated calls
    (e.g. across `CliRunner` invocations in tests) never stack handlers.

    The stderr handler is attached to the root logger so records from any
    managed logger reach it, while the root level stays at WARNING so unrelated
    third-party loggers remain quiet. Visibility is therefore controlled per
    logger via `_MANAGED_LOGGERS`, not by the root level.
    """
    level = _LEVELS.get(verbosity, logging.DEBUG)
    env = os.environ.get("FCW_LOG_LEVEL")
    if env:
        resolved = logging.getLevelName(env.strip().upper())
        if isinstance(resolved, int):
            level = resolved

    root = logging.getLogger()
    if not any(getattr(h, _HANDLER_TAG, False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        setattr(handler, _HANDLER_TAG, True)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)

    for name in _MANAGED_LOGGERS:
        logging.getLogger(name).setLevel(level)
