"""Tests for core shared utilities."""

import logging

import pytest

from fcw.core import (
    SLURM_FAILED_STATES,
    configure_logging,
    format_sbatch_lines,
    get_error_console,
    get_global_sbatch_options,
    get_output_console,
)


class TestSlurmFailedStates:
    def test_contains_expected_states(self):
        assert "FAILED" in SLURM_FAILED_STATES
        assert "CANCELLED" in SLURM_FAILED_STATES
        assert "TIMEOUT" in SLURM_FAILED_STATES

    def test_is_frozenset(self):
        assert isinstance(SLURM_FAILED_STATES, frozenset)

    def test_completed_not_failed(self):
        assert "COMPLETED" not in SLURM_FAILED_STATES


class TestGetGlobalSbatchOptions:
    _all_vars = (
        "FIRECREST_RESERVATION",
        "FIRECREST_PARTITION",
        "FIRECREST_NODELIST",
        "FIRECREST_EXCLUDE",
    )

    def _clear(self, monkeypatch):
        for var in self._all_vars:
            monkeypatch.delenv(var, raising=False)

    def test_empty_when_no_env(self, monkeypatch):
        self._clear(monkeypatch)
        assert get_global_sbatch_options() == {}

    def test_reservation_from_env(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("FIRECREST_RESERVATION", "my-reservation")
        opts = get_global_sbatch_options()
        assert opts == {"reservation": "my-reservation"}

    def test_partition_from_env(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("FIRECREST_PARTITION", "debug")
        opts = get_global_sbatch_options()
        assert opts == {"partition": "debug"}

    def test_nodelist_from_env(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("FIRECREST_NODELIST", "nid001234")
        opts = get_global_sbatch_options()
        assert opts == {"nodelist": "nid001234"}

    def test_exclude_from_env(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("FIRECREST_EXCLUDE", "nid00[5-9]")
        opts = get_global_sbatch_options()
        assert opts == {"exclude": "nid00[5-9]"}


class TestFormatSbatchLines:
    def test_empty_dict(self):
        assert format_sbatch_lines({}) == ""

    def test_single_option(self):
        result = format_sbatch_lines({"reservation": "test"})
        assert result == "#SBATCH --reservation=test\n"

    def test_multiple_options(self):
        result = format_sbatch_lines({"reservation": "test", "partition": "gpu"})
        assert "#SBATCH --reservation=test\n" in result
        assert "#SBATCH --partition=gpu\n" in result


class TestPollCadence:
    """fcw replaces pyfirecrest's exponential backoff (sleep_generator) with a
    tight, bounded poll cadence so completion is detected within a few seconds."""

    def test_intervals_bounded_by_default_cap(self, monkeypatch):
        from fcw.core.client import _fcw_poll_intervals

        monkeypatch.delenv("FCW_POLL_MAX", raising=False)
        gen = _fcw_poll_intervals()
        values = [next(gen) for _ in range(20)]
        assert values[0] <= 0.5
        assert all(v <= 3.0 + 1e-9 for v in values)  # never exceeds default cap
        assert max(values) == 3.0  # the ramp does reach the cap (not stuck low)

    def test_cap_is_env_overridable(self, monkeypatch):
        from fcw.core.client import _fcw_poll_intervals

        monkeypatch.setenv("FCW_POLL_MAX", "1.5")
        gen = _fcw_poll_intervals()
        values = [next(gen) for _ in range(20)]  # span the full ramp
        assert all(v <= 1.5 + 1e-9 for v in values)

    def test_floor_prevents_busy_loop(self, monkeypatch):
        from fcw.core.client import _fcw_poll_intervals

        monkeypatch.setenv("FCW_POLL_MAX", "0")  # pathological; must not busy-loop
        gen = _fcw_poll_intervals()
        assert all(next(gen) >= 0.2 for _ in range(5))

    def test_install_rebinds_pyfirecrest_sleep_generator(self):
        import firecrest.v2._async.Client as async_mod
        import firecrest.v2._sync.Client as sync_mod

        from fcw.core.client import _fcw_poll_intervals, _install_fast_polling

        _install_fast_polling()
        assert sync_mod.sleep_generator is _fcw_poll_intervals
        assert async_mod.sleep_generator is _fcw_poll_intervals


class TestGetAuthCaching:
    """The async client builds a fresh client per command; the auth object is
    cached so its OAuth token is reused instead of re-fetched every call."""

    def test_returns_same_instance(self, monkeypatch):
        from fcw.core import client as client_mod

        monkeypatch.setenv("FIRECREST_CLIENT_ID", "id")
        monkeypatch.setenv("FIRECREST_CLIENT_SECRET", "secret")
        monkeypatch.setenv("AUTH_TOKEN_URL", "https://example.invalid/token")
        client_mod._get_auth.cache_clear()
        try:
            assert client_mod._get_auth() is client_mod._get_auth()
        finally:
            client_mod._get_auth.cache_clear()  # don't leak cached auth to other tests


class TestConsoles:
    """Diagnostics go to stderr; primary command output goes to stdout."""

    def test_console_is_stderr(self):
        assert get_error_console().stderr is True

    def test_output_console_is_stdout(self):
        assert get_output_console().stderr is False

    def test_streams_are_separated(self, capsys):
        # Rich resolves the stream at print time, so capsys captures correctly.
        get_output_console().print("PRIMARY_OUT")
        get_error_console().print("DIAG_ERR")
        captured = capsys.readouterr()
        assert "PRIMARY_OUT" in captured.out
        assert "PRIMARY_OUT" not in captured.err
        assert "DIAG_ERR" in captured.err
        assert "DIAG_ERR" not in captured.out


class TestConfigureLogging:
    """One knob drives fcw + pyfirecrest in lockstep on a root handler; FCW_LOG_LEVEL
    overrides; unrelated loggers (httpx) stay quiet; handlers don't stack."""

    @pytest.fixture(autouse=True)
    def _restore_loggers(self):
        """Snapshot/restore the root, `fcw`, and `firecrest` loggers so tests don't leak."""
        names = [None, "fcw", "firecrest"]  # None -> root
        saved = [(logging.getLogger(n), logging.getLogger(n).handlers[:], logging.getLogger(n).level)
                 for n in names]
        root = logging.getLogger()
        root.handlers = [h for h in root.handlers if not getattr(h, "_fcw_handler", False)]
        yield
        for logger, handlers, level in saved:
            logger.handlers = handlers
            logger.setLevel(level)

    def test_fcw_and_firecrest_track_verbosity(self):
        for verbosity, expected in [(0, logging.WARNING), (1, logging.INFO), (2, logging.DEBUG), (3, logging.DEBUG)]:
            configure_logging(verbosity)
            assert logging.getLogger("fcw").level == expected
            assert logging.getLogger("firecrest").level == expected

    def test_handler_on_root_and_root_level_unchanged(self):
        configure_logging(2)
        root = logging.getLogger()
        tagged = [h for h in root.handlers if getattr(h, "_fcw_handler", False)]
        assert len(tagged) == 1
        # Root stays quiet so unrelated third-party loggers aren't surfaced.
        assert root.level in (logging.WARNING, logging.NOTSET)

    def test_env_var_overrides_verbosity(self, monkeypatch):
        monkeypatch.setenv("FCW_LOG_LEVEL", "debug")
        configure_logging(0)  # would be WARNING without the override
        assert logging.getLogger("fcw").level == logging.DEBUG
        assert logging.getLogger("firecrest").level == logging.DEBUG

    def test_firecrest_surfaced_but_httpx_quiet_at_debug(self):
        """The integration contract: at -vv pyfirecrest's stream is enabled (so its
        records reach the root handler) while httpx stays gated by root WARNING."""
        configure_logging(2)  # -vv -> DEBUG
        assert logging.getLogger("firecrest.v2._async.Client").isEnabledFor(logging.DEBUG)
        assert not logging.getLogger("httpx").isEnabledFor(logging.DEBUG)

    def test_no_duplicate_handlers(self):
        configure_logging(1)
        configure_logging(2)
        configure_logging(0)
        root = logging.getLogger()
        tagged = [h for h in root.handlers if getattr(h, "_fcw_handler", False)]
        assert len(tagged) == 1
