"""Tests for core shared utilities."""

from fcw.core import (
    SLURM_FAILED_STATES,
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
    def test_empty_when_no_env(self, monkeypatch):
        monkeypatch.delenv("FIRECREST_RESERVATION", raising=False)
        monkeypatch.delenv("FIRECREST_PARTITION", raising=False)
        assert get_global_sbatch_options() == {}

    def test_reservation_from_env(self, monkeypatch):
        monkeypatch.delenv("FIRECREST_PARTITION", raising=False)
        monkeypatch.setenv("FIRECREST_RESERVATION", "my-reservation")
        opts = get_global_sbatch_options()
        assert opts == {"reservation": "my-reservation"}

    def test_partition_from_env(self, monkeypatch):
        monkeypatch.delenv("FIRECREST_RESERVATION", raising=False)
        monkeypatch.setenv("FIRECREST_PARTITION", "debug")
        opts = get_global_sbatch_options()
        assert opts == {"partition": "debug"}


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
