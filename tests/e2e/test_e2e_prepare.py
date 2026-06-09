"""Producer slice for engine-less e2e runs (temporary).

Run on a machine WITH a container engine to build each container's local stages
and save them as per-stage tars, then copy the directory to the engine-less client:

    pytest tests/e2e/test_e2e_prepare.py --run-e2e --example node-burn --prepare-stage-tars ./tars

The engine-less client then consumes them for the full suite:

    pytest tests/e2e --run-e2e --example node-burn --stage-tars ./tars

TODO(temporary): remove together with the --stage-tars / --prepare-stage-tars
engine-less affordance.
"""

import pytest
from helpers import save_stage_tars

pytestmark = pytest.mark.e2e


def test_save_stage_tars(runner, request):
    """Build + save all local-stage tars for the selected example (needs an engine)."""
    out_dir = request.config.getoption("--prepare-stage-tars")
    if not out_dir:
        pytest.skip("--prepare-stage-tars not set (producer slice)")
    save_stage_tars(runner, out_dir)
