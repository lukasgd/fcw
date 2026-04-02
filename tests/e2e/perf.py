"""Performance timing utilities for e2e tests."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import yaml


@dataclass
class StepTiming:
    """Recorded timing for a single e2e step."""

    test_name: str
    step_name: str
    elapsed_s: float
    threshold_s: float | None = None


class TimingCollector:
    """Session-level collector for all step timings."""

    def __init__(self) -> None:
        self.timings: list[StepTiming] = []

    def record(self, timing: StepTiming) -> None:
        self.timings.append(timing)

    def violations(self) -> list[StepTiming]:
        """Return timings that exceed their threshold."""
        return [
            t
            for t in self.timings
            if t.threshold_s is not None and t.elapsed_s > t.threshold_s
        ]


@contextmanager
def timed_step(
    collector: TimingCollector,
    test_name: str,
    step_name: str,
    threshold_s: float | None = None,
) -> Generator[None, None, None]:
    """Time a block and record the result.

    Always prints elapsed time. Notes threshold status if one is set.
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start

    timing = StepTiming(
        test_name=test_name,
        step_name=step_name,
        elapsed_s=elapsed,
        threshold_s=threshold_s,
    )
    collector.record(timing)

    status = ""
    if threshold_s is not None:
        if elapsed > threshold_s:
            status = f"  ** EXCEEDED threshold {threshold_s:.0f}s **"
        else:
            status = f"  (threshold: {threshold_s:.0f}s)"
    print(f"E2E PERF: {step_name}: {elapsed:.1f}s{status}")


def load_thresholds(config_path: str, system: str) -> dict[str, float]:
    """Load thresholds for a given system, falling back to defaults."""
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    defaults = data.get("default", {})
    system_specific = data.get(system, {})

    return {**defaults, **system_specific}
