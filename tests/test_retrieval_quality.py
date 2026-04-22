"""MVP retrieval-quality gate.

Thin wrapper over ``dikw_core.eval.runner`` — the same machinery behind
``dikw eval --dataset mvp`` so the pytest gate and the CLI can never drift.
Data + thresholds live in ``evals/datasets/mvp/``; this file only asserts
"pass at the configured thresholds".
"""

from __future__ import annotations

import pytest

from dikw_core.eval.dataset import load_dataset
from dikw_core.eval.runner import run_eval


@pytest.mark.asyncio
async def test_mvp_retrieval_quality_meets_thresholds() -> None:
    spec = load_dataset("mvp")
    report = await run_eval(spec)
    assert report.passed, "\n" + report.diagnostics_table()
