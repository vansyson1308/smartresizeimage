"""Tests for the ablation harness bookkeeping (no full runs)."""

from __future__ import annotations

from backend.tools.run_ablations import CONFIGS, RunRecord, _wilson, build_report, summarize


def _rec(config: str, split: str, verdict: str, secs: float = 1.0, repairs: int = 0) -> RunRecord:
    return RunRecord(
        config=config,
        case="case_01",
        size="1080x1080",
        split=split,
        verdict=verdict,
        accepted=verdict == "accepted",
        elapsed_s=secs,
        repair_steps=repairs,
        family="f",
    )


def test_summarize_splits_tuning_and_holdout() -> None:
    records = [
        _rec("full", "tuning", "accepted", 1.0, 0),
        _rec("full", "tuning", "failed", 2.0, 2),
        _rec("full", "holdout", "needs_review", 3.0, 1),
    ]
    s = summarize(records)
    assert s["tuning"]["runs"] == 2 and s["tuning"]["accepted"] == 1
    assert s["tuning"]["acceptance_rate"] == 0.5 and s["tuning"]["mean_repair_steps"] == 1.0
    assert s["holdout"]["needs_review"] == 1 and s["holdout"]["acceptance_rate"] == 0.0
    assert s["all"]["runs"] == 3


def test_wilson_interval_is_sane() -> None:
    lo, hi = _wilson(44, 48)
    assert 0.80 < lo < 0.92 < hi <= 1.0
    assert _wilson(0, 0) == (0.0, 0.0)
    lo0, hi0 = _wilson(0, 12)
    assert lo0 == 0.0 and 0.2 < hi0 < 0.3


def test_report_lists_every_configuration() -> None:
    results = {
        name: [_rec(name, "tuning", "accepted"), _rec(name, "holdout", "failed")]
        for name in ("full", "zones")
    }
    meta = {
        "environment": {"python": "3.11"},
        "git_commit": "abc",
        "sizes": ["1080x1080"],
        "n_cases": 1,
        "holdout_from": 10,
    }
    report = build_report(results, meta)
    assert "| full | tuning |" in report and "| zones | holdout |" in report
    assert CONFIGS["zones"].planner == "zones"
    assert "Synthetic fixtures" in report
