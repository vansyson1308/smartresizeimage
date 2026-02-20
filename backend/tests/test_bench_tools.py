"""Tests for benchmark fixture generator and metrics pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from backend.app.enums import ElementRole
from backend.app.layout.bench_metrics import evaluate_bench_run
from backend.app.models import BoundingBox, DesignElement, LayoutResult
from backend.tools.generate_bench_fixtures import generate_fixtures


def test_generate_bench_fixtures_outputs_valid_files(tmp_path: Path) -> None:
    names = generate_fixtures(tmp_path, cases=3, seed=11)
    assert len(names) == 3

    for name in names:
        case_dir = tmp_path / name
        assert (case_dir / "input.png").exists()
        assert (case_dir / "metadata.json").exists()

        meta = json.loads((case_dir / "metadata.json").read_text())
        assert meta["case_name"] == name
        assert len(meta["elements"]) >= 5

        img = Image.open(case_dir / "input.png")
        assert img.size == (
            meta["source_size"]["width"],
            meta["source_size"]["height"],
        )


def test_bench_metrics_eval_runs_on_tiny_case() -> None:
    elements = [
        DesignElement(
            id="bg",
            name="bg",
            layer_type="pixel",
            bbox=BoundingBox(0, 0, 200, 100),
            role=ElementRole.BACKGROUND,
            priority=9,
        ),
        DesignElement(
            id="headline",
            name="headline",
            layer_type="type",
            bbox=BoundingBox(10, 10, 90, 24),
            role=ElementRole.HEADLINE,
            text_content="Hello",
            priority=1,
        ),
        DesignElement(
            id="hero",
            name="hero",
            layer_type="pixel",
            bbox=BoundingBox(110, 20, 80, 70),
            role=ElementRole.HERO_IMAGE,
            priority=2,
        ),
    ]
    layout = [
        LayoutResult("bg", BoundingBox(0, 0, 200, 100), 1.0),
        LayoutResult("headline", BoundingBox(12, 10, 94, 26), 1.0),
        LayoutResult("hero", BoundingBox(110, 20, 80, 70), 1.0),
    ]

    result = evaluate_bench_run(
        elements=elements,
        layout_results=layout,
        target_size=(200, 100),
        text_plate_meta={"applied": True},
        busy_expected=True,
    )

    assert isinstance(result.passed, bool)
    assert result.metrics.total_score > -10000
    assert 0.0 <= result.metrics.overlap_area_ratio <= 1.0
