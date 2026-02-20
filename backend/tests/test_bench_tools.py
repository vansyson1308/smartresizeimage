"""Tests for benchmark fixture generator and metrics pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from backend.app.enums import ElementRole
from backend.app.layout.bench_metrics import evaluate_bench_run
from backend.app.models import BoundingBox, DesignElement, LayoutResult
from backend.tools.generate_bench_fixtures import generate_fixtures
from backend.tools.run_layout_bench import _run_one_mode


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


def test_layout_debug_contains_repair_and_fallback_fields(tmp_path: Path) -> None:
    elements = [
        DesignElement(
            id="bg",
            name="bg",
            layer_type="pixel",
            bbox=BoundingBox(0, 0, 200, 100),
            image=Image.new("RGBA", (200, 100), (240, 240, 240, 255)),
            role=ElementRole.BACKGROUND,
            priority=9,
        ),
        DesignElement(
            id="headline",
            name="headline",
            layer_type="type",
            bbox=BoundingBox(10, 8, 160, 42),
            image=Image.new("RGBA", (160, 42), (0, 0, 0, 0)),
            role=ElementRole.HEADLINE,
            text_content="Debug fields",
            priority=1,
        ),
    ]

    run = _run_one_mode(
        elements=elements,
        source_size=(200, 100),
        target_size=(1080, 1080),
        mode="phase21",
        out_dir=tmp_path,
        busy_expected=False,
    )

    debug = run["layout_debug"]
    assert "profile_name" in debug
    assert "repair_applied" in debug
    assert "repair_steps" in debug
    assert "fallback_used" in debug
    assert "fallback_reason" in debug
    assert "text_plate" in debug
    assert "busy_threshold" in debug["text_plate"]
    assert "redesign" in debug


def test_run_one_mode_phase3_emits_redesign_debug(tmp_path: Path) -> None:
    elements = [
        DesignElement(
            id="bg",
            name="bg",
            layer_type="pixel",
            bbox=BoundingBox(0, 0, 200, 100),
            image=Image.new("RGBA", (200, 100), (240, 240, 240, 255)),
            role=ElementRole.BACKGROUND,
            priority=9,
        ),
        DesignElement(
            id="logo",
            name="logo",
            layer_type="pixel",
            bbox=BoundingBox(10, 8, 40, 20),
            image=Image.new("RGBA", (40, 20), (220, 20, 20, 255)),
            role=ElementRole.LOGO,
            priority=1,
        ),
    ]

    run = _run_one_mode(
        elements=elements,
        source_size=(200, 100),
        target_size=(1080, 1080),
        mode="phase3",
        out_dir=tmp_path,
        busy_expected=False,
    )

    debug = run["layout_debug"]
    assert debug["mode"] == "phase3"
    assert "repair_applied" in debug
