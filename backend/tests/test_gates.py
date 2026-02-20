"""Tests for quality gates and debug logging."""

from __future__ import annotations

import numpy as np
from PIL import Image

from backend.app.enums import ElementRole
from backend.app.generative.gates import evaluate_quality_gates
from backend.app.models import BoundingBox, DesignElement, LayoutResult


def test_evaluate_quality_gates_fails_and_logs_reasons(caplog) -> None:
    base = Image.new("RGBA", (60, 40), (120, 120, 120, 255))
    cand = Image.new("RGBA", (60, 40), (200, 20, 20, 255))

    elements = [
        DesignElement(
            id="logo",
            name="logo",
            layer_type="pixel",
            bbox=BoundingBox(5, 5, 20, 12),
            image=Image.new("RGBA", (20, 12), (0, 0, 255, 255)),
            role=ElementRole.LOGO,
            priority=1,
        )
    ]
    layout = [LayoutResult("logo", BoundingBox(5, 5, 20, 12), 1.0)]

    protected = np.zeros((40, 60), dtype=bool)
    protected[5:17, 5:25] = True

    with caplog.at_level("WARNING", logger="autobanner.generative.gates"):
        report = evaluate_quality_gates(
            baseline=base,
            candidate=cand,
            elements=elements,
            layout_results=layout,
            protected_mask=protected,
            ocr_extractor=lambda _img: [],
        )

    assert report.gates_passed is False
    assert report.used_fallback is True
    assert "logo_similarity_failed" in report.fail_reasons
    assert "color_drift_failed" in report.fail_reasons
    assert "Quality gates failed" in caplog.text
