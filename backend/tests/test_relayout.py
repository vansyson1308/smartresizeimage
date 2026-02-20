"""Tests for the ReLayoutEngine orchestrator."""

import os
import tempfile

import pytest
from PIL import Image

from backend.app.exceptions import ValidationError
from backend.app.relayout import ReLayoutEngine


class TestReLayoutEngine:
    def setup_method(self):
        self.engine = ReLayoutEngine(use_ai=False)

    def test_relayout_without_loading_raises(self):
        with pytest.raises(ValueError, match="No file loaded"):
            self.engine.relayout((500, 500))

    def test_load_png_file(self):
        img = Image.new("RGB", (200, 150), (255, 0, 0))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            analysis = self.engine.load_file(tmp_path)
            assert analysis["total_layers"] == 1
            assert analysis["size"] == (200, 150)
            assert len(self.engine.elements) == 1
        finally:
            os.unlink(tmp_path)

    def test_relayout_png_produces_output(self):
        img = Image.new("RGB", (200, 150), (0, 255, 0))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            self.engine.load_file(tmp_path)
            result = self.engine.relayout((500, 500))
            assert result.image.size == (500, 500)
            assert result.image.mode == "RGB"
        finally:
            os.unlink(tmp_path)

    def test_batch_relayout(self):
        img = Image.new("RGB", (100, 100), (0, 0, 255))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            self.engine.load_file(tmp_path)
            targets = [
                (500, 500, "Square"),
                (1080, 1920, "Story"),
            ]
            results = self.engine.batch_relayout(targets)
            assert "Square" in results
            assert "Story" in results
            assert results["Square"].image.size == (500, 500)
            assert results["Story"].image.size == (1080, 1920)
        finally:
            os.unlink(tmp_path)

    def test_relayout_invalid_dimensions_raises(self):
        img = Image.new("RGB", (100, 100))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            self.engine.load_file(tmp_path)
            with pytest.raises(ValidationError):
                self.engine.relayout((0, 500))
        finally:
            os.unlink(tmp_path)

    def test_update_element_role(self):
        img = Image.new("RGB", (100, 100))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            self.engine.load_file(tmp_path)
            elem_id = self.engine.elements[0].id
            success = self.engine.update_element_role(elem_id, "hero_image")
            assert success is True
            assert self.engine.elements[0].role.value == "hero_image"
        finally:
            os.unlink(tmp_path)

    def test_update_element_role_invalid_role(self):
        img = Image.new("RGB", (100, 100))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            self.engine.load_file(tmp_path)
            elem_id = self.engine.elements[0].id
            success = self.engine.update_element_role(elem_id, "invalid_role")
            assert success is False
        finally:
            os.unlink(tmp_path)

    def test_update_element_priority(self):
        img = Image.new("RGB", (100, 100))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            self.engine.load_file(tmp_path)
            elem_id = self.engine.elements[0].id
            success = self.engine.update_element_priority(elem_id, 1)
            assert success is True
            assert self.engine.elements[0].priority == 1
        finally:
            os.unlink(tmp_path)

    def test_get_preview_image_returns_image(self):
        img = Image.new("RGB", (100, 100))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            self.engine.load_file(tmp_path)
            preview = self.engine.get_preview_image()
            assert preview is not None
            assert preview.mode == "RGB"
        finally:
            os.unlink(tmp_path)

    def test_get_preview_without_loading(self):
        preview = self.engine.get_preview_image()
        assert preview is None


    def test_relayout_saves_generative_metadata(self, monkeypatch):
        from backend.app.config import Config
        from backend.app.enums import ElementRole
        from backend.app.models import BoundingBox, DesignElement

        class MockBackend:
            model_id = "mock-diffusion-v1"

            def is_available(self):
                return True

            def generate(self, base_canvas, editable_mask, seed):
                _ = editable_mask, seed
                return base_canvas

        monkeypatch.setattr(Config, "GENERATIVE_BG_ENABLED", True)
        monkeypatch.setattr(Config, "GENERATIVE_BG_POLICY", "BG_ONLY")
        monkeypatch.setattr(Config, "GENERATIVE_BG_SEED", 99)

        self.engine.generative_engine.backend = MockBackend()
        self.engine.source_size = (100, 100)
        self.engine.elements = [
            DesignElement(
                id="bg",
                name="bg",
                layer_type="pixel",
                bbox=BoundingBox(0, 0, 100, 100),
                image=Image.new("RGBA", (100, 100), (240, 240, 240, 255)),
                role=ElementRole.BACKGROUND,
                priority=9,
            ),
            DesignElement(
                id="logo",
                name="logo",
                layer_type="pixel",
                bbox=BoundingBox(10, 10, 20, 20),
                image=Image.new("RGBA", (20, 20), (0, 0, 255, 255)),
                role=ElementRole.LOGO,
                priority=1,
                z_index=1,
            ),
        ]

        result = self.engine.relayout((120, 120))
        assert "generative" in result.metadata
        assert result.metadata["generative"]["policy"] == "BG_ONLY"
        assert result.metadata["generative"]["seed"] == 99
        assert result.metadata["generative"]["model_id"] == "mock-diffusion-v1"


    def test_failed_quality_gate_triggers_fallback(self, monkeypatch):
        from backend.app.config import Config
        from backend.app.enums import ElementRole
        from backend.app.generative.gates import GateReport
        from backend.app.models import BoundingBox, DesignElement

        class MockBackend:
            model_id = "mock-diffusion-v1"

            def is_available(self):
                return True

            def generate(self, base_canvas, editable_mask, seed):
                _ = editable_mask, seed
                # Make visible change in editable area
                return Image.new("RGBA", base_canvas.size, (255, 0, 255, 255))

        monkeypatch.setattr(Config, "GENERATIVE_BG_ENABLED", True)
        monkeypatch.setattr(Config, "GENERATIVE_BG_POLICY", "BG_ONLY")
        monkeypatch.setattr(Config, "GENERATIVE_BG_SEED", 777)

        def fake_gate(*args, **kwargs):
            _ = args, kwargs
            return GateReport(
                gates_passed=False,
                fail_reasons=["logo_similarity_failed"],
                used_fallback=True,
            )

        monkeypatch.setattr("backend.app.relayout.evaluate_quality_gates", fake_gate)

        self.engine.generative_engine.backend = MockBackend()
        self.engine.source_size = (100, 100)
        self.engine.elements = [
            DesignElement(
                id="bg",
                name="bg",
                layer_type="pixel",
                bbox=BoundingBox(0, 0, 100, 100),
                image=Image.new("RGBA", (100, 100), (240, 240, 240, 255)),
                role=ElementRole.BACKGROUND,
                priority=9,
            ),
            DesignElement(
                id="logo",
                name="logo",
                layer_type="pixel",
                bbox=BoundingBox(10, 10, 20, 20),
                image=Image.new("RGBA", (20, 20), (0, 0, 255, 255)),
                role=ElementRole.LOGO,
                priority=1,
                z_index=1,
            ),
        ]

        result = self.engine.relayout((120, 120))
        assert result.gates_passed is False
        assert result.used_fallback is True
        assert result.fail_reasons == ["logo_similarity_failed"]
        assert result.metadata["quality_gates"]["used_fallback"] is True


    def test_gate_report_fields_present_on_success_path(self):
        from backend.app.enums import ElementRole
        from backend.app.models import BoundingBox, DesignElement

        self.engine.source_size = (50, 50)
        self.engine.elements = [
            DesignElement(
                id="bg",
                name="bg",
                layer_type="pixel",
                bbox=BoundingBox(0, 0, 50, 50),
                image=Image.new("RGBA", (50, 50), (200, 200, 200, 255)),
                role=ElementRole.BACKGROUND,
                priority=9,
            )
        ]

        result = self.engine.relayout((60, 60))
        assert hasattr(result, "gates_passed")
        assert hasattr(result, "fail_reasons")
        assert hasattr(result, "used_fallback")
        assert isinstance(result.fail_reasons, list)
