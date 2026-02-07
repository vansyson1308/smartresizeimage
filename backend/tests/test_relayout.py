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
