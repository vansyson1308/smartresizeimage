"""Tests for input validation."""

import os
import tempfile

import pytest

from backend.app.exceptions import ValidationError
from backend.app.validators import validate_dimensions, validate_file_path


class TestValidateDimensions:
    def test_valid_dimensions(self):
        validate_dimensions(1080, 1920)  # No exception

    def test_zero_width_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_dimensions(0, 100)

    def test_negative_height_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_dimensions(100, -50)

    def test_oversized_width_rejected(self):
        with pytest.raises(ValidationError, match="exceed maximum"):
            validate_dimensions(5000, 100)

    def test_too_small_dimensions_rejected(self):
        with pytest.raises(ValidationError, match="below minimum"):
            validate_dimensions(5, 5)

    def test_boundary_max(self):
        validate_dimensions(4096, 4096)  # Should pass (equals max)

    def test_boundary_min(self):
        validate_dimensions(10, 10)  # Should pass (equals min)


class TestValidateFilePath:
    def test_nonexistent_file_rejected(self):
        with pytest.raises(ValidationError, match="File not found"):
            validate_file_path("/nonexistent/path/file.psd")

    def test_unsupported_extension_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            f.write(b"fake")
            tmp_path = f.name
        try:
            with pytest.raises(ValidationError, match="Unsupported format"):
                validate_file_path(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_valid_png_path(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake png data")
            tmp_path = f.name
        try:
            validate_file_path(tmp_path)  # No exception
        finally:
            os.unlink(tmp_path)

    def test_valid_psd_path(self):
        with tempfile.NamedTemporaryFile(suffix=".psd", delete=False) as f:
            f.write(b"fake psd data")
            tmp_path = f.name
        try:
            validate_file_path(tmp_path)  # No exception
        finally:
            os.unlink(tmp_path)
