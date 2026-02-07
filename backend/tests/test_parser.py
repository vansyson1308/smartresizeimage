"""Tests for file parsers."""

import os
import tempfile

import pytest
from PIL import Image

from backend.app.enums import ElementRole
from backend.app.exceptions import ParseError, UnsupportedFormatError
from backend.app.parser import get_parser
from backend.app.parser.image_parser import ImageParser
from backend.app.parser.psd_parser import PSDParser


class TestParserFactory:
    def test_png_returns_image_parser(self):
        parser = get_parser("test.png")
        assert isinstance(parser, ImageParser)

    def test_jpg_returns_image_parser(self):
        parser = get_parser("test.jpg")
        assert isinstance(parser, ImageParser)

    def test_jpeg_returns_image_parser(self):
        parser = get_parser("test.jpeg")
        assert isinstance(parser, ImageParser)

    def test_webp_returns_image_parser(self):
        parser = get_parser("test.webp")
        assert isinstance(parser, ImageParser)

    def test_psd_returns_psd_parser(self):
        parser = get_parser("test.psd")
        assert isinstance(parser, PSDParser)

    def test_unsupported_format_raises(self):
        with pytest.raises(UnsupportedFormatError):
            get_parser("test.bmp")

    def test_unsupported_svg_raises(self):
        with pytest.raises(UnsupportedFormatError):
            get_parser("test.svg")


class TestImageParser:
    def test_parse_png(self):
        # Create a temp PNG file
        img = Image.new("RGB", (200, 150), (255, 0, 0))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            parser = ImageParser()
            elements, size = parser.parse(tmp_path)

            assert len(elements) == 1
            assert size == (200, 150)
            assert elements[0].role == ElementRole.BACKGROUND
            assert elements[0].image is not None
            assert elements[0].image.mode == "RGBA"
        finally:
            os.unlink(tmp_path)

    def test_parse_jpg(self):
        img = Image.new("RGB", (300, 200), (0, 255, 0))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f, format="JPEG")
            tmp_path = f.name

        try:
            parser = ImageParser()
            elements, size = parser.parse(tmp_path)

            assert len(elements) == 1
            assert size == (300, 200)
        finally:
            os.unlink(tmp_path)

    def test_parse_nonexistent_file_raises(self):
        parser = ImageParser()
        with pytest.raises(ParseError, match="Failed to open"):
            parser.parse("/nonexistent/file.png")

    def test_supports_png(self):
        parser = ImageParser()
        assert parser.supports("image.png") is True
        assert parser.supports("image.PNG") is True

    def test_does_not_support_psd(self):
        parser = ImageParser()
        assert parser.supports("file.psd") is False

    def test_parse_webp(self):
        img = Image.new("RGB", (100, 100), (0, 0, 255))
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            img.save(f, format="WEBP")
            tmp_path = f.name

        try:
            parser = ImageParser()
            elements, size = parser.parse(tmp_path)

            assert len(elements) == 1
            assert size == (100, 100)
        finally:
            os.unlink(tmp_path)


class TestPSDParser:
    def test_supports_psd(self):
        parser = PSDParser()
        assert parser.supports("file.psd") is True

    def test_does_not_support_png(self):
        parser = PSDParser()
        assert parser.supports("file.png") is False
