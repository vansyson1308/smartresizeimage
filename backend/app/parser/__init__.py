"""Parser factory for AutoBanner."""

from __future__ import annotations

from ..exceptions import UnsupportedFormatError
from .base_parser import BaseParser
from .image_parser import ImageParser
from .psd_parser import PSDParser

_PARSERS: list[BaseParser] = [PSDParser(), ImageParser()]


def get_parser(file_path: str) -> BaseParser:
    """Get the appropriate parser for a file.

    Args:
        file_path: Path to the input file.

    Returns:
        A parser instance that supports the file.

    Raises:
        UnsupportedFormatError: If no parser supports the file.
    """
    for parser in _PARSERS:
        if parser.supports(file_path):
            return parser
    raise UnsupportedFormatError(
        f"No parser for '{file_path}'. Supported: PSD, PNG, JPG, WEBP"
    )


__all__ = ["get_parser", "BaseParser", "PSDParser", "ImageParser"]
