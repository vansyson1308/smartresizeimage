"""Custom exception hierarchy for AutoBanner."""

from __future__ import annotations


class AutoBannerError(Exception):
    """Base exception for all AutoBanner errors."""


class ParseError(AutoBannerError):
    """Raised when input file parsing fails."""


class UnsupportedFormatError(ParseError):
    """Raised when input file format is not supported."""


class ClassificationError(AutoBannerError):
    """Raised when element classification fails."""


class LayoutError(AutoBannerError):
    """Raised when layout computation fails."""


class CompositionError(AutoBannerError):
    """Raised when final image composition fails."""


class ValidationError(AutoBannerError):
    """Raised when input validation fails."""
