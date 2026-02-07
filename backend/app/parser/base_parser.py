"""Abstract base parser for design files."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import DesignElement


class BaseParser(ABC):
    """Abstract base class for file parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> tuple[list[DesignElement], tuple[int, int]]:
        """Parse input file into design elements.

        Args:
            file_path: Path to the input file.

        Returns:
            Tuple of (list of elements, (width, height)).
        """
        ...

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Return True if this parser supports the given file.

        Args:
            file_path: Path to check.

        Returns:
            True if supported.
        """
        ...
