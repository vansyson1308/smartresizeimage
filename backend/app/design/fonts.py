"""Font registry: resolve (family, weight, italic) to a font file with disclosure.

Missing fonts never substitute silently: ``resolve`` returns the status
(``available`` / ``substituted`` / ``missing``) alongside the file used, and
documents record it in ``FontRef`` so exports can disclose substitutions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import ImageFont

logger = logging.getLogger("autobanner.design.fonts")

_DEFAULT_DIRS = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
    "/Library/Fonts",
    "C:/Windows/Fonts",
)
_EXTENSIONS = {".ttf", ".otf", ".ttc"}
FALLBACK_FAMILY = "DejaVu Sans"

_BOLD_WORDS = ("bold", "black", "heavy", "extrabold", "semibold", "demibold")
_ITALIC_WORDS = ("italic", "oblique")


@dataclass(frozen=True)
class FontFace:
    family: str
    style: str  # as reported by the font (e.g. "Bold Oblique")
    path: str

    @property
    def bold(self) -> bool:
        s = self.style.lower()
        return any(w in s for w in _BOLD_WORDS)

    @property
    def italic(self) -> bool:
        s = self.style.lower()
        return any(w in s for w in _ITALIC_WORDS)


@dataclass(frozen=True)
class ResolvedFont:
    requested_family: str
    requested_weight: str
    requested_italic: bool
    path: str | None
    family: str | None
    style: str | None
    status: str  # available | substituted | missing

    @property
    def substituted(self) -> bool:
        return self.status != "available"


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


class FontRegistry:
    """Scans font directories once and resolves faces by family/weight/italic."""

    def __init__(self, extra_dirs: list[str | Path] | None = None, scan_system: bool = True):
        self._faces: list[FontFace] = []
        self._by_family: dict[str, list[FontFace]] = {}
        dirs: list[str] = []
        if extra_dirs:
            dirs.extend(str(d) for d in extra_dirs)
        if scan_system:
            dirs.extend(_DEFAULT_DIRS)
        for d in dirs:
            self._scan(Path(d))

    # ---- scanning ------------------------------------------------------------------
    def _scan(self, root: Path) -> None:
        if not root.exists():
            return
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in _EXTENSIONS or not path.is_file():
                continue
            self.add_file(path)

    def add_file(self, path: str | Path) -> FontFace | None:
        path = Path(path)
        try:
            font = ImageFont.truetype(str(path), size=12)
            family, style = font.getname()
        except Exception as exc:  # noqa: BLE001
            logger.debug("skip font %s: %s", path, exc)
            return None
        face = FontFace(family=family or path.stem, style=style or "Regular", path=str(path))
        self._faces.append(face)
        self._by_family.setdefault(_normalize(face.family), []).append(face)
        return face

    @property
    def families(self) -> list[str]:
        return sorted({f.family for f in self._faces})

    # ---- resolution ------------------------------------------------------------------
    def find(self, family: str, weight: str = "regular", italic: bool = False) -> FontFace | None:
        candidates = self._by_family.get(_normalize(family), [])
        if not candidates:
            # Accept "DejaVuSans-Bold" style names that embed the style.
            key = _normalize(family)
            for fam_key, faces in self._by_family.items():
                if key.startswith(fam_key) and fam_key:
                    candidates = faces
                    break
        if not candidates:
            return None
        want_bold = weight.lower() in _BOLD_WORDS or weight.lower() in ("700", "800", "900")
        want_italic = bool(italic)

        def score(face: FontFace) -> int:
            s = 0
            s += 2 if face.bold == want_bold else 0
            s += 1 if face.italic == want_italic else 0
            return s

        return max(candidates, key=score)

    def resolve(self, family: str, weight: str = "regular", italic: bool = False) -> ResolvedFont:
        face = self.find(family, weight, italic)
        if face is not None:
            return ResolvedFont(
                family, weight, italic, face.path, face.family, face.style, "available"
            )
        fallback = self.find(FALLBACK_FAMILY, weight, italic)
        if fallback is not None:
            logger.warning("font '%s' missing; substituted %s", family, fallback.family)
            return ResolvedFont(
                family,
                weight,
                italic,
                fallback.path,
                fallback.family,
                fallback.style,
                "substituted",
            )
        return ResolvedFont(family, weight, italic, None, None, None, "missing")

    def load(
        self, resolved: ResolvedFont, size: float
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        px = max(1, int(round(size)))
        if resolved.path:
            try:
                return ImageFont.truetype(resolved.path, size=px)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not load %s: %s", resolved.path, exc)
        return ImageFont.load_default()


_registry: FontRegistry | None = None


def default_registry() -> FontRegistry:
    global _registry
    if _registry is None:
        _registry = FontRegistry()
    return _registry
