"""Generative outpainting engine with adapter pattern and safe fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

logger = logging.getLogger("autobanner.generative.engine")


class GenerativeBackend(Protocol):
    """Adapter interface for pluggable outpainting backends."""

    @property
    def model_id(self) -> str:
        """Stable backend model identifier."""

    def is_available(self) -> bool:
        """Return True if backend is ready to generate."""

    def generate(
        self,
        base_canvas: Image.Image,
        editable_mask: np.ndarray,
        seed: int,
    ) -> Image.Image:
        """Generate an outpainted candidate image."""


@dataclass
class OutpaintPolicy:
    """Policy configuration for outpainting."""

    name: str = "BG_ONLY"


class NullGenerativeBackend:
    """No-op backend used when generative backend is disabled/unavailable."""

    @property
    def model_id(self) -> str:
        return "deterministic-fallback"

    def is_available(self) -> bool:
        return False

    def generate(
        self,
        base_canvas: Image.Image,
        editable_mask: np.ndarray,
        seed: int,
    ) -> Image.Image:
        return base_canvas


class GenerativeOutpaintEngine:
    """Coordinates BG_ONLY outpainting with strict mask enforcement."""

    def __init__(self, backend: GenerativeBackend | None = None) -> None:
        self.backend = backend or NullGenerativeBackend()
        self.last_run_metadata: dict[str, object] = {}

    def outpaint_background(
        self,
        base_canvas: Image.Image,
        editable_mask: np.ndarray,
        policy: str,
        seed: int,
    ) -> Image.Image:
        """Outpaint background while constraining writes to editable mask only."""
        base_rgba = base_canvas.convert("RGBA")
        h, w = editable_mask.shape
        if base_rgba.size != (w, h):
            raise ValueError("editable_mask shape must match base_canvas size")

        if policy not in {"BG_ONLY", "BG_PLUS_DECOR"}:
            logger.warning("Unsupported policy '%s', fallback to deterministic canvas", policy)
            self.last_run_metadata = {
                "policy": policy,
                "seed": int(seed),
                "model_id": self.backend.model_id,
                "backend_used": False,
                "fallback_reason": "unsupported_policy",
            }
            return base_rgba

        if not self.backend.is_available():
            logger.info("Generative backend unavailable -> deterministic fallback")
            self.last_run_metadata = {
                "policy": policy,
                "seed": int(seed),
                "model_id": self.backend.model_id,
                "backend_used": False,
                "fallback_reason": "backend_unavailable",
            }
            return base_rgba

        candidate = self.backend.generate(base_rgba, editable_mask, seed)
        candidate_rgba = candidate.convert("RGBA")

        if candidate_rgba.size != base_rgba.size:
            logger.warning(
                "Generative backend returned size %s, expected %s; fallback to deterministic",
                candidate_rgba.size,
                base_rgba.size,
            )
            self.last_run_metadata = {
                "policy": policy,
                "seed": int(seed),
                "model_id": self.backend.model_id,
                "backend_used": False,
                "fallback_reason": "size_mismatch",
            }
            return base_rgba

        out = _apply_editable_mask(base_rgba, candidate_rgba, editable_mask)
        self.last_run_metadata = {
            "policy": policy,
            "seed": int(seed),
            "model_id": self.backend.model_id,
            "backend_used": True,
            "editable_ratio": float(editable_mask.mean()),
        }

        return out


def _apply_editable_mask(
    base_canvas: Image.Image,
    candidate_canvas: Image.Image,
    editable_mask: np.ndarray,
) -> Image.Image:
    """Apply candidate only on editable pixels; keep protected pixels intact."""
    base_arr = np.asarray(base_canvas.convert("RGBA"), dtype=np.uint8)
    cand_arr = np.asarray(candidate_canvas.convert("RGBA"), dtype=np.uint8)

    mask3 = editable_mask[:, :, None]
    out_arr = np.where(mask3, cand_arr, base_arr)
    return Image.fromarray(out_arr, mode="RGBA")
