"""Main ReLayout orchestrator engine."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .classifier import SemanticClassifier
from .composition import CompositionEngine
from .config import Config
from .enums import ElementRole
from .generative.decor import apply_optional_decor_synthesis
from .generative.engine import GenerativeOutpaintEngine
from .generative.gates import evaluate_quality_gates
from .generative.harmonize import (
    apply_color_grading_safe,
    apply_grounding_shadow_safe,
    extract_mascot_masks,
)
from .generative.masks import build_layout_masks
from .layout import LayoutEngine
from .layout.solver import export_layout_debug_json, render_layout_debug_overlay
from .models import CompositionResult, DesignElement, LayoutResult
from .parser import get_parser
from .validators import validate_dimensions, validate_file_path

logger = logging.getLogger("autobanner.relayout")


class ReLayoutEngine:
    """Main engine that orchestrates the entire re-layout process."""

    def __init__(self, use_ai: bool = True) -> None:
        self.classifier = SemanticClassifier(use_ai=use_ai)
        self.layout_engine = LayoutEngine()
        self.compositor = CompositionEngine(use_ai_inpainting=use_ai)
        self.generative_engine = GenerativeOutpaintEngine()

        self.elements: list[DesignElement] = []
        self.source_size: tuple[int, int] = (0, 0)
        self.file_path: str | None = None
        self._last_outpaint_metadata: dict[str, Any] = {}

    def load_file(self, file_path: str) -> dict[str, Any]:
        """Load and analyze a design file (PSD, PNG, JPG, WEBP).

        Args:
            file_path: Path to the design file.

        Returns:
            Dict with analysis results for UI display.

        Raises:
            ValidationError: If file path is invalid.
            ParseError: If parsing fails.
        """
        validate_file_path(file_path)

        self.file_path = file_path

        # Get appropriate parser
        parser = get_parser(file_path)

        # Parse file
        self.elements, self.source_size = parser.parse(file_path)

        # Classify elements
        self.elements = self.classifier.classify_all(self.elements, self.source_size)

        # Prepare analysis for UI
        analysis: dict[str, Any] = {
            "file": os.path.basename(file_path),
            "size": self.source_size,
            "total_layers": len(self.elements),
            "elements": [],
        }

        for elem in self.elements:
            analysis["elements"].append(
                {
                    "id": elem.id,
                    "name": elem.name,
                    "type": elem.layer_type,
                    "role": elem.role.value,
                    "priority": elem.priority,
                    "bbox": {
                        "x": elem.bbox.x,
                        "y": elem.bbox.y,
                        "width": elem.bbox.width,
                        "height": elem.bbox.height,
                    },
                    "has_image": elem.image is not None,
                    "text": (
                        elem.text_content[:50] + "..."
                        if elem.text_content and len(elem.text_content) > 50
                        else elem.text_content
                    ),
                }
            )

        return analysis

    def update_element_role(self, element_id: str, new_role: str) -> bool:
        """Update an element's role (for user correction).

        Args:
            element_id: ID of the element to update.
            new_role: New role value string.

        Returns:
            True if updated successfully.
        """
        for elem in self.elements:
            if elem.id == element_id:
                try:
                    elem.role = ElementRole(new_role)
                    return True
                except ValueError as e:
                    logger.warning("Invalid role '%s': %s", new_role, e)
                    return False
        return False

    def update_element_priority(self, element_id: str, new_priority: int) -> bool:
        """Update an element's priority.

        Args:
            element_id: ID of the element to update.
            new_priority: New priority (1-9).

        Returns:
            True if updated successfully.
        """
        for elem in self.elements:
            if elem.id == element_id:
                elem.priority = max(1, min(9, new_priority))
                return True
        return False

    def relayout(self, target_size: tuple[int, int]) -> CompositionResult:
        """Re-layout elements to target size.

        Args:
            target_size: Target canvas size (width, height).

        Returns:
            CompositionResult with final image.

        Raises:
            ValueError: If no file has been loaded.
            ValidationError: If dimensions are invalid.
        """
        if not self.elements:
            raise ValueError("No file loaded. Call load_file() first.")

        validate_dimensions(target_size[0], target_size[1])

        # Calculate layout
        layout_results = self.layout_engine.calculate_layout(
            self.elements, self.source_size, target_size
        )
        self._maybe_export_layout_debug(layout_results, target_size)

        # Deterministic baseline (fallback target)
        deterministic_result = self.compositor.compose(
            self.elements,
            layout_results,
            self.source_size,
            target_size,
        )
        deterministic_result = self._apply_harmonize_and_grounding(
            deterministic_result,
            layout_results,
            target_size,
            apply_decor=False,
        )

        # Candidate with generative stages
        self._last_outpaint_metadata = {}
        candidate = self.compositor.compose(
            self.elements,
            layout_results,
            self.source_size,
            target_size,
            bg_outpaint_fn=(
                lambda canvas: self._bg_only_outpaint(canvas, layout_results, target_size)
            ),
        )
        if self._last_outpaint_metadata:
            candidate.metadata["generative"] = self._last_outpaint_metadata

        candidate = self._apply_harmonize_and_grounding(
            candidate,
            layout_results,
            target_size,
            apply_decor=True,
        )

        masks = build_layout_masks(self.elements, layout_results, target_size)
        gate_report = evaluate_quality_gates(
            baseline=deterministic_result.image,
            candidate=candidate.image,
            elements=self.elements,
            layout_results=layout_results,
            protected_mask=masks.protected_mask,
        )

        candidate.gates_passed = gate_report.gates_passed
        candidate.fail_reasons = gate_report.fail_reasons
        candidate.used_fallback = False

        if gate_report.gates_passed:
            return candidate

        deterministic_result.gates_passed = gate_report.gates_passed
        deterministic_result.fail_reasons = gate_report.fail_reasons
        deterministic_result.used_fallback = True
        deterministic_result.metadata.setdefault("quality_gates", {})
        deterministic_result.metadata["quality_gates"].update(
            {
                "gates_passed": gate_report.gates_passed,
                "fail_reasons": gate_report.fail_reasons,
                "used_fallback": True,
            }
        )
        logger.warning(
            "Using deterministic fallback due to gate failures: %s",
            gate_report.fail_reasons,
        )
        return deterministic_result



    def _maybe_export_layout_debug(
        self,
        layout_results: list[LayoutResult],
        target_size: tuple[int, int],
    ) -> None:
        """Optionally export layout debug overlay and JSON metadata."""
        if not Config.LAYOUT_DEBUG_ENABLED:
            return

        role_by_id = {e.id: e.role for e in self.elements}
        overlay = render_layout_debug_overlay(target_size, layout_results, role_by_id)

        out_dir = Path(Config.LAYOUT_DEBUG_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = out_dir / "layout_debug_overlay.png"
        json_path = out_dir / "layout_debug.json"
        overlay.save(overlay_path)

        payload = {
            "target_size": {"width": target_size[0], "height": target_size[1]},
            **self.layout_engine.last_layout_debug,
        }
        export_layout_debug_json(json_path, payload)
        logger.info("Layout debug exported overlay=%s json=%s", overlay_path, json_path)

    def _apply_harmonize_and_grounding(
        self,
        result: CompositionResult,
        layout_results: list[LayoutResult],
        target_size: tuple[int, int],
        apply_decor: bool,
    ) -> CompositionResult:
        """Apply safe color harmonization and grounding shadow without changing protected pixels."""
        masks = build_layout_masks(self.elements, layout_results, target_size)

        rgba = result.image.convert("RGBA")
        graded = apply_color_grading_safe(rgba, masks.protected_mask)

        mascot_masks = extract_mascot_masks(self.elements, layout_results, target_size)
        grounded = apply_grounding_shadow_safe(
            graded,
            mascot_masks=mascot_masks,
            protected_mask=masks.protected_mask,
        )

        final_rgba = grounded

        decor_policy = Config.GENERATIVE_DECOR_POLICY if apply_decor else "OFF"
        decor_seed = Config.GENERATIVE_DECOR_SEED
        final_rgba, decor_meta = apply_optional_decor_synthesis(
            base_canvas=final_rgba,
            protected_mask=masks.protected_mask,
            policy=decor_policy,
            seed=decor_seed,
            outpaint_engine=self.generative_engine,
        )

        result.image = final_rgba.convert("RGB")
        result.metadata.setdefault("harmonize", {})
        result.metadata["harmonize"].update(
            {
                "applied": True,
                "mascot_count": len(mascot_masks),
                "protected_ratio": masks.protected_ratio,
            }
        )
        result.metadata["decor"] = decor_meta
        return result

    def _bg_only_outpaint(
        self,
        canvas: Image.Image,
        layout_results: list[LayoutResult],
        target_size: tuple[int, int],
    ) -> Image.Image:
        """Run BG_ONLY outpaint on background only, preserving protected regions."""
        policy = Config.GENERATIVE_BG_POLICY
        seed = Config.GENERATIVE_BG_SEED

        masks = build_layout_masks(self.elements, layout_results, target_size)

        if not Config.GENERATIVE_BG_ENABLED:
            self._last_outpaint_metadata = {
                "policy": policy,
                "seed": int(seed),
                "model_id": Config.GENERATIVE_BG_MODEL_ID,
                "backend_used": False,
                "fallback_reason": "disabled",
                "protected_ratio": masks.protected_ratio,
                "editable_ratio": masks.editable_ratio,
            }
            return canvas

        outpainted = self.generative_engine.outpaint_background(
            base_canvas=canvas,
            editable_mask=masks.editable_mask,
            policy=policy,
            seed=seed,
        )

        self._last_outpaint_metadata = {
            **self.generative_engine.last_run_metadata,
            "protected_ratio": masks.protected_ratio,
            "editable_ratio": masks.editable_ratio,
        }

        return outpainted

    def batch_relayout(
        self, target_sizes: list[tuple[int, int, str]]
    ) -> dict[str, CompositionResult]:
        """Re-layout to multiple sizes.

        Args:
            target_sizes: List of (width, height, name) tuples.

        Returns:
            Dict mapping name to CompositionResult.
        """
        results: dict[str, CompositionResult] = {}

        for width, height, name in target_sizes:
            try:
                result = self.relayout((width, height))
                results[name] = result
            except Exception as e:
                logger.error("Error processing %s: %s", name, e, exc_info=True)

        return results

    def get_preview_image(self) -> Image.Image | None:
        """Get a preview of the loaded file with element bounding boxes."""
        if not self.elements:
            return None

        # Find background or create white canvas
        bg = None
        for elem in self.elements:
            if elem.role == ElementRole.BACKGROUND and elem.image:
                bg = elem.image.copy()
                break

        if bg is None:
            bg = Image.new("RGBA", self.source_size, (240, 240, 240, 255))
        else:
            bg = bg.convert("RGBA")
            if bg.size != self.source_size:
                bg = bg.resize(self.source_size, Image.Resampling.LANCZOS)

        # Draw bounding boxes
        draw = ImageDraw.Draw(bg)

        # Color coding by role
        role_colors = {
            ElementRole.HEADLINE: (255, 100, 100, 200),
            ElementRole.SUBHEADLINE: (255, 150, 100, 200),
            ElementRole.CTA: (100, 255, 100, 200),
            ElementRole.BADGE: (255, 255, 100, 200),
            ElementRole.LOGO: (100, 100, 255, 200),
            ElementRole.HERO_IMAGE: (255, 100, 255, 200),
            ElementRole.BACKGROUND: (150, 150, 150, 100),
            ElementRole.DECORATION: (200, 200, 200, 100),
        }
        default_color = (180, 180, 180, 150)

        for elem in self.elements:
            if elem.role == ElementRole.BACKGROUND:
                continue

            color = role_colors.get(elem.role, default_color)
            bbox = elem.bbox.to_tuple()

            draw.rectangle(bbox, outline=color[:3], width=2)
            label = f"{elem.role.value[:8]}: {elem.name[:15]}"
            draw.text((bbox[0] + 2, bbox[1] + 2), label, fill=color[:3])

        return bg.convert("RGB")
