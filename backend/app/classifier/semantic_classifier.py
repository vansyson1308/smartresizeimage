"""Semantic classification of design elements."""

from __future__ import annotations

import logging

from ..config import Config
from ..constants import ROLE_PRIORITIES
from ..enums import ElementRole
from ..models import DesignElement

logger = logging.getLogger("autobanner.classifier")

# Optional AI imports
try:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    logger.info("CLIP not available. Using rule-based classification only.")


class SemanticClassifier:
    """Classify design elements by their semantic role.

    Uses three strategies:
    1. Rule-based: Fast, based on naming conventions
    2. AI-based (CLIP): More accurate for ambiguous cases
    3. Heuristics: Fallback based on visual properties
    """

    # Naming convention patterns
    NAMING_PATTERNS: dict[ElementRole, list[str]] = {
        ElementRole.HEADLINE: [
            "headline", "title", "heading", "h1", "h2", "main_text", "header_text",
        ],
        ElementRole.SUBHEADLINE: [
            "subhead", "subtitle", "sub_title", "h3", "h4", "secondary",
        ],
        ElementRole.BODY_TEXT: [
            "body", "paragraph", "text", "content", "description", "desc",
        ],
        ElementRole.CTA: [
            "cta", "button", "btn", "action", "click", "learn_more", "buy_now", "shop_now",
        ],
        ElementRole.BADGE: [
            "badge", "tag", "label", "alert", "warning", "notice", "ribbon",
        ],
        ElementRole.LABEL: ["label", "caption", "note", "small_text"],
        ElementRole.LOGO: ["logo", "brand", "wordmark", "trademark"],
        ElementRole.HERO_IMAGE: [
            "hero", "main_image", "featured", "mascot", "character", "key_visual",
        ],
        ElementRole.ICON: ["icon", "ico", "symbol", "glyph"],
        ElementRole.PHOTO: ["photo", "photograph", "picture", "image", "img"],
        ElementRole.ILLUSTRATION: [
            "illust", "illustration", "artwork", "drawing", "graphic",
        ],
        ElementRole.DECORATION: [
            "decor", "decoration", "ornament", "sparkle", "effect", "particle",
        ],
        ElementRole.BACKGROUND: ["bg", "background", "backdrop", "base"],
        ElementRole.BACKGROUND_PATTERN: ["bg_pattern", "pattern", "texture", "tile"],
        ElementRole.OVERLAY: ["overlay", "gradient_overlay", "color_overlay", "mask"],
        ElementRole.GROUP: ["group", "folder", "section"],
    }

    # CLIP prompts for each role
    CLIP_PROMPTS: dict[ElementRole, str] = {
        ElementRole.HEADLINE: "a large title text, headline, main text",
        ElementRole.SUBHEADLINE: "a subtitle, secondary text, smaller heading",
        ElementRole.BODY_TEXT: "paragraph text, body text, description",
        ElementRole.CTA: "a button, call to action, clickable element",
        ElementRole.BADGE: "a badge, tag, label, notification",
        ElementRole.LOGO: "a company logo, brand mark",
        ElementRole.HERO_IMAGE: "a main image, hero image, mascot, character",
        ElementRole.ICON: "an icon, small symbol",
        ElementRole.PHOTO: "a photograph, real photo",
        ElementRole.ILLUSTRATION: "an illustration, artwork, drawing",
        ElementRole.DECORATION: "decorative element, sparkle, particle effect",
        ElementRole.BACKGROUND: "background, backdrop",
        ElementRole.BACKGROUND_PATTERN: "a repeating pattern, texture, tile background",
        ElementRole.LABEL: "a small text label, caption, annotation",
        ElementRole.OVERLAY: "a color overlay, gradient overlay, translucent layer",
    }

    def __init__(self, use_ai: bool = True) -> None:
        self._use_ai = use_ai and HAS_CLIP
        self._clip_model = None
        self._clip_processor = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load CLIP model on first AI classification request."""
        if self._loaded or not self._use_ai:
            return
        self._loaded = True
        try:
            logger.info("Loading CLIP model...")
            self._clip_model = CLIPModel.from_pretrained(Config.CLIP_MODEL)
            self._clip_processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL)
            logger.info("CLIP model loaded")
        except Exception as e:
            logger.warning("Could not load CLIP: %s", e)
            self._use_ai = False

    def classify(
        self, element: DesignElement, canvas_size: tuple[int, int]
    ) -> ElementRole:
        """Classify an element's semantic role.

        Args:
            element: The design element to classify.
            canvas_size: Original canvas size (width, height).

        Returns:
            ElementRole enum value.
        """
        # Skip groups
        if element.layer_type == "group":
            return ElementRole.GROUP

        # Step 1: Rule-based (fast)
        rule_role = self._classify_by_rules(element)
        if rule_role != ElementRole.UNKNOWN:
            return rule_role

        # Step 2: AI classification if available
        if self._use_ai and element.image:
            ai_role = self._classify_by_ai(element)
            if ai_role != ElementRole.UNKNOWN:
                return ai_role

        # Step 3: Fallback heuristics
        return self._classify_by_heuristics(element, canvas_size)

    def _classify_by_rules(self, element: DesignElement) -> ElementRole:
        """Classify based on naming conventions."""
        name_lower = element.name.lower()

        # Check naming patterns
        for role, patterns in self.NAMING_PATTERNS.items():
            for pattern in patterns:
                if pattern in name_lower:
                    return role

        # Check for priority suffix
        if "_1" in name_lower or "_p1" in name_lower:
            element.priority = 1
        elif "_2" in name_lower or "_p2" in name_lower:
            element.priority = 2
        elif "_9" in name_lower or "_low" in name_lower:
            element.priority = 9

        return ElementRole.UNKNOWN

    def _classify_by_ai(self, element: DesignElement) -> ElementRole:
        """Classify using CLIP model."""
        self._ensure_loaded()

        if not self._clip_model or not element.image:
            return ElementRole.UNKNOWN

        try:
            image = element.image.convert("RGB")

            text_prompts = list(self.CLIP_PROMPTS.values())
            roles = list(self.CLIP_PROMPTS.keys())

            inputs = self._clip_processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            best_idx = probs.argmax().item()
            confidence = probs[0, best_idx].item()

            if confidence >= Config.CONFIDENCE_THRESHOLD:
                return roles[best_idx]

        except Exception as e:
            logger.warning("CLIP classification error: %s", e)

        return ElementRole.UNKNOWN

    def _classify_by_heuristics(
        self, element: DesignElement, canvas_size: tuple[int, int]
    ) -> ElementRole:
        """Classify using visual heuristics."""
        canvas_w, canvas_h = canvas_size
        elem_area = element.bbox.area
        canvas_area = canvas_w * canvas_h
        area_ratio = elem_area / canvas_area if canvas_area > 0 else 0

        # Text layer heuristics
        if element.layer_type == "type" and element.text_content:
            text_len = len(element.text_content)

            # Large text at top = headline
            if element.bbox.y < canvas_h * 0.3 and (area_ratio > 0.05 or text_len < 30):
                return ElementRole.HEADLINE

            # Medium text = subheadline or body
            if text_len < 50:
                return ElementRole.SUBHEADLINE
            return ElementRole.BODY_TEXT

        # Image layer heuristics
        if element.layer_type in ["pixel", "smartobject"]:
            if area_ratio > 0.5:
                return ElementRole.BACKGROUND
            elif area_ratio > 0.15:
                return ElementRole.HERO_IMAGE
            elif area_ratio < 0.02:
                return ElementRole.ICON
            else:
                return ElementRole.ILLUSTRATION

        # Shape layer
        if element.layer_type == "shape":
            if area_ratio < 0.05:
                return ElementRole.DECORATION
            return ElementRole.BACKGROUND

        return ElementRole.UNKNOWN

    def classify_all(
        self, elements: list[DesignElement], canvas_size: tuple[int, int]
    ) -> list[DesignElement]:
        """Classify all elements and update their roles.

        Args:
            elements: List of design elements.
            canvas_size: Original canvas size (width, height).

        Returns:
            The same list with roles and priorities updated.
        """
        for element in elements:
            element.role = self.classify(element, canvas_size)

            # Set priority based on role
            if element.role in ROLE_PRIORITIES:
                element.priority = min(element.priority, ROLE_PRIORITIES[element.role])

        return elements
