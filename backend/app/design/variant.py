"""Generate one variant from a document: plan -> render -> verify -> repair.

Planning reuses the existing layout engine through the adapter (native text
elements carry no raster, so the engine reflows them). Text is then fitted and
rasterized with real fonts at the planned size, composited by the existing
compositor, and verified by the quality contract. A bounded repair loop fixes
the most common failures (text hidden behind a subject, overflow) and
re-verifies; nothing is ever hidden to make the verdict better.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from PIL import Image

from ..composition.engine import CompositionEngine
from ..config import Config
from ..constants import BACKGROUND_ROLES
from ..enums import ElementRole
from ..layout.engine import LayoutEngine
from ..models import BoundingBox, DesignElement, LayoutResult
from ..quality import CheckResult, CheckStatus, QualityConfig, QualityReport, Severity
from ..quality.contract import derive_verdict, summarize
from ..quality.evaluate import evaluate_composition
from ..quality.structural import REQUIRED_ROLES_DEFAULT
from .adapter import elements_from_document
from .assets import AssetStore
from .document import DesignDocument, Element, TextContent
from .fonts import FontRegistry, default_registry
from .planner import Family, plan_layout
from .text_render import fit_text, render_text

logger = logging.getLogger("autobanner.design.variant")

ProgressFn = Callable[[str, float], None]


class CancelledError(RuntimeError):
    """Raised when a cancellation token is set during generation."""


@dataclass
class VariantBrief:
    """What the user asked for in this variant."""

    width: int
    height: int
    name: str = ""
    text_overrides: dict[str, str] = field(default_factory=dict)  # element id -> new copy
    locale: str | None = None
    hidden_elements: list[str] = field(default_factory=list)
    channel_preset: str | None = None
    # campaign row this variant belongs to ({"id", "label"}), None for a plain run
    row: dict | None = None
    # creative direction (see grammar.parse_direction), None to let the planner choose
    direction: dict | None = None

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "name": self.name,
            "text_overrides": dict(self.text_overrides),
            "locale": self.locale,
            "hidden_elements": list(self.hidden_elements),
            "channel_preset": self.channel_preset,
            "row": dict(self.row) if self.row else None,
            "direction": dict(self.direction) if self.direction else None,
        }


@dataclass
class VariantResult:
    image: Image.Image
    layout: list[LayoutResult]
    elements: list[DesignElement]
    report: QualityReport
    plan: dict
    warnings: list[str] = field(default_factory=list)
    repair_steps: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return self.report.verdict.value


_TEXT_BOUNDS = {
    ElementRole.HEADLINE: (20, 96, 4),
    ElementRole.SUBHEADLINE: (16, 56, 5),
    ElementRole.CTA: (14, 40, 2),
    ElementRole.BODY_TEXT: (12, 32, 8),
    ElementRole.LABEL: (10, 24, 3),
    ElementRole.BADGE: (12, 40, 2),
}


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("variant generation cancelled")


def generate_variant(
    doc: DesignDocument,
    assets: AssetStore,
    brief: VariantBrief,
    *,
    registry: FontRegistry | None = None,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
    max_repairs: int = 3,
    quality_config: QualityConfig | None = None,
    planner: str | None = None,
    family: Family | None = None,
    reference: dict | None = None,
) -> VariantResult:
    """Generate one variant.

    ``reference`` is a previous variant's plan (``placements`` + ``typography``) for
    the same target. When given, elements present in it keep their placed boxes and
    font sizes, so a campaign revision (copy, asset, style, hidden element) changes
    pixels only inside the edited elements' boxes (H3, local edits). A text whose
    new copy no longer fits its reference box falls back to the fresh plan and is
    reported as ``layout_change``.
    """
    reg = registry or default_registry()
    target = (int(brief.width), int(brief.height))
    report_progress = progress or (lambda _stage, _frac: None)

    # 1) Materialize engine elements from the document (native text, no raster).
    report_progress("plan", 0.05)
    elements = elements_from_document(
        doc, assets, registry=reg, text_overrides=brief.text_overrides, locale=brief.locale
    )
    elements = [e for e in elements if e.id not in set(brief.hidden_elements)]
    by_id = {e.id: e for e in elements}
    doc_by_id = {e.id: e for e in doc.elements}
    _check_cancel(cancel)

    # 2) Plan placement: constraint-aware families (default) or the legacy zone engine.
    planner_name = planner or Config.DESIGN_PLANNER
    plan_meta: dict = {}
    if planner_name == "constraints":
        plan_result = plan_layout(
            doc, elements, target, registry=reg, families=[family] if family else None,
            direction=brief.direction,
        )
        layout = plan_result.layout
        layout_debug = {"profile_name": plan_result.family, "fallback_reason": ""}
        plan_meta = {
            **plan_result.to_dict(),
            "joint_family": family is not None,
            "from_examples": family is not None and family.name.startswith("learned_"),
            "direction": dict(brief.direction) if brief.direction else None,
        }
    else:
        layout_engine = LayoutEngine()
        layout = layout_engine.calculate_layout(
            elements, (doc.canvas_width, doc.canvas_height), target
        )
        layout_debug = dict(layout_engine.last_layout_debug)
    _check_cancel(cancel)
    report_progress("typeset", 0.35)

    # 2b) Local edits: keep the reference variant's boxes for elements it placed.
    fresh_boxes = {r.element_id: r for r in layout}
    kept: list[str] = []
    replanned: list[str] = []
    ref_px: dict[str, int] = {}
    ref_notes: list[str] = []
    if reference and planner_name == "constraints":
        layout, kept, replanned, ref_px, ref_notes = _apply_reference(
            elements, layout, reference, by_id, doc, target
        )
        plan_meta["reference"] = {"kept": kept, "replanned": replanned}

    # 3) Fit + rasterize native text at the planned boxes.
    typography: dict[str, dict] = {}
    layout, warnings = _typeset_text(
        elements,
        layout,
        doc_by_id,
        target,
        reg,
        typography,
        doc.constraints,
        canvas=(doc.canvas_width, doc.canvas_height),
        px_caps=ref_px,
    )
    warnings = ref_notes + warnings
    if kept:
        # A text whose new copy no longer fits the reference box gets its fresh box.
        layout_map = {r.element_id: r for r in layout}
        for eid in list(kept):
            typo = typography.get(eid)
            if typo is None or not typo.get("overflow"):
                continue
            lr, fresh = layout_map.get(eid), fresh_boxes.get(eid)
            if lr is None or fresh is None:
                continue
            lr.new_bbox = fresh.new_bbox
            lr.visible = fresh.visible
            elem = by_id[eid]
            warnings.extend(
                _typeset_one(
                    elem,
                    lr,
                    doc_by_id.get(eid),
                    target,
                    reg,
                    typography,
                    doc.constraints,
                    (doc.canvas_width, doc.canvas_height),
                )
            )
            warnings.append(f"layout_change:{eid}: copy no longer fits its previous box")
            kept.remove(eid)
            replanned.append(eid)
        plan_meta["reference"] = {"kept": kept, "replanned": replanned}
    compact_steps = _compact_groups(doc, layout) if not kept else []
    _check_cancel(cancel)

    # 4) Render, verify, repair (bounded).
    compositor = CompositionEngine(use_ai_inpainting=False)
    repair_steps: list[str] = list(compact_steps)
    report: QualityReport | None = None
    image: Image.Image | None = None
    base_cfg = quality_config or QualityConfig()
    # Intermediate rounds skip OCR (expensive, and pixels decide occlusion);
    # the final render always gets the full contract.
    quick_cfg = QualityConfig(
        visibility=base_cfg.visibility,
        legibility=base_cfg.legibility,
        structural=base_cfg.structural,
        required_roles=base_cfg.required_roles,
        run_ocr=False,
    )
    for attempt in range(max_repairs + 1):
        report_progress("render", 0.5 + 0.1 * attempt)
        result = compositor.compose(
            elements,
            layout,
            (doc.canvas_width, doc.canvas_height),
            target,
            plate_rects=(reference or {}).get("text_plate_rects") if kept else None,
        )
        image = result.image
        plate_rects = list((result.metadata.get("text_plate") or {}).get("rects") or [])
        _check_cancel(cancel)
        report_progress("verify", 0.65 + 0.1 * attempt)
        final_round = attempt == max_repairs
        report = _verify(
            doc,
            elements,
            layout,
            image,
            target,
            typography,
            base_cfg if final_round else quick_cfg,
        )
        structural_ok = not any(
            c.status == CheckStatus.FAIL and c.check_id in ("element_visible", "text_overlap")
            for c in report.checks
        )
        if structural_ok and not final_round:
            report = _verify(doc, elements, layout, image, target, typography, base_cfg)
            break
        if final_round:
            break
        changed, steps = _repair(doc, elements, layout, report, target, reg, typography, by_id)
        repair_steps.extend(steps)
        if not changed:
            report = _verify(doc, elements, layout, image, target, typography, base_cfg)
            break

    assert report is not None and image is not None
    plan = {
        "target": {"width": target[0], "height": target[1]},
        "brief": brief.to_dict(),
        "planner": planner_name,
        "planner_meta": plan_meta,
        "layout_profile": layout_debug.get("profile_name"),
        "layout_fallback": layout_debug.get("fallback_reason", ""),
        "layout_scoring": bool(Config.LAYOUT_PROFILE_SCORING_ENABLED),
        "text_plate_rects": plate_rects,
        "placements": [
            {
                "element_id": r.element_id,
                "x": r.new_bbox.x,
                "y": r.new_bbox.y,
                "width": r.new_bbox.width,
                "height": r.new_bbox.height,
                "visible": r.visible,
            }
            for r in layout
        ],
        "typography": typography,
        "repair_steps": repair_steps,
        "fonts": _font_disclosure(typography),
    }
    report_progress("done", 1.0)
    return VariantResult(
        image=image,
        layout=layout,
        elements=elements,
        report=report,
        plan=plan,
        warnings=warnings + list(result.warnings),
        repair_steps=repair_steps,
    )


def _typeset_text(
    elements: list[DesignElement],
    layout: list[LayoutResult],
    doc_by_id: dict[str, Element],
    target: tuple[int, int],
    reg: FontRegistry,
    typography: dict[str, dict],
    doc_constraints: list | None = None,
    canvas: tuple[int, int] | None = None,
    px_caps: dict[str, int] | None = None,
) -> tuple[list[LayoutResult], list[str]]:
    warnings: list[str] = []
    layout_map = {r.element_id: r for r in layout}
    caps = px_caps or {}
    for elem in elements:
        if elem.layer_type != "type" or not elem.text_content:
            continue
        lr = layout_map.get(elem.id)
        if lr is None or not lr.visible:
            continue
        warnings.extend(
            _typeset_one(
                elem,
                lr,
                doc_by_id.get(elem.id),
                target,
                reg,
                typography,
                doc_constraints or [],
                canvas or target,
                max_px_cap=caps.get(elem.id),
            )
        )
    return layout, warnings


def _apply_reference(
    elements: list[DesignElement],
    layout: list[LayoutResult],
    reference: dict,
    by_id: dict[str, DesignElement],
    doc: DesignDocument,
    target: tuple[int, int],
) -> tuple[list[LayoutResult], list[str], list[str], dict[str, int], list[str]]:
    """Replace planned boxes with the reference variant's boxes where it placed the element.

    Backgrounds always fill the canvas; images whose asset aspect changed are refitted
    uniformly inside the reference box (never distorted). A kept box that violates a
    hard rule in force *now* (a rule added after the reference was made) is not
    copied: every element of that rule falls back to the fresh plan and the change is
    reported. Returns the new layout, the ids kept from the reference, the ids planned
    fresh, the reference font sizes, and layout-change notes.
    """
    ref_boxes = {p["element_id"]: p for p in reference.get("placements", [])}
    ref_typo = reference.get("typography", {}) or {}
    kept: list[str] = []
    replanned: list[str] = []
    px: dict[str, int] = {}
    out: list[LayoutResult] = []
    for r in layout:
        e = by_id.get(r.element_id)
        p = ref_boxes.get(r.element_id)
        if e is None or e.role in BACKGROUND_ROLES:
            out.append(r)
            continue
        if p is None:
            out.append(r)
            replanned.append(r.element_id)
            continue
        box = BoundingBox(
            int(p["x"]), int(p["y"]), max(1, int(p["width"])), max(1, int(p["height"]))
        )
        if e.layer_type != "type" and e.image is not None:
            ia = e.image.width / max(1, e.image.height)
            ba = box.width / max(1, box.height)
            if abs(ia - ba) / max(ia, ba) > 0.02:
                k = min(box.width / max(1, e.image.width), box.height / max(1, e.image.height))
                w = max(1, int(e.image.width * k))
                h = max(1, int(e.image.height * k))
                box = BoundingBox(box.x + (box.width - w) // 2, box.y + (box.height - h) // 2, w, h)
        out.append(
            LayoutResult(r.element_id, box, r.scale_factor, visible=bool(p.get("visible", True)))
        )
        kept.append(r.element_id)
        if r.element_id in ref_typo and ref_typo[r.element_id].get("font_px"):
            px[r.element_id] = int(ref_typo[r.element_id]["font_px"])

    # Hard rules are evaluated on the kept layout before it is trusted (finding 2).
    notes: list[str] = []
    fresh = {r.element_id: r for r in layout}
    by_cid = {c.id: c for c in doc.constraints}
    for chk in constraint_checks(doc, out, target, {}, set()):
        if chk.status != CheckStatus.FAIL or chk.severity != Severity.CRITICAL:
            continue
        rule = by_cid.get(str(chk.details.get("constraint_id")))
        if rule is None:
            continue
        for eid in rule.elements:
            if eid not in kept or eid not in fresh:
                continue
            kept.remove(eid)
            replanned.append(eid)
            px.pop(eid, None)
            out = [fresh[eid] if r.element_id == eid else r for r in out]
            notes.append(
                f"layout_change:{eid}: hard rule '{rule.type}' is not satisfied by the "
                "previous layout"
            )
    return out, kept, replanned, px, notes


def _typeset_one(
    elem: DesignElement,
    lr: LayoutResult,
    de: Element | None,
    target: tuple[int, int],
    reg: FontRegistry,
    typography: dict[str, dict],
    doc_constraints: list,
    canvas: tuple[int, int],
    *,
    max_px_cap: int | None = None,
) -> list[str]:
    """Fit and rasterize one native text element into its planned box.

    The font size is chosen so the wrapped block fits the planned width *and*
    height; hierarchy is preserved by capping at the master size scaled to the
    target. Overflow at the minimum size is reported, never clipped.
    """
    warnings: list[str] = []
    content = _content_for(elem, de)
    min_px, max_px, max_lines = _TEXT_BOUNDS.get(elem.role, (12, 48, 6))
    min_override = _min_text_px_constraint(doc_constraints, elem.id, target)
    if min_override:
        min_px = max(min_px, min_override)
    if de is not None and de.text is not None and de.text.max_lines:
        max_lines = de.text.max_lines
    base_px = content.primary_style.font_size
    scale = min(target[0] / max(1, canvas[0]), target[1] / max(1, canvas[1]))
    preferred = int(round(base_px * max(scale, 0.35)))
    max_px = max(min_px, min(max_px, max(preferred, min_px)))
    if max_px_cap is not None:
        max_px = max(min_px, min(max_px, int(max_px_cap)))
    box_w = max(8, lr.new_bbox.width)
    box_h = max(8, lr.new_bbox.height)
    fitted = fit_text(
        content, box_w, box_h, min_px=min_px, max_px=max_px, registry=reg, max_lines=max_lines
    )
    new_h = max(fitted.height, 1)
    elem.image = render_text(content, fitted, box_w, new_h, registry=reg)
    lr.new_bbox = BoundingBox(lr.new_bbox.x, lr.new_bbox.y, box_w, new_h)
    lr.scale_factor = fitted.font_px / max(1.0, base_px)
    # Glyph coverage is checked per run: each run may use a different face.
    run_fonts = fitted.run_fonts or [fitted.resolved_font] * max(1, len(content.runs))
    run_px = fitted.run_px or [fitted.font_px] * len(run_fonts)
    runs_meta: list[dict] = []
    missing: list[str] | None = []
    for idx, run in enumerate(content.runs or [type("R", (), {"text": content.plain})()]):
        rf = run_fonts[idx] if idx < len(run_fonts) else fitted.resolved_font
        px = run_px[idx] if idx < len(run_px) else fitted.font_px
        text = run.text.upper() if getattr(getattr(run, "style", None), "uppercase", False) \
            else run.text
        run_missing = reg.missing_glyphs(rf.path, text)
        if run_missing is None:
            missing = None
        elif missing is not None:
            missing.extend(ch for ch in run_missing if ch not in missing)
        style = getattr(run, "style", None)
        runs_meta.append(
            {
                "text": run.text[:40],
                "font_px": int(px),
                "font_family_requested": rf.requested_family,
                "font_family_used": rf.family,
                "font_status": rf.status,
                "weight": getattr(style, "weight", "regular"),
                "italic": bool(getattr(style, "italic", False)),
                "color": getattr(style, "color", "#000000"),
                "missing_glyphs": None if run_missing is None else "".join(run_missing),
            }
        )
    typography[elem.id] = {
        "font_px": fitted.font_px,
        "min_px": min_px,
        "lines": len(fitted.lines),
        "overflow": bool(fitted.overflow),
        "font_family_requested": fitted.resolved_font.requested_family,
        "font_family_used": fitted.resolved_font.family,
        "font_status": fitted.resolved_font.status,
        "missing_glyphs": None if missing is None else "".join(missing),
        "box_h": box_h,
        "runs": runs_meta,
    }
    if missing:
        warnings.append(
            f"text '{content.plain[:30]}' has {len(missing)} character(s) the font cannot draw"
        )
    if fitted.overflow:
        warnings.append(f"text '{content.plain[:30]}' does not fit at minimum size")
    seen: set[str] = set()
    for rf in run_fonts:
        if rf.substituted and rf.requested_family not in seen:
            seen.add(rf.requested_family)
            warnings.append(
                f"font '{rf.requested_family}' missing; substituted {rf.family}"
            )
    return warnings


def _content_for(elem: DesignElement, de: Element | None) -> TextContent:
    if de is not None and de.text is not None:
        content = TextContent(
            runs=[type(r)(text=r.text, style=r.style) for r in de.text.runs],
            locale=de.text.locale,
            max_lines=de.text.max_lines,
            protected=de.text.protected,
        )
        if elem.text_content and elem.text_content != de.text.plain:
            content.replace_text(elem.text_content)
        return content
    content = TextContent()
    content.replace_text(elem.text_content or "")
    return content


def _min_text_px_constraint(
    constraints: list, element_id: str, target: tuple[int, int] | None = None
) -> int | None:
    """Largest enabled minimum size for ``element_id``.

    A minimum learned from a correction carries ``measured_on`` (the size it was
    observed at) and scales with the target canvas so it means the same thing on
    every size.
    """
    best: int | None = None
    for c in constraints:
        if c.enabled and c.type == "min_text_size" and element_id in c.elements:
            px = int(c.params.get("px", 0))
            measured = c.params.get("measured_on")
            if target is not None and measured and len(measured) == 2:
                mw, mh = int(measured[0]), int(measured[1])
                if mw > 0 and mh > 0:
                    px = int(round(px * min(target[0] / mw, target[1] / mh)))
            best = px if best is None else max(best, px)
    return best


def _verify(
    doc: DesignDocument,
    elements: list[DesignElement],
    layout: list[LayoutResult],
    image: Image.Image,
    target: tuple[int, int],
    typography: dict[str, dict],
    quality_config: QualityConfig | None,
) -> QualityReport:
    required = set(REQUIRED_ROLES_DEFAULT)
    measured = {eid: int(t["font_px"]) for eid, t in typography.items()}
    report = evaluate_composition(
        elements=elements,
        layout_results=layout,
        image=image,
        target_size=target,
        config=quality_config,
        measured_font_px=measured,
        allowed_overlaps=doc.allowed_overlaps(),
        extra_context={"pipeline": "design.variant", "document_id": doc.id},
    )
    # Constraint checks from the document.
    extra = constraint_checks(doc, layout, target, typography, required)
    extra.extend(_typography_checks(typography, doc))
    extra.extend(_provenance_checks(doc, layout))
    profile = doc.metadata.get("brand_profile") if isinstance(doc.metadata, dict) else None
    if isinstance(profile, dict) and profile:
        from .brand import brand_profile_checks

        extra.extend(brand_profile_checks(doc, layout, typography, profile))
    if extra:
        checks = report.checks + extra
        report = QualityReport(
            contract_version=report.contract_version,
            verdict=derive_verdict(checks),
            checks=checks,
            config=report.config,
            summary=summarize(checks),
        )
    return report


def _provenance_checks(doc: DesignDocument, layout: list[LayoutResult]) -> list[CheckResult]:
    """Recovered (inferred) elements need a human confirmation before acceptance."""
    placed = {r.element_id for r in layout if r.visible}
    unconfirmed = [
        e
        for e in doc.elements
        if e.id in placed
        and not e.is_background
        and e.provenance.origin == "recovered"
        and e.role_confidence < 0.95
    ]
    if not unconfirmed:
        return []
    names = ", ".join(e.name for e in unconfirmed[:4])
    return [
        CheckResult(
            "recovered_unconfirmed",
            CheckStatus.NEEDS_REVIEW,
            Severity.MAJOR,
            f"{len(unconfirmed)} element(s) were recovered from a flat image and not yet "
            f"confirmed: {names}",
            subject_id=unconfirmed[0].id,
            details={"element_ids": [e.id for e in unconfirmed]},
        )
    ]


def _typography_checks(typography: dict[str, dict], doc: DesignDocument) -> list[CheckResult]:
    """Glyph coverage and overflow are customer-visible text failures."""
    out: list[CheckResult] = []
    for eid, t in typography.items():
        missing = t.get("missing_glyphs")
        if missing is None:
            status, msg = CheckStatus.NOT_CHECKED, "Font glyph coverage could not be verified"
        elif missing:
            status = CheckStatus.FAIL
            msg = f"'{_name(doc, eid)}' contains characters the font cannot draw: {missing[:12]}"
        else:
            status, msg = CheckStatus.PASS, f"'{_name(doc, eid)}' glyphs are all available"
        out.append(
            CheckResult(
                "font_coverage",
                status,
                Severity.CRITICAL,
                msg,
                subject_id=eid,
                details={"missing": missing, "font": t.get("font_family_used")},
            )
        )
        if t.get("overflow"):
            out.append(
                CheckResult(
                    "text_fits",
                    CheckStatus.FAIL,
                    Severity.MAJOR,
                    f"'{_name(doc, eid)}' does not fit at its minimum size ({t.get('min_px')}px)",
                    subject_id=eid,
                    details={"font_px": t.get("font_px"), "lines": t.get("lines")},
                )
            )
    return out


def constraint_checks(
    doc: DesignDocument,
    layout: list[LayoutResult],
    target: tuple[int, int],
    typography: dict[str, dict],
    required_roles: set[ElementRole],
) -> list[CheckResult]:
    """Evaluate every enabled document constraint against the final layout.

    Policy (independent audit, finding 2): a *hard* rule is evaluated after all
    transformations with CRITICAL severity; a violation FAILs and a rule that cannot
    be evaluated (its element was dropped, its font size is unknown) is NOT_CHECKED,
    which the verdict treats as "no evidence", never as a pass. A *soft* rule is
    advisory: a violation is NEEDS_REVIEW and an unevaluable one is skipped.
    """
    layout_map = {r.element_id: r for r in layout}
    tw, th = target
    out: list[CheckResult] = []

    def visible(eid: str) -> LayoutResult | None:
        lr = layout_map.get(eid)
        return lr if lr is not None and lr.visible else None

    for c in doc.constraints:
        if not c.enabled:
            continue
        hard = bool(c.hard)
        check_id = f"constraint_{_CHECK_SUFFIX.get(c.type, c.type)}"

        def emit(ok: bool, subject: str, ok_msg: str, bad_msg: str, details: dict,
                 *, soft_severity: Severity = Severity.MINOR, _c=c, _hard=hard,
                 _id=check_id) -> None:
            if ok:
                status, severity = CheckStatus.PASS, (
                    Severity.CRITICAL if _hard else soft_severity
                )
            elif _hard:
                status, severity = CheckStatus.FAIL, Severity.CRITICAL
            else:
                status, severity = CheckStatus.NEEDS_REVIEW, soft_severity
            out.append(
                CheckResult(
                    _id, status, severity, ok_msg if ok else bad_msg,
                    subject_id=subject, details={"constraint_id": _c.id, **details},
                )
            )

        def unchecked(subject: str, why: str, _c=c, _hard=hard, _id=check_id) -> None:
            if not _hard:
                return
            out.append(
                CheckResult(
                    _id, CheckStatus.NOT_CHECKED, Severity.CRITICAL,
                    f"hard rule '{_c.type}' could not be evaluated: {why}",
                    subject_id=subject, details={"constraint_id": _c.id, "reason": why},
                )
            )

        if c.type == "keep_visible":
            for eid in c.elements:
                emit(
                    visible(eid) is not None, eid,
                    f"'{_name(doc, eid)}' stays visible", f"'{_name(doc, eid)}' was dropped",
                    {}, soft_severity=Severity.MAJOR,
                )
        elif c.type == "order_below" and len(c.elements) == 2:
            a, b = visible(c.elements[0]), visible(c.elements[1])
            if a is None or b is None:
                gone = c.elements[0] if a is None else c.elements[1]
                unchecked(c.elements[0], f"'{_name(doc, gone)}' is not placed")
                continue
            emit(
                a.new_bbox.y >= b.new_bbox.y, c.elements[0],
                f"'{_name(doc, c.elements[0])}' follows '{_name(doc, c.elements[1])}'",
                f"'{_name(doc, c.elements[0])}' is placed above '{_name(doc, c.elements[1])}'",
                {"y": a.new_bbox.y, "other_y": b.new_bbox.y},
            )
        elif c.type == "min_text_size" and c.elements:
            eid = c.elements[0]
            px = typography.get(eid, {}).get("font_px")
            floor = int(c.params.get("px", 10))
            if px is None:
                why = "not placed" if visible(eid) is None else "rendered size unknown"
                unchecked(eid, f"'{_name(doc, eid)}' {why}")
                continue
            emit(
                px >= floor, eid,
                f"'{_name(doc, eid)}' is {px}px (minimum {floor}px)",
                f"'{_name(doc, eid)}' is {px}px, below the minimum {floor}px",
                {"font_px": px, "min_px": floor}, soft_severity=Severity.MAJOR,
            )
        elif c.type == "clear_space" and c.elements:
            eid = c.elements[0]
            lr = visible(eid)
            if lr is None:
                unchecked(eid, f"'{_name(doc, eid)}' is not placed")
                continue
            ratio = float(c.params.get("ratio", 0.5))
            pad = int(lr.new_bbox.height * ratio)
            zone = BoundingBox(
                lr.new_bbox.x - pad,
                lr.new_bbox.y - pad,
                lr.new_bbox.width + 2 * pad,
                lr.new_bbox.height + 2 * pad,
            )
            intruders = []
            for other in layout:
                if other.element_id == eid or not other.visible:
                    continue
                de = _doc_elem(doc, other.element_id)
                if de is None or de.is_background:
                    continue
                if _overlap(zone, other.new_bbox) > 0:
                    intruders.append(other.element_id)
            emit(
                not intruders, eid,
                f"'{_name(doc, eid)}' has its clear space",
                f"'{_name(doc, eid)}' clear space is crowded by "
                + ", ".join(_name(doc, i) for i in intruders[:3]),
                {"intruders": intruders},
            )
        elif c.type == "keep_group" and len(c.elements) >= 2:
            boxes = [visible(e).new_bbox for e in c.elements if visible(e) is not None]
            if len(boxes) < 2:
                unchecked(c.elements[0], "fewer than two members are placed")
                continue
            # Members should be near each other: gap smaller than the taller member.
            boxes_sorted = sorted(boxes, key=lambda b: b.y)
            max_gap = 0
            for prev, cur in zip(boxes_sorted, boxes_sorted[1:], strict=False):
                max_gap = max(max_gap, cur.y - prev.y2)
            limit = max(b.height for b in boxes) * 1.5
            emit(
                max_gap <= limit, c.elements[0],
                "Grouped elements stay together", "Grouped elements drifted apart",
                {"max_gap": max_gap, "limit": limit},
            )
        elif c.type == "anchor_edge" and c.elements:
            eid = c.elements[0]
            lr = visible(eid)
            edge = str(c.params.get("edge", "")).lower()
            if lr is None:
                unchecked(eid, f"'{_name(doc, eid)}' is not placed")
                continue
            if edge not in ("top", "bottom", "left", "right"):
                unchecked(eid, f"unknown edge '{edge}'")
                continue
            b = lr.new_bbox
            tol = 2 * int(0.04 * min(tw, th))
            dist = {
                "top": b.y, "bottom": th - b.y2, "left": b.x, "right": tw - b.x2,
            }[edge]
            emit(
                0 <= dist <= tol, eid,
                f"'{_name(doc, eid)}' sits on the {edge} edge",
                f"'{_name(doc, eid)}' is {dist}px from the {edge} edge (allowed {tol}px)",
                {"edge": edge, "distance": dist, "tolerance": tol},
            )
        elif c.type == "scale_range" and c.elements:
            eid = c.elements[0]
            lr = visible(eid)
            if lr is None:
                unchecked(eid, f"'{_name(doc, eid)}' is not placed")
                continue
            lo = float(c.params.get("min", 0.0))
            hi = float(c.params.get("max", 1.0))
            rel = lr.new_bbox.height / max(1, th)
            emit(
                lo - 0.01 <= rel <= hi + 0.01, eid,
                f"'{_name(doc, eid)}' is {rel:.2f} of the canvas height (allowed {lo}-{hi})",
                f"'{_name(doc, eid)}' is {rel:.2f} of the canvas height, outside {lo}-{hi}",
                {"relative_height": round(rel, 4), "min": lo, "max": hi},
            )
        elif c.type == "allowed_overlap":
            continue  # a permission consumed by the overlap check, not a rule to verify
        elif hard:
            unchecked(c.elements[0] if c.elements else "", f"unsupported rule type '{c.type}'")
    return out


_CHECK_SUFFIX = {
    "order_below": "order",
    "keep_visible": "keep_visible",
    "min_text_size": "min_text_size",
    "clear_space": "clear_space",
    "keep_group": "keep_group",
    "anchor_edge": "anchor_edge",
    "scale_range": "scale_range",
}


def _repair(
    doc: DesignDocument,
    elements: list[DesignElement],
    layout: list[LayoutResult],
    report: QualityReport,
    target: tuple[int, int],
    reg: FontRegistry,
    typography: dict[str, dict],
    by_id: dict[str, DesignElement],
) -> tuple[bool, list[str]]:
    """Bounded, local repairs for the most common rendered failures.

    Policy (cheapest, least visible change first):
    1. shrink the text's font towards its minimum;
    2. move the text clear of the blocker inside the safe area;
    3. shrink the non-text blocker uniformly (never below 60% of its plan);
    4. move the non-text blocker.
    """
    steps: list[str] = []
    layout_map = {r.element_id: r for r in layout}
    doc_by_id = {e.id: e for e in doc.elements}
    w, h = target
    margin = int(0.04 * min(w, h))
    changed = False
    handled: set[str] = set()

    problems = [
        c
        for c in report.checks
        if c.check_id in ("element_visible", "text_overlap")
        and c.status == CheckStatus.FAIL
        and c.subject_id
    ]
    for chk in problems:
        eid = chk.subject_id
        if eid in handled:
            continue
        lr = layout_map.get(eid)
        elem = by_id.get(eid)
        if lr is None or elem is None:
            continue
        blocker_id = chk.details.get("other_id") if chk.check_id == "text_overlap" else None
        if blocker_id is None:
            blocker = _find_blocker(lr, layout, elements, eid)
        else:
            blocker = layout_map.get(blocker_id)
        if blocker is None:
            continue
        b_elem = by_id.get(blocker.element_id)
        if b_elem is None:
            continue
        handled.add(eid)

        text_side = (
            (elem, lr)
            if elem.layer_type == "type"
            else ((b_elem, blocker) if b_elem.layer_type == "type" else None)
        )
        other_side = (b_elem, blocker) if text_side and text_side[0] is elem else (elem, lr)

        # 1) shrink text
        if text_side is not None:
            t_elem, t_lr = text_side
            info = typography.get(t_elem.id, {})
            font_px = int(info.get("font_px", 0))
            min_px = int(info.get("min_px", 0))
            if font_px > min_px:
                cap = max(min_px, int(font_px * 0.85))
                _typeset_one(
                    t_elem,
                    t_lr,
                    doc_by_id.get(t_elem.id),
                    target,
                    reg,
                    typography,
                    doc.constraints,
                    (doc.canvas_width, doc.canvas_height),
                    max_px_cap=cap,
                )
                steps.append(
                    f"shrink_text:{t_elem.id}:{font_px}->{typography[t_elem.id]['font_px']}"
                )
                changed = True
                if _overlap(t_lr.new_bbox, other_side[1].new_bbox) == 0:
                    continue
            # 2) move text (never onto another element)
            obstacles = _obstacles_for(layout, elements, {t_elem.id, other_side[0].id})
            if _move_clear(t_lr, other_side[1], w, h, margin, obstacles):
                steps.append(f"move:{t_elem.id}:clear_of:{other_side[0].id}")
                changed = True
                continue

        # 3) shrink the non-text element (or the lower-priority one when both are images)
        victim_elem, victim_lr = (
            other_side
            if text_side is not None
            else ((elem, lr) if _priority(elem) >= _priority(b_elem) else (b_elem, blocker))
        )
        partner_lr = lr if victim_lr is blocker else blocker
        if victim_elem.layer_type != "type" and _shrink(victim_lr, victim_elem, w, h, margin):
            steps.append(f"shrink:{victim_elem.id}")
            changed = True
            if _overlap(victim_lr.new_bbox, partner_lr.new_bbox) == 0:
                continue
        # 4) move the non-text element
        obstacles = _obstacles_for(layout, elements, {victim_elem.id, partner_lr.element_id})
        if victim_elem.layer_type != "type" and _move_clear(
            victim_lr, partner_lr, w, h, margin, obstacles
        ):
            steps.append(f"move:{victim_elem.id}:clear_of:{partner_lr.element_id}")
            changed = True

    return changed, steps


def _priority(elem: DesignElement | None) -> int:
    return elem.priority if elem is not None else 9


def _find_blocker(
    lr: LayoutResult, layout: list[LayoutResult], elements: list[DesignElement], eid: str
) -> LayoutResult | None:
    z = {e.id: e.z_index for e in elements}
    roles = {e.id: e.role for e in elements}
    best, best_area = None, 0
    for other in layout:
        if other.element_id == eid or not other.visible:
            continue
        if roles.get(other.element_id) in BACKGROUND_ROLES:
            continue
        if z.get(other.element_id, 0) < z.get(eid, 0):
            continue
        area = _overlap(lr.new_bbox, other.new_bbox)
        if area > best_area:
            best, best_area = other, area
    return best


def _overlap(a: BoundingBox, b: BoundingBox) -> int:
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    return (ix2 - ix1) * (iy2 - iy1) if ix1 < ix2 and iy1 < iy2 else 0


def _move_clear(
    victim: LayoutResult,
    other: LayoutResult,
    w: int,
    h: int,
    margin: int,
    obstacles: list[BoundingBox] | None = None,
) -> bool:
    """Move ``victim`` next to ``other`` without landing on any obstacle."""
    v, o = victim.new_bbox, other.new_bbox
    gap = max(8, int(0.02 * h))
    candidates = [
        BoundingBox(v.x, o.y - v.height - gap, v.width, v.height),  # above
        BoundingBox(v.x, o.y2 + gap, v.width, v.height),  # below
        BoundingBox(o.x - v.width - gap, v.y, v.width, v.height),  # left
        BoundingBox(o.x2 + gap, v.y, v.width, v.height),  # right
    ]
    blockers = [o] + list(obstacles or [])
    for c in candidates:
        if c.x < margin or c.y < margin or c.x2 > w - margin or c.y2 > h - margin:
            continue
        if all(_overlap(c, b) == 0 for b in blockers):
            victim.new_bbox = c
            return True
    return False


def _obstacles_for(
    layout: list[LayoutResult], elements: list[DesignElement], exclude: set[str]
) -> list[BoundingBox]:
    roles = {e.id: e.role for e in elements}
    return [
        r.new_bbox
        for r in layout
        if r.visible
        and r.element_id not in exclude
        and roles.get(r.element_id) not in BACKGROUND_ROLES
    ]


def _compact_groups(doc: DesignDocument, layout: list[LayoutResult]) -> list[str]:
    """Tighten vertical gaps inside keep_group constraints after typesetting.

    Planned boxes are often taller than the text finally needs; members of a
    group are re-stacked so they read as one unit.
    """
    steps: list[str] = []
    layout_map = {r.element_id: r for r in layout}
    for c in doc.constraints:
        if not c.enabled or c.type != "keep_group" or len(c.elements) < 2:
            continue
        members = [layout_map[e] for e in c.elements if e in layout_map and layout_map[e].visible]
        if len(members) < 2:
            continue
        members.sort(key=lambda r: r.new_bbox.y)
        moved = False
        for prev, cur in zip(members, members[1:], strict=False):
            gap = max(6, int(prev.new_bbox.height * 0.35))
            wanted = prev.new_bbox.y2 + gap
            if cur.new_bbox.y > wanted:
                cur.new_bbox = BoundingBox(
                    cur.new_bbox.x, wanted, cur.new_bbox.width, cur.new_bbox.height
                )
                moved = True
        if moved:
            steps.append(f"compact_group:{c.id}")
    return steps


_SHRINK_FLOOR = 0.6


def _shrink(victim: LayoutResult, elem: DesignElement, w: int, h: int, margin: int) -> bool:
    b = victim.new_bbox
    factor = 0.85
    planned = getattr(victim, "_planned_area", None)
    if planned is None:
        victim._planned_area = b.area  # type: ignore[attr-defined]
        planned = b.area
    new_w, new_h = max(1, int(b.width * factor)), max(1, int(b.height * factor))
    if new_w * new_h < planned * _SHRINK_FLOOR * _SHRINK_FLOOR:
        return False
    if new_w == b.width and new_h == b.height:
        return False
    cx, cy = b.center
    nb = BoundingBox(cx - new_w // 2, cy - new_h // 2, new_w, new_h)
    nb.x = max(margin, min(w - margin - nb.width, nb.x))
    nb.y = max(margin, min(h - margin - nb.height, nb.y))
    victim.new_bbox = nb
    victim.scale_factor *= factor
    return True


def _name(doc: DesignDocument, eid: str) -> str:
    de = _doc_elem(doc, eid)
    return de.name if de is not None else eid


def _doc_elem(doc: DesignDocument, eid: str) -> Element | None:
    for e in doc.elements:
        if e.id == eid:
            return e
    return None


def _font_disclosure(typography: dict[str, dict]) -> list[dict]:
    """Every face the variant used (one entry per requested family and status).

    Multi-run text contributes each of its runs' faces, so a substituted bold face
    inside an otherwise available family is disclosed, not hidden behind run 0.
    """
    seen: dict[tuple[str, str], dict] = {}
    for t in typography.values():
        for r in t.get("runs") or [t]:
            key = (r["font_family_requested"], r["font_status"])
            seen.setdefault(
                key,
                {
                    "requested": r["font_family_requested"],
                    "used": r["font_family_used"],
                    "status": r["font_status"],
                },
            )
    return list(seen.values())
