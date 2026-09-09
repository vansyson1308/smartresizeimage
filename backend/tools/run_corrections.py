"""H5 measurement: do rules derived from corrections stop the same rejection recurring?

A rule-based stand-in for a reviewer rejects variants for two reasons ("logo too
small" when the logo is under a fraction of the canvas height, "CTA hard to read"
when the CTA font is under a fraction of the canvas height). Campaigns are the
fixture cases in order, all for one synthetic brand. For each campaign the tool:

1. generates the size set and lets the reviewer decide;
2. for each rejection, applies the designer's fix (the matching constraint),
   re-plans, and approves if the reviewer accepts — the rejection/approval pair
   is then fed to ``derive_corrections``;
3. carries the derived rules (keyed by role) into the next campaigns when memory
   is on.

Reported: reviewer rejections per campaign without and with memory, and repeat
rejections (a reason that already occurred in an earlier campaign). The reviewer
is a rule, not a person; this measures whether the mechanism closes the loop,
not whether the rules are what a brand wants.

Usage:
    python backend/tools/run_corrections.py --outdir /tmp/corrections
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.design.adapter import document_from_elements
from backend.app.design.assets import AssetStore
from backend.app.design.corrections import (
    CorrectionProposal,
    RejectionSnapshot,
    derive_corrections,
)
from backend.app.design.document import Constraint, DesignDocument, Provenance, new_id
from backend.app.design.serialize import document_from_dict, document_to_dict
from backend.app.design.variant import VariantBrief, generate_variant
from backend.app.quality import CONTRACT_VERSION, QualityConfig
from backend.app.quality.evaluate import environment_fingerprint
from backend.tools.generate_bench_fixtures import generate_fixtures
from backend.tools.run_ablations import _native_elements
from backend.tools.run_layout_bench import _elements_from_meta

SIZES = [(1200, 628), (1080, 1080), (300, 250)]
LOGO_MIN_FRACTION = 0.09  # reviewer: logo height / canvas height below this -> reject
CTA_MIN_FRACTION = 0.032  # reviewer: CTA font px / canvas height below this -> reject


@dataclass
class Decision:
    campaign: str
    size: str
    memory: bool
    rejected: bool
    reason: str
    repeat: bool  # the same reason already occurred in an earlier campaign
    rules_carried: int
    elapsed_s: float


def _review(doc: DesignDocument, plan: dict, size: tuple[int, int]) -> str:
    """The stand-in reviewer: returns a rejection reason or ''."""
    th = size[1]
    boxes = {p["element_id"]: p for p in plan.get("placements", []) if p.get("visible", True)}
    typo = plan.get("typography", {}) or {}
    for e in doc.elements:
        if e.role == "logo" and e.id in boxes and boxes[e.id]["height"] / th < LOGO_MIN_FRACTION:
            return "logo too small"
    for e in doc.elements:
        if e.role == "cta" and e.id in typo and typo[e.id]["font_px"] / th < CTA_MIN_FRACTION:
            return "CTA hard to read"
    return ""


def _designer_fix(doc: DesignDocument, reason: str, size: tuple[int, int]) -> DesignDocument:
    """What the designer does after rejecting: add the constraint that fixes it."""
    d = document_from_dict(document_to_dict(doc))
    prov = Provenance(origin="user", confidence=1.0, notes="designer fix in the H5 protocol")
    for e in d.elements:
        if reason == "logo too small" and e.role == "logo":
            d.add_constraint(
                Constraint(
                    id=new_id("c"),
                    type="scale_range",
                    elements=[e.id],
                    params={"min": LOGO_MIN_FRACTION + 0.01, "max": 1.0},
                    hard=False,
                    provenance=prov,
                )
            )
        if reason == "CTA hard to read" and e.role == "cta":
            d.add_constraint(
                Constraint(
                    id=new_id("c"),
                    type="min_text_size",
                    elements=[e.id],
                    params={"px": int(CTA_MIN_FRACTION * size[1]) + 1},
                    hard=False,
                    provenance=prov,
                )
            )
    return d


def _apply_memory(doc: DesignDocument, memory: dict[tuple[str, str], CorrectionProposal]) -> None:
    """Carry brand rules (keyed by kind + role) into a new campaign's document."""
    for (_kind, role), prop in memory.items():
        for e in doc.elements:
            if e.role == role:
                c = prop.to_constraint()
                c.elements = [e.id]
                doc.add_constraint(c)


def run(cases: list[Path], sizes: list[tuple[int, int]], memory_on: bool) -> list[Decision]:
    decisions: list[Decision] = []
    memory: dict[tuple[str, str], CorrectionProposal] = {}
    seen_reasons: set[str] = set()
    qc = QualityConfig(run_ocr=False)
    for case_dir in cases:
        meta = json.loads((case_dir / "metadata.json").read_text())
        source = Image.open(case_dir / "input.png").convert("RGBA")
        bg_path = case_dir / "background.png"
        background = Image.open(bg_path).convert("RGBA") if bg_path.exists() else source
        elements = _native_elements(_elements_from_meta(meta, source, background))
        source_size = (meta["source_size"]["width"], meta["source_size"]["height"])
        with tempfile.TemporaryDirectory() as tmp:
            store = AssetStore(Path(tmp) / "assets")
            doc = document_from_elements(
                elements, source_size, store, name=case_dir.name, origin="fixture"
            )
            if memory_on:
                _apply_memory(doc, memory)
            roles = {e.id: e.role for e in doc.elements}
            for w, h in sizes:
                t0 = time.perf_counter()
                res = generate_variant(
                    doc,
                    store,
                    VariantBrief(w, h, name="v"),
                    quality_config=qc,
                    planner="constraints",
                )
                reason = _review(doc, res.plan, (w, h))
                decisions.append(
                    Decision(
                        campaign=case_dir.name,
                        size=f"{w}x{h}",
                        memory=memory_on,
                        rejected=bool(reason),
                        reason=reason,
                        repeat=bool(reason) and reason in seen_reasons,
                        rules_carried=len(memory),
                        elapsed_s=round(time.perf_counter() - t0, 3),
                    )
                )
                if not reason:
                    continue
                # designer fix -> approved follow-up -> derived rule into brand memory
                rejection = RejectionSnapshot(
                    variant_id="v",
                    name="v",
                    width=w,
                    height=h,
                    reason=reason,
                    ts="2026-01-01T00:00:00",
                    placements=res.plan["placements"],
                    typography=res.plan.get("typography", {}),
                )
                fixed_doc = _designer_fix(doc, reason, (w, h))
                fixed = generate_variant(
                    fixed_doc,
                    store,
                    VariantBrief(w, h, name="v"),
                    quality_config=qc,
                    planner="constraints",
                )
                if _review(fixed_doc, fixed.plan, (w, h)):
                    continue  # the designer's fix did not satisfy the reviewer; no pair
                approved = (
                    {"id": "v", "width": w, "height": h, "updated_at": "2026-01-02T00:00:00"},
                    fixed.plan,
                )
                proposals, _ = derive_corrections(doc, [rejection], [approved])
                for prop in proposals:
                    memory[(prop.kind, roles.get(prop.element_id, ""))] = prop
            seen_reasons.update(
                d.reason for d in decisions if d.campaign == case_dir.name and d.reason
            )
        print(
            f"{case_dir.name}: done (memory={'on' if memory_on else 'off'}, rules={len(memory)})",
            flush=True,
        )
    return decisions


def summarize(decisions: list[Decision]) -> dict:
    out: dict = {}
    for memory_on in (False, True):
        subset = [d for d in decisions if d.memory == memory_on]
        if not subset:
            continue
        key = "memory_on" if memory_on else "memory_off"
        out[key] = {
            "variants": len(subset),
            "rejections": sum(1 for d in subset if d.rejected),
            "repeat_rejections": sum(1 for d in subset if d.repeat),
            "by_reason": {
                r: sum(1 for d in subset if d.reason == r)
                for r in sorted({d.reason for d in subset if d.reason})
            },
            "mean_s": round(sum(d.elapsed_s for d in subset) / len(subset), 3),
        }
    return out


def build_report(decisions: list[Decision], run_meta: dict) -> str:
    s = summarize(decisions)
    lines = [
        "# Rules from corrections (H5): repeat rejections with and without brand memory",
        "",
        f"Contract v{CONTRACT_VERSION}. Reviewer = rule (logo height < {LOGO_MIN_FRACTION:.2f} "
        f"of canvas height → 'logo too small'; CTA font < {CTA_MIN_FRACTION:.3f} of canvas "
        "height → 'CTA hard to read'). Campaigns = fixture cases in order, one synthetic "
        "brand; the designer's fix after a rejection is the matching constraint; derived "
        "rules are carried by role into later campaigns when memory is on.",
        "",
        f"- environment: `{json.dumps(run_meta['environment'], sort_keys=True)}`",
        f"- git commit: `{run_meta.get('git_commit')}`",
        f"- sizes: {', '.join(run_meta['sizes'])}; campaigns: {run_meta['cases']}",
        "",
        "| Memory | Variants | Rejections | Repeat rejections | By reason | Mean s |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for key, m in s.items():
        by = ", ".join(f"{r}: {n}" for r, n in m["by_reason"].items()) or "-"
        lines.append(
            f"| {key} | {m['variants']} | {m['rejections']} | {m['repeat_rejections']} | "
            f"{by} | {m['mean_s']:.2f} |"
        )
    lines += [
        "",
        "## Per campaign",
        "",
        "| Campaign | Rejections (memory off) | Rejections (memory on) | Rules carried |",
        "|---|---:|---:|---:|",
    ]
    for camp in sorted({d.campaign for d in decisions}):
        off = sum(1 for d in decisions if d.campaign == camp and not d.memory and d.rejected)
        on = [d for d in decisions if d.campaign == camp and d.memory]
        lines.append(
            f"| {camp} | {off} | {sum(1 for d in on if d.rejected)} | "
            f"{on[0].rules_carried if on else 0} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- The reviewer is a deterministic rule, so 'repeat rejections' measures whether the "
        "mechanism closes the loop, not whether the rules match a real brand's taste.",
        "- Rules are carried by role across campaigns here; the product stores corrections per "
        "project and applies a rule only after a person adds it.",
        "- OCR is off; verdicts are not part of this measurement.",
    ]
    return "\n".join(lines)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="backend/tests/bench_fixtures")
    parser.add_argument("--outdir", default="backend/tests/fixtures/outputs/corrections")
    parser.add_argument("--cases", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    fixtures = Path(args.fixtures)
    if len(list(fixtures.glob("case_*/metadata.json"))) < args.cases:
        generate_fixtures(fixtures, cases=args.cases, seed=args.seed)
    cases = sorted(p for p in fixtures.glob("case_*") if (p / "metadata.json").exists())
    decisions = run(cases, SIZES, memory_on=False) + run(cases, SIZES, memory_on=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "environment": environment_fingerprint(),
        "git_commit": _git_commit(),
        "sizes": [f"{w}x{h}" for w, h in SIZES],
        "cases": len(cases),
        "contract": CONTRACT_VERSION,
    }
    (outdir / "records.json").write_text(
        json.dumps({"run": run_meta, "decisions": [asdict(d) for d in decisions]}, indent=2)
    )
    (outdir / "report.md").write_text(build_report(decisions, run_meta))
    print(json.dumps(summarize(decisions), indent=2))
    print(f"Report: {outdir / 'report.md'}")


if __name__ == "__main__":
    main()
