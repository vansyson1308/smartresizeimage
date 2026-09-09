"""Ablation harness for the design pipeline.

Runs the same fixture corpus under named configurations with identical seeds
and reports verdict counts, timing and repair rounds per configuration, on the
tuning split and on a frozen holdout split. Every configuration is recorded in
the output so a number can always be traced to its settings.

Usage::

    python backend/tools/run_ablations.py --outdir /tmp/ablations
    python backend/tools/run_ablations.py --configs full,no_repair,no_plates --sizes 1080x1920

Holdout policy: fixture cases whose index is >= --holdout-from (default 10, i.e.
case_10..case_12) are never used for tuning thresholds; they are reported
separately.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from backend.app.config import Config
from backend.app.design.adapter import document_from_elements, elements_from_document
from backend.app.design.assets import AssetStore
from backend.app.design.examples import example_from_plan, learned_families
from backend.app.design.planner import Family, aspect_class, choose_families, plan_layout
from backend.app.design.variant import VariantBrief, generate_variant
from backend.app.models import DesignElement
from backend.app.quality import CONTRACT_VERSION, QualityConfig
from backend.app.quality.evaluate import environment_fingerprint
from backend.app.quality.family import VariantSnapshot, family_consistency_checks
from backend.tools.generate_bench_fixtures import generate_fixtures
from backend.tools.run_layout_bench import _elements_from_meta, _git_commit


@dataclass(frozen=True)
class AblationConfig:
    name: str
    planner: str = "constraints"  # constraints | zones
    repair: bool = True
    plates: bool = True
    ocr: bool = True
    joint: bool = False  # choose one family per orientation for the whole size set
    learned: bool = False  # infer families from one "approved" example per orientation (H1)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "planner": self.planner,
            "repair": self.repair,
            "plates": self.plates,
            "ocr": self.ocr,
            "joint": self.joint,
            "learned": self.learned,
            "notes": self.notes,
        }


CONFIGS: dict[str, AblationConfig] = {
    "full": AblationConfig("full", notes="production defaults"),
    "zones": AblationConfig("zones", planner="zones", notes="legacy zone-template planner"),
    "no_repair": AblationConfig("no_repair", repair=False, notes="verify only, no repair loop"),
    "no_plates": AblationConfig("no_plates", plates=False, notes="text-safe plates disabled"),
    "zones_no_repair": AblationConfig(
        "zones_no_repair", planner="zones", repair=False, notes="zone planner without repair"
    ),
    "no_ocr": AblationConfig(
        "no_ocr", ocr=False, notes="legibility not checked (reports needs_review)"
    ),
    "joint": AblationConfig(
        "joint", joint=True, notes="one family per orientation chosen jointly for the size set"
    ),
    "learned": AblationConfig(
        "learned",
        joint=True,
        learned=True,
        notes="H1: families inferred from one approved example per orientation "
        "(the designer's example uses the second-best hand-written family)",
    ),
    "designer_ref": AblationConfig(
        "designer_ref",
        joint=True,
        notes="reference: the designer's (second-best) family applied directly to all sizes",
    ),
}


@dataclass
class RunRecord:
    config: str
    case: str
    size: str
    split: str
    verdict: str
    accepted: bool
    elapsed_s: float
    repair_steps: int
    family: str | None
    issues: list[str] = field(default_factory=list)
    family_issues: int = 0  # cross-variant consistency checks not passing
    agreement: float | None = None  # mean IoU with the designer-family plan (H1 protocol)
    role: str = "test"  # "example" when the size served as the approved example


def _parse_sizes(raw: str) -> list[tuple[int, int]]:
    out = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            w, h = token.split("x", 1)
            out.append((int(w), int(h)))
    return out


def _native_elements(elements: list[DesignElement]) -> list[DesignElement]:
    native = []
    for e in elements:
        copy = DesignElement(
            id=e.id,
            name=e.name,
            layer_type=e.layer_type,
            bbox=e.bbox,
            image=e.image,
            text_content=e.text_content,
            font_info=e.font_info,
            role=e.role,
            priority=e.priority,
            z_index=e.z_index,
            effects=dict(e.effects),
        )
        copy.effects["_role_source"] = "fixture"
        if copy.layer_type == "type" and copy.text_content:
            copy.image = None
            copy.font_info = {
                "font_name": "DejaVu Sans",
                "font_size": int(max(8, copy.bbox.height * 0.45)),
                "color": [1.0, 0.1, 0.1, 0.1],
            }
        native.append(copy)
    return native


def run_config(
    cfg: AblationConfig,
    cases: list[Path],
    sizes: list[tuple[int, int]],
    holdout_from: int,
) -> list[RunRecord]:
    prev = (Config.TEXT_SAFE_PLATE_ENABLED, Config.LAYOUT_PROFILE_SCORING_ENABLED)
    Config.TEXT_SAFE_PLATE_ENABLED = cfg.plates
    Config.LAYOUT_PROFILE_SCORING_ENABLED = True
    records: list[RunRecord] = []
    try:
        for case_dir in cases:
            meta = json.loads((case_dir / "metadata.json").read_text())
            source = Image.open(case_dir / "input.png").convert("RGBA")
            bg_path = case_dir / "background.png"
            background = Image.open(bg_path).convert("RGBA") if bg_path.exists() else source
            elements = _native_elements(_elements_from_meta(meta, source, background))
            source_size = (meta["source_size"]["width"], meta["source_size"]["height"])
            case_index = int(case_dir.name.split("_")[1])
            split = "holdout" if case_index >= holdout_from else "tuning"
            with tempfile.TemporaryDirectory() as tmp:
                store = AssetStore(Path(tmp) / "assets")
                doc = document_from_elements(
                    elements, source_size, store, name=case_dir.name, origin="fixture"
                )
                families: dict[str, Family] = {}
                designer: dict[str, Family] = {}
                example_sizes: set[tuple[int, int]] = set()
                setup_s = 0.0
                engine_elements = elements_from_document(doc, store)
                if cfg.planner == "constraints":
                    # H1 protocol (same for every config so agreement is comparable):
                    # the designer's family per orientation, and the largest size of
                    # each orientation as the approved example; the other sizes of
                    # that orientation are the held-out sizes.
                    designer = _designer_families(doc, engine_elements, sizes)
                    example_sizes = {
                        sorted(ts, key=lambda t: -(t[0] * t[1]))[0]
                        for ts in _sizes_by_class(sizes).values()
                    }
                if cfg.name == "designer_ref":
                    families = designer
                elif cfg.learned and cfg.planner == "constraints":
                    t_setup = time.perf_counter()
                    examples = [
                        _example_document(doc, store, designer, t) for t in sorted(example_sizes)
                    ]
                    learned = learned_families(doc, examples)
                    families = choose_families(doc, engine_elements, sizes, learned=learned)
                    setup_s = time.perf_counter() - t_setup
                elif cfg.joint and cfg.planner == "constraints":
                    families = choose_families(doc, engine_elements, sizes)
                snapshots: list[VariantSnapshot] = []
                case_records: list[RunRecord] = []
                for w, h in sizes:
                    t0 = time.perf_counter()
                    fam = families.get(aspect_class(w / max(1, h))) if families else None
                    result = generate_variant(
                        doc,
                        store,
                        VariantBrief(w, h, name=f"{w}x{h}"),
                        planner=cfg.planner,
                        max_repairs=3 if cfg.repair else 0,
                        quality_config=QualityConfig(run_ocr=cfg.ocr),
                        family=fam,
                    )
                    elapsed = time.perf_counter() - t0
                    snapshots.append(
                        VariantSnapshot(
                            variant_id=f"{w}x{h}",
                            target=(w, h),
                            family=result.plan.get("planner_meta", {}).get("family"),
                            placements=result.plan.get("placements", []),
                            typography=result.plan.get("typography", {}),
                            roles={e.id: e.role for e in doc.elements},
                            master_px={
                                e.id: float(e.text.primary_style.font_size)
                                for e in doc.elements
                                if e.kind == "text" and e.text is not None
                            },
                        )
                    )
                    agreement = None
                    if designer:
                        ref_fam = designer.get(aspect_class(w / max(1, h)))
                        if ref_fam is not None:
                            ref_plan = plan_layout(doc, engine_elements, (w, h), families=[ref_fam])
                            agreement = _plan_agreement(result.plan.get("placements", []), ref_plan)
                    case_records.append(
                        RunRecord(
                            config=cfg.name,
                            case=case_dir.name,
                            size=f"{w}x{h}",
                            split=split,
                            verdict=result.verdict,
                            accepted=result.verdict == "accepted",
                            elapsed_s=round(
                                elapsed + (setup_s if (w, h) in example_sizes else 0.0), 3
                            ),
                            repair_steps=len(result.repair_steps),
                            family=result.plan.get("planner_meta", {}).get("family"),
                            issues=[c.message for c in result.report.issues()][:4],
                            agreement=agreement,
                            role="example" if (w, h) in example_sizes else "test",
                        )
                    )
                family_checks = family_consistency_checks(snapshots)
                for rec in case_records:
                    checks = family_checks.get(rec.size, [])
                    rec.family_issues = sum(1 for c in checks if c.status.value != "pass")
                records.extend(case_records)
    finally:
        Config.TEXT_SAFE_PLATE_ENABLED, Config.LAYOUT_PROFILE_SCORING_ENABLED = prev
    return records


def _sizes_by_class(sizes: list[tuple[int, int]]) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for w, h in sizes:
        out.setdefault(aspect_class(w / max(1, h)), []).append((w, h))
    return out


def _designer_families(doc, elements, sizes) -> dict[str, Family]:
    """The 'designer's choice': the second-best hand-written family per orientation.

    This stands in for a composition the planner would not pick on its own, so
    the H1 protocol measures whether one approved example is enough to make the
    planner reproduce it on the other sizes.
    """
    from backend.app.design.fonts import default_registry
    from backend.app.design.planner import _plan_family, families_for

    reg = default_registry()
    chosen: dict[str, Family] = {}
    for cls, ts in _sizes_by_class(sizes).items():
        candidates = families_for(ts[0][0] / max(1, ts[0][1]))
        scored = sorted(
            candidates,
            key=lambda fam: -sum(_plan_family(doc, elements, t, fam, reg).score for t in ts),
        )
        chosen[cls] = scored[1] if len(scored) > 1 else scored[0]
    return chosen


def _example_document(doc, store, designer: dict[str, Family], target: tuple[int, int]):
    """Render an 'approved' variant with the designer family and turn its plan into a document."""
    fam = designer[aspect_class(target[0] / max(1, target[1]))]
    result = generate_variant(
        doc,
        store,
        VariantBrief(target[0], target[1], name="example"),
        planner="constraints",
        family=fam,
        quality_config=QualityConfig(run_ocr=False),
    )
    return example_from_plan(
        doc,
        target,
        result.plan.get("placements", []),
        result.plan.get("typography", {}),
        example_id=doc.id + "_ex",
    )


def _plan_agreement(placements: list[dict], ref_plan) -> float:
    """Mean IoU between this plan's boxes and the designer-family plan's boxes."""
    ref = {r.element_id: r.new_bbox for r in ref_plan.layout if r.visible}
    ious = []
    for p in placements:
        b = ref.get(p["element_id"])
        if b is None or not p.get("visible", True):
            continue
        ax1, ay1, ax2, ay2 = p["x"], p["y"], p["x"] + p["width"], p["y"] + p["height"]
        ix1, iy1 = max(ax1, b.x), max(ay1, b.y)
        ix2, iy2 = min(ax2, b.x2), min(ay2, b.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = p["width"] * p["height"] + b.area - inter
        ious.append(inter / union if union > 0 else 0.0)
    return round(sum(ious) / len(ious), 4) if ious else 0.0


def summarize(records: list[RunRecord]) -> dict:
    out: dict = {}
    for split in ("tuning", "holdout", "all"):
        subset = [r for r in records if split == "all" or r.split == split]
        if not subset:
            continue
        n = len(subset)
        out[split] = {
            "runs": n,
            "accepted": sum(1 for r in subset if r.verdict == "accepted"),
            "needs_review": sum(1 for r in subset if r.verdict == "needs_review"),
            "failed": sum(1 for r in subset if r.verdict == "failed"),
            "acceptance_rate": round(sum(1 for r in subset if r.accepted) / n, 4),
            "mean_s": round(sum(r.elapsed_s for r in subset) / n, 3),
            "mean_repair_steps": round(sum(r.repair_steps for r in subset) / n, 3),
            "family_issue_runs": sum(1 for r in subset if r.family_issues > 0),
            "agreement_heldout_sizes": _mean(
                [r.agreement for r in subset if r.agreement is not None and r.role == "test"]
            ),
        }
    return out


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return round((centre - margin) / denom, 4), round((centre + margin) / denom, 4)


def build_report(results: dict[str, list[RunRecord]], run_meta: dict) -> str:
    lines = [
        "# Design pipeline ablations",
        "",
        f"Contract v{CONTRACT_VERSION}. Same fixtures, seeds and sizes for every configuration; "
        "holdout cases were never used for tuning. Wilson 95% intervals on acceptance.",
        "",
        f"- environment: `{json.dumps(run_meta['environment'], sort_keys=True)}`",
        f"- git commit: `{run_meta['git_commit']}`",
        f"- sizes: {', '.join(run_meta['sizes'])}; cases: {run_meta['n_cases']} "
        f"(holdout from case {run_meta['holdout_from']})",
        "",
        "| Config | Split | Runs | Accepted | Needs review | Failed | "
        "Acceptance (95% CI) | Mean s | Repair steps | Family-issue runs | "
        "Agreement w/ designer (held-out sizes) |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for name, records in results.items():
        summary = summarize(records)
        for split in ("tuning", "holdout"):
            s = summary.get(split)
            if not s:
                continue
            lo, hi = _wilson(s["accepted"], s["runs"])
            lines.append(
                f"| {name} | {split} | {s['runs']} | {s['accepted']} | {s['needs_review']} | "
                f"{s['failed']} | {s['acceptance_rate']:.2f} ({lo:.2f}–{hi:.2f}) | "
                f"{s['mean_s']:.2f} | {s['mean_repair_steps']:.2f} | {s['family_issue_runs']} | "
                f"{'-' if s['agreement_heldout_sizes'] is None else s['agreement_heldout_sizes']} |"
            )
    lines += ["", "## Configurations", ""]
    for name in results:
        cfg = CONFIGS[name]
        lines.append(
            f"- `{name}`: planner={cfg.planner}, repair={cfg.repair}, plates={cfg.plates}, "
            f"ocr={cfg.ocr}, joint={cfg.joint}, learned={cfg.learned} — {cfg.notes}"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `no_ocr` cannot reach `accepted` by contract (a critical check did not run); it "
        "measures how much OCR costs, not quality.",
        "- Synthetic fixtures; not customer validation.",
        "- H1 protocol: `designer_ref` plans every size with the second-best hand-written "
        "family per orientation (a stand-in for a composition the planner would not pick "
        "on its own). `learned` sees exactly one designer_ref variant per aspect class as "
        "the approved example (the largest size of that class) and must reproduce the "
        "designer's plan on the other sizes of that class. `Agreement` is the mean IoU of "
        "planned element boxes against the designer plan on those held-out sizes; it is "
        "reported for every constraint-planner config, so `joint`/`full` give the "
        "no-example counterfactual and `designer_ref` the ceiling (1.0 by construction). "
        "`Mean s` for `learned` includes the setup (render example, match, infer) on the "
        "example sizes.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="backend/tests/bench_fixtures")
    parser.add_argument("--outdir", default="backend/tests/fixtures/outputs/ablations")
    parser.add_argument("--configs", default="full,zones,no_repair,no_plates,zones_no_repair")
    parser.add_argument("--sizes", default="1200x628,1080x1080,1080x1920,300x250")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-from", type=int, default=10)
    parser.add_argument(
        "--cases",
        type=int,
        default=15,
        help="fixture cases to ensure (15 puts a busy-background case into the holdout)",
    )
    args = parser.parse_args()

    fixtures = Path(args.fixtures)
    if len(list(fixtures.glob("case_*/metadata.json"))) < args.cases:
        # Deterministic per case index: regenerating never changes existing cases.
        generate_fixtures(fixtures, cases=args.cases, seed=args.seed)
    cases = sorted(p for p in fixtures.glob("case_*") if (p / "metadata.json").exists())
    sizes = _parse_sizes(args.sizes)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "contract_version": CONTRACT_VERSION,
        "seed": args.seed,
        "sizes": [f"{w}x{h}" for w, h in sizes],
        "n_cases": len(cases),
        "holdout_from": args.holdout_from,
        "environment": environment_fingerprint(),
        "python": platform.python_version(),
        "git_commit": _git_commit(),
    }
    results: dict[str, list[RunRecord]] = {}
    started = time.perf_counter()
    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        cfg = CONFIGS[name]
        records = run_config(cfg, cases, sizes, args.holdout_from)
        results[name] = records
        (outdir / f"{name}.json").write_text(
            json.dumps(
                {
                    "config": cfg.to_dict(),
                    "run": run_meta,
                    "summary": summarize(records),
                    "records": [r.__dict__ for r in records],
                },
                indent=1,
            )
        )
        print(f"{name}: {summarize(records)}")
    run_meta["wall_time_s"] = round(time.perf_counter() - started, 2)
    (outdir / "report.md").write_text(build_report(results, run_meta))
    (outdir / "run.json").write_text(json.dumps(run_meta, indent=1))
    print(f"Ablations completed. Report: {outdir / 'report.md'}")


if __name__ == "__main__":
    main()
