"""Run Phase 2.1 layout benchmark and generate artifacts + report."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from backend.app.composition.engine import CompositionEngine
from backend.app.config import Config
from backend.app.enums import ElementRole
from backend.app.layout.bench_metrics import evaluate_bench_run
from backend.app.layout.profiles import pick_profile
from backend.app.layout.solver import render_layout_debug_overlay
from backend.app.layout.typography import load_font
from backend.app.models import BoundingBox, DesignElement
from backend.tools.generate_bench_fixtures import generate_fixtures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="backend/tests/bench_fixtures")
    parser.add_argument("--outdir", default="backend/tests/fixtures/outputs/bench_phase21")
    parser.add_argument("--mode", choices=["baseline", "phase21", "phase3", "both"], default="both")
    parser.add_argument("--sizes", default="1200x628,1080x1920,1080x1080")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate", action="store_true", help="Generate fixtures before run")
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)

    fixtures_dir = Path(args.fixtures)
    has_metadata = bool(list(fixtures_dir.glob("case_*/metadata.json")))
    if args.generate or not fixtures_dir.exists() or not has_metadata:
        generate_fixtures(fixtures_dir, cases=12, seed=args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sizes = _parse_sizes(args.sizes)
    cases = _select_cases(fixtures_dir, args.cases)

    runs: list[dict] = []
    for case_dir in cases:
        meta = json.loads((case_dir / "metadata.json").read_text())
        source_img = Image.open(case_dir / "input.png").convert("RGBA")

        elements = _elements_from_meta(meta, source_img)
        source_size = (meta["source_size"]["width"], meta["source_size"]["height"])

        for size in sizes:
            size_tag = f"{size[0]}x{size[1]}"
            run_dir = outdir / meta["case_name"] / size_tag
            run_dir.mkdir(parents=True, exist_ok=True)

            if args.mode in ("baseline", "both"):
                base = _run_one_mode(elements, source_size, size, mode="baseline", out_dir=run_dir)
                base["image"].save(run_dir / "before.png")
                runs.append(base["record"])

            if args.mode in ("phase21", "phase3", "both"):
                phase_mode = "phase21" if args.mode == "both" else args.mode
                p21 = _run_one_mode(
                    elements,
                    source_size,
                    size,
                    mode=phase_mode,
                    out_dir=run_dir,
                    busy_expected=bool(meta.get("tags", {}).get("busy_background", False)),
                )
                p21["image"].save(run_dir / "after.png")
                _write_json(run_dir / "layout_debug.json", p21["layout_debug"])
                if p21["overlay"] is not None:
                    p21["overlay"].save(run_dir / "overlay.png")
                runs.append(p21["record"])

    report_path = outdir / "report.md"
    report_path.write_text(_build_report(runs, outdir))
    return report_path


def _run_one_mode(
    elements: list[DesignElement],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    mode: str,
    out_dir: Path,
    busy_expected: bool = False,
) -> dict:
    use_phase21 = mode in ("phase21", "phase3")
    prev_cfg = (
        Config.LAYOUT_PROFILE_SCORING_ENABLED,
        Config.TEXT_SAFE_PLATE_ENABLED,
        Config.LAYOUT_DEBUG_ENABLED,
        Config.LAYOUT_DEBUG_DIR,
        Config.GENERATIVE_BG_ENABLED,
        Config.GENERATIVE_DECOR_POLICY,
    )

    Config.LAYOUT_PROFILE_SCORING_ENABLED = use_phase21
    Config.TEXT_SAFE_PLATE_ENABLED = use_phase21
    Config.LAYOUT_DEBUG_ENABLED = use_phase21
    Config.LAYOUT_DEBUG_DIR = str(out_dir)
    Config.GENERATIVE_BG_ENABLED = False
    Config.GENERATIVE_DECOR_POLICY = "OFF"

    try:
        from backend.app.layout.engine import LayoutEngine

        layout_engine = LayoutEngine()
        layout = layout_engine.calculate_layout(elements, source_size, target_size)

        if mode == "phase3":
            from backend.app.redesign.api import run_target_first_redesign

            result = run_target_first_redesign(
                elements=elements,
                layout_results=layout,
                source_size=source_size,
                target_size=target_size,
                manual_anchors=None,
                n_candidates=4,
            )
        else:
            compositor = CompositionEngine(use_ai_inpainting=False)
            result = compositor.compose(elements, layout, source_size, target_size)

        profile = pick_profile(*target_size)
        eval_res = evaluate_bench_run(
            elements=elements,
            layout_results=layout,
            target_size=target_size,
            text_plate_meta=result.metadata.get("text_plate", {}),
            busy_expected=busy_expected,
        )

        role_by_id = {e.id: e.role for e in elements}
        overlay = None
        if use_phase21:
            overlay = render_layout_debug_overlay(target_size, layout, role_by_id)

        record = {
            "case": out_dir.parent.name,
            "size": f"{target_size[0]}x{target_size[1]}",
            "mode": mode,
            "passed": eval_res.passed,
            "fail_reasons": eval_res.fail_reasons,
            "metrics": asdict(eval_res.metrics),
            "profile": profile.name,
            "artifacts": {
                "before": str((out_dir / "before.png").relative_to(out_dir.parent.parent)),
                "after": str((out_dir / "after.png").relative_to(out_dir.parent.parent)),
                "overlay": str((out_dir / "overlay.png").relative_to(out_dir.parent.parent)),
                "layout_debug": str(
                    (out_dir / "layout_debug.json").relative_to(out_dir.parent.parent)
                ),
            },
        }

        layout_debug = {
            "profile": profile.name,
            "profile_name": profile.name,
            "target_size": {"width": target_size[0], "height": target_size[1]},
            "mode": mode,
            "score": eval_res.metrics.total_score,
            "violations": eval_res.metrics.violations,
            "metrics": asdict(eval_res.metrics),
            "repair_applied": bool(layout_engine.last_layout_debug.get("repair_applied", False)),
            "repair_steps": list(layout_engine.last_layout_debug.get("repair_steps", [])),
            "fallback_used": bool(layout_engine.last_layout_debug.get("fallback_used", False)),
            "fallback_reason": str(layout_engine.last_layout_debug.get("fallback_reason", "")),
            "text_plate": dict(result.metadata.get("text_plate", {})),
            "redesign": dict(result.metadata.get("redesign", {})),
            "results": [
                {
                    "element_id": lr.element_id,
                    "bbox": {
                        "x": lr.new_bbox.x,
                        "y": lr.new_bbox.y,
                        "width": lr.new_bbox.width,
                        "height": lr.new_bbox.height,
                    },
                    "visible": lr.visible,
                }
                for lr in layout
            ],
        }

        return {
            "image": result.image,
            "record": record,
            "layout_debug": layout_debug,
            "overlay": overlay,
        }
    finally:
        (
            Config.LAYOUT_PROFILE_SCORING_ENABLED,
            Config.TEXT_SAFE_PLATE_ENABLED,
            Config.LAYOUT_DEBUG_ENABLED,
            Config.LAYOUT_DEBUG_DIR,
            Config.GENERATIVE_BG_ENABLED,
            Config.GENERATIVE_DECOR_POLICY,
        ) = prev_cfg


def _elements_from_meta(meta: dict, source_img: Image.Image) -> list[DesignElement]:
    elements: list[DesignElement] = []
    for item in meta["elements"]:
        bbox = BoundingBox(**item["bbox"])
        role = ElementRole(item["role"])
        text = item.get("text")
        kind = item.get("kind", "pixel")

        if role == ElementRole.BACKGROUND:
            img = source_img.copy()
        elif kind == "hero":
            img = _hero_image((bbox.width, bbox.height))
        elif kind == "logo":
            img = _logo_image((bbox.width, bbox.height))
        elif kind == "text":
            img = _text_block_image((bbox.width, bbox.height), text or item["name"])
        else:
            img = Image.new("RGBA", (bbox.width, bbox.height), (180, 180, 180, 255))

        elements.append(
            DesignElement(
                id=item["id"],
                name=item["name"],
                layer_type=item.get("layer_type", "pixel"),
                bbox=bbox,
                image=img,
                text_content=text,
                role=role,
                priority=int(item.get("priority", 5)),
                z_index=int(item.get("z_index", 0)),
            )
        )
    return elements


def _hero_image(size: tuple[int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, w - 4, h - 4), fill=(236, 88, 88, 255))
    return img


def _logo_image(size: tuple[int, int]) -> Image.Image:
    img = Image.new("RGBA", size, (245, 245, 245, 255))
    d = ImageDraw.Draw(img)
    d.text((10, max(2, size[1] // 3)), "LOGO", fill=(22, 22, 22, 255))
    return img


def _text_block_image(size: tuple[int, int], text: str) -> Image.Image:
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = load_font(None, max(12, int(h * 0.35)))
    d.multiline_text((4, 4), text, fill=(25, 25, 25, 255), font=font, spacing=2)
    return img


def _parse_sizes(raw: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for token in raw.split(","):
        part = token.strip()
        if not part:
            continue
        w_s, h_s = part.split("x", 1)
        sizes.append((int(w_s), int(h_s)))
    return sizes


def _select_cases(fixtures_dir: Path, cases: str) -> list[Path]:
    all_cases = sorted(p for p in fixtures_dir.glob("case_*") if (p / "metadata.json").exists())
    if cases == "all":
        return all_cases
    wanted = {c.strip() for c in cases.split(",") if c.strip()}
    return [p for p in all_cases if p.name in wanted]


def _build_report(runs: list[dict], outdir: Path) -> str:
    phase_runs = [r for r in runs if r["mode"] == "phase21"]
    if not phase_runs:
        phase_runs = [r for r in runs if r["mode"] == "phase3"]
    total = len(phase_runs)
    passed = sum(1 for r in phase_runs if r["passed"])
    pass_rate = (passed / total * 100.0) if total else 0.0

    rows = []
    for r in sorted(phase_runs, key=lambda x: (x["case"], x["size"])):
        rows.append(
            "| {case} | {size} | {score:.1f} | {passed} | {violations} | "
            "{after} | {overlay} |".format(
                case=r["case"],
                size=r["size"],
                score=r["metrics"]["total_score"],
                passed="PASS" if r["passed"] else "FAIL",
                violations=len(r["metrics"]["violations"]),
                after=r["artifacts"]["after"],
                overlay=r["artifacts"]["overlay"],
            )
        )

    worst = sorted(
        phase_runs,
        key=lambda r: (0 if r["passed"] else -1, r["metrics"]["total_score"]),
    )[:10]

    report_mode = phase_runs[0]["mode"] if phase_runs else "phase21"
    report_title = (
        "Phase 3 Benchmark Report"
        if report_mode == "phase3"
        else "Phase 2.1 Benchmark Report"
    )
    report = [
        f"# {report_title}",
        "",
        f"- Total {report_mode} runs: **{total}**",
        f"- Passed: **{passed}**",
        f"- Pass rate: **{pass_rate:.1f}%**",
        "",
        f"## Results table ({report_mode})",
        "",
        "| Case | Size | Score | Status | Violations | After | Overlay |",
        "|---|---:|---:|---|---:|---|---|",
        *rows,
        "",
        "## Top failures / weakest runs",
        "",
    ]

    for r in worst:
        if r["passed"] and len(worst) > 0:
            continue
        report.append(
            f"- `{r['case']}` `{r['size']}` score={r['metrics']['total_score']:.1f} "
            f"reasons={','.join(r['fail_reasons']) or 'n/a'} "
            f"debug={r['artifacts']['layout_debug']}"
        )

    report.append("")
    report.append("## Raw summary JSON")
    summary_json = outdir / "summary.json"
    _write_json(summary_json, {"runs": runs})
    report.append(f"- `{summary_json.name}`")

    return "\n".join(report)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    report = run_benchmark(args)
    print(f"Benchmark completed. Report: {report}")


if __name__ == "__main__":
    main()
