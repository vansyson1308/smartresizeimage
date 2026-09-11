"""Run Phase 2.1 layout benchmark and generate artifacts + report."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from backend.app.config import Config
from backend.app.enums import ElementRole
from backend.app.layout.bench_metrics import evaluate_bench_run
from backend.app.layout.profiles import pick_profile
from backend.app.layout.solver import render_layout_debug_overlay
from backend.app.layout.typography import load_font
from backend.app.models import BoundingBox, DesignElement
from backend.app.quality import CONTRACT_VERSION, Verdict
from backend.app.quality.evaluate import environment_fingerprint
from backend.app.relayout import ReLayoutEngine
from backend.tools.generate_bench_fixtures import generate_fixtures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="backend/tests/bench_fixtures")
    parser.add_argument("--outdir", default="backend/tests/fixtures/outputs/bench_phase21")
    parser.add_argument(
        "--mode",
        choices=["baseline", "phase21", "phase3", "design", "both"],
        default="both",
        help="design = document/native-text pipeline (backend.app.design.variant)",
    )
    parser.add_argument("--sizes", default="1200x628,1080x1920,1080x1080")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="Phase 3 candidate count; defaults to the production value "
        f"(Config.PHASE3_N_CANDIDATES={Config.PHASE3_N_CANDIDATES}) and is recorded.",
    )
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
    n_candidates = int(args.n_candidates or Config.PHASE3_N_CANDIDATES)

    run_config = {
        "contract_version": CONTRACT_VERSION,
        "seed": int(args.seed),
        "phase3_n_candidates": n_candidates,
        "phase3_n_candidates_is_production_default": n_candidates == Config.PHASE3_N_CANDIDATES,
        "sizes": [f"{w}x{h}" for w, h in sizes],
        "cases": [c.name for c in cases],
        "mode": args.mode,
        "environment": environment_fingerprint(),
        "git_commit": _git_commit(),
    }

    runs: list[dict] = []
    started = time.perf_counter()
    for case_dir in cases:
        meta = json.loads((case_dir / "metadata.json").read_text())
        source_img = Image.open(case_dir / "input.png").convert("RGBA")
        # Fixtures generated since contract v2 ship a clean background layer so
        # the hero/logo are not baked into the background (duplicate subjects).
        bg_path = case_dir / "background.png"
        background_img = Image.open(bg_path).convert("RGBA") if bg_path.exists() else source_img

        elements = _elements_from_meta(meta, source_img, background_img)
        source_size = (meta["source_size"]["width"], meta["source_size"]["height"])

        for size in sizes:
            size_tag = f"{size[0]}x{size[1]}"
            run_dir = outdir / meta["case_name"] / size_tag
            run_dir.mkdir(parents=True, exist_ok=True)

            if args.mode in ("baseline", "both"):
                base = _run_one_mode(elements, source_size, size, mode="baseline", out_dir=run_dir)
                base["image"].save(run_dir / "before.png")
                runs.append(base["record"])

            if args.mode in ("phase21", "phase3", "design", "both"):
                phase_mode = "phase21" if args.mode == "both" else args.mode
                p21 = _run_one_mode(
                    elements,
                    source_size,
                    size,
                    mode=phase_mode,
                    out_dir=run_dir,
                    busy_expected=bool(meta.get("tags", {}).get("busy_background", False)),
                    n_candidates=n_candidates,
                )
                p21["image"].save(run_dir / "after.png")
                _write_json(run_dir / "layout_debug.json", p21["layout_debug"])
                if p21["overlay"] is not None:
                    p21["overlay"].save(run_dir / "overlay.png")
                runs.append(p21["record"])

    run_config["wall_time_s"] = round(time.perf_counter() - started, 2)
    report_path = outdir / "report.md"
    report_path.write_text(_build_report(runs, outdir, run_config))
    return report_path


def _run_one_mode(
    elements: list[DesignElement],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    mode: str,
    out_dir: Path,
    busy_expected: bool = False,
    n_candidates: int | None = None,
) -> dict:
    """Run one (case, size, mode) through the production ``ReLayoutEngine``.

    The benchmark deliberately shares the production code path (layout,
    composition, harmonization, quality evaluation) so that a number reported
    here describes what users get. Config toggles are recorded in the record.
    """
    if mode == "design":
        return _run_design_mode(elements, source_size, target_size, out_dir, busy_expected)

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
    Config.LAYOUT_DEBUG_ENABLED = False
    Config.GENERATIVE_BG_ENABLED = False
    Config.GENERATIVE_DECOR_POLICY = "OFF"
    n_cand = int(n_candidates or Config.PHASE3_N_CANDIDATES)

    try:
        engine = ReLayoutEngine(use_ai=False)
        engine.load_elements(elements, source_size)

        t0 = time.perf_counter()
        if mode == "phase3":
            result = engine.relayout_redesign(target_size, manual_anchors=None, n_candidates=n_cand)
        else:
            result = engine.relayout(target_size)
        elapsed = time.perf_counter() - t0

        layout = result.layout_results
        layout_engine = engine.layout_engine
        profile = pick_profile(*target_size)

        text_plate_meta = dict(result.metadata.get("text_plate", {}))
        text_plate_meta.setdefault("busy_threshold", Config.TEXT_SAFE_BUSY_THRESHOLD)
        legacy = evaluate_bench_run(
            elements=elements,
            layout_results=layout,
            target_size=target_size,
            text_plate_meta=text_plate_meta,
            busy_expected=busy_expected,
        )

        quality = result.quality
        if quality is None:
            raise RuntimeError("production pipeline did not attach a quality report")

        role_by_id = {e.id: e.role for e in elements}
        overlay = None
        if use_phase21:
            overlay = render_layout_debug_overlay(target_size, layout, role_by_id)

        config_flags = {
            "layout_profile_scoring": Config.LAYOUT_PROFILE_SCORING_ENABLED,
            "text_safe_plate": Config.TEXT_SAFE_PLATE_ENABLED,
            "generative_bg": Config.GENERATIVE_BG_ENABLED,
            "decor_policy": Config.GENERATIVE_DECOR_POLICY,
            "phase3_n_candidates": n_cand if mode == "phase3" else None,
        }

        record = {
            "case": out_dir.parent.name,
            "size": f"{target_size[0]}x{target_size[1]}",
            "mode": mode,
            # Contract v2: accepted only when every implemented check passed.
            "passed": quality.verdict == Verdict.ACCEPTED,
            "verdict": quality.verdict.value,
            "fail_reasons": [
                f"{c.check_id}:{c.subject_id or 'canvas'}" for c in quality.failed
            ],
            "issues": [
                {
                    "check": c.check_id,
                    "subject": c.subject_id,
                    "status": c.status.value,
                    "severity": c.severity.value,
                    "message": c.message,
                }
                for c in quality.issues()
            ],
            "quality_summary": quality.summary,
            "used_fallback": bool(result.used_fallback),
            "pipeline_fail_reasons": list(result.fail_reasons),
            "legacy_v1": {
                "passed": legacy.passed,
                "fail_reasons": legacy.fail_reasons,
                "metrics": asdict(legacy.metrics),
            },
            "metrics": asdict(legacy.metrics),
            "profile": profile.name,
            "elapsed_s": round(elapsed, 3),
            "config": config_flags,
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
            "score": legacy.metrics.total_score,
            "violations": legacy.metrics.violations,
            "metrics": asdict(legacy.metrics),
            "repair_applied": bool(layout_engine.last_layout_debug.get("repair_applied", False)),
            "repair_steps": list(layout_engine.last_layout_debug.get("repair_steps", [])),
            "fallback_used": bool(layout_engine.last_layout_debug.get("fallback_used", False)),
            "fallback_reason": str(layout_engine.last_layout_debug.get("fallback_reason", "")),
            "typography": list(layout_engine.last_layout_debug.get("typography", [])),
            "text_plate": text_plate_meta,
            "redesign": dict(result.metadata.get("redesign", {})),
            "quality": quality.to_dict(),
            "config": config_flags,
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
            "quality": quality,
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


def _run_design_mode(
    elements: list[DesignElement],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    out_dir: Path,
    busy_expected: bool,
) -> dict:
    """Run the document-based pipeline: native text, constraints, verify+repair."""
    import tempfile

    from backend.app.design.adapter import document_from_elements
    from backend.app.design.assets import AssetStore
    from backend.app.design.variant import VariantBrief, generate_variant

    prev_scoring = Config.LAYOUT_PROFILE_SCORING_ENABLED
    Config.LAYOUT_PROFILE_SCORING_ENABLED = True
    try:
        native: list[DesignElement] = []
        for e in elements:
            copy = DesignElement(
                id=e.id, name=e.name, layer_type=e.layer_type, bbox=e.bbox, image=e.image,
                text_content=e.text_content, font_info=e.font_info, role=e.role,
                priority=e.priority, z_index=e.z_index, effects=dict(e.effects),
            )
            copy.effects["_role_source"] = "fixture"
            if copy.layer_type == "type" and copy.text_content:
                # Fixture text becomes native text (the raster was only a stand-in).
                copy.image = None
                copy.font_info = {
                    "font_name": "DejaVu Sans",
                    "font_size": int(max(8, copy.bbox.height * 0.45)),
                    "color": [1.0, 0.1, 0.1, 0.1],
                }
            native.append(copy)
        with tempfile.TemporaryDirectory() as tmp:
            store = AssetStore(Path(tmp) / "assets")
            doc = document_from_elements(native, source_size, store, name=out_dir.parent.name,
                                         origin="fixture")
            t0 = time.perf_counter()
            result = generate_variant(
                doc, store, VariantBrief(target_size[0], target_size[1], name="bench")
            )
            elapsed = time.perf_counter() - t0
        quality = result.report
        legacy = evaluate_bench_run(
            elements=result.elements,
            layout_results=result.layout,
            target_size=target_size,
            text_plate_meta={"applied": False, "busy_threshold": Config.TEXT_SAFE_BUSY_THRESHOLD},
            busy_expected=busy_expected,
        )
        profile = pick_profile(*target_size)
        role_by_id = {e.id: e.role for e in result.elements}
        overlay = render_layout_debug_overlay(target_size, result.layout, role_by_id)
        config_flags = {
            "pipeline": "design.variant",
            "layout_profile_scoring": True,
            "native_text": True,
            "repair_steps": result.repair_steps,
        }
        record = {
            "case": out_dir.parent.name,
            "size": f"{target_size[0]}x{target_size[1]}",
            "mode": "design",
            "passed": quality.verdict == Verdict.ACCEPTED,
            "verdict": quality.verdict.value,
            "fail_reasons": [f"{c.check_id}:{c.subject_id or 'canvas'}" for c in quality.failed],
            "issues": [
                {
                    "check": c.check_id,
                    "subject": c.subject_id,
                    "status": c.status.value,
                    "severity": c.severity.value,
                    "message": c.message,
                }
                for c in quality.issues()
            ],
            "quality_summary": quality.summary,
            "used_fallback": False,
            "pipeline_fail_reasons": [],
            "legacy_v1": {
                "passed": legacy.passed,
                "fail_reasons": legacy.fail_reasons,
                "metrics": asdict(legacy.metrics),
            },
            "metrics": asdict(legacy.metrics),
            "profile": profile.name,
            "elapsed_s": round(elapsed, 3),
            "config": config_flags,
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
            "mode": "design",
            "score": legacy.metrics.total_score,
            "violations": legacy.metrics.violations,
            "metrics": asdict(legacy.metrics),
            "repair_applied": bool(result.repair_steps),
            "repair_steps": list(result.repair_steps),
            "fallback_used": False,
            "fallback_reason": str(result.plan.get("layout_fallback", "")),
            "typography": result.plan.get("typography", {}),
            "text_plate": {"applied": False, "busy_threshold": Config.TEXT_SAFE_BUSY_THRESHOLD},
            "redesign": {},
            "quality": quality.to_dict(),
            "config": config_flags,
            "plan": result.plan,
            "results": result.plan.get("placements", []),
        }
        return {
            "image": result.image,
            "record": record,
            "layout_debug": layout_debug,
            "overlay": overlay,
            "quality": quality,
        }
    finally:
        Config.LAYOUT_PROFILE_SCORING_ENABLED = prev_scoring


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _elements_from_meta(
    meta: dict,
    source_img: Image.Image,
    background_img: Image.Image | None = None,
) -> list[DesignElement]:
    elements: list[DesignElement] = []
    bg_source = background_img if background_img is not None else source_img
    for item in meta["elements"]:
        bbox = BoundingBox(**item["bbox"])
        role = ElementRole(item["role"])
        text = item.get("text")
        kind = item.get("kind", "pixel")

        if role == ElementRole.BACKGROUND:
            img = bg_source.copy()
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


def _build_report(runs: list[dict], outdir: Path, run_config: dict | None = None) -> str:
    phase_runs = [r for r in runs if r["mode"] == "phase21"]
    for alt in ("phase3", "design"):
        if not phase_runs:
            phase_runs = [r for r in runs if r["mode"] == alt]
    total = len(phase_runs)
    accepted = sum(1 for r in phase_runs if r["verdict"] == "accepted")
    review = sum(1 for r in phase_runs if r["verdict"] == "needs_review")
    failed = sum(1 for r in phase_runs if r["verdict"] == "failed")
    legacy_passed = sum(1 for r in phase_runs if r["legacy_v1"]["passed"])
    pass_rate = (accepted / total * 100.0) if total else 0.0
    baseline_runs = [r for r in runs if r["mode"] == "baseline"]
    baseline_accepted = sum(1 for r in baseline_runs if r["verdict"] == "accepted")

    rows = []
    for r in sorted(phase_runs, key=lambda x: (x["case"], x["size"])):
        top = "; ".join(i["message"] for i in r["issues"][:2]) or "-"
        rows.append(
            "| {case} | {size} | {verdict} | {legacy} | {score:.1f} | {top} | {after} |".format(
                case=r["case"],
                size=r["size"],
                verdict=r["verdict"].upper(),
                legacy="PASS" if r["legacy_v1"]["passed"] else "FAIL",
                score=r["legacy_v1"]["metrics"]["total_score"],
                top=top.replace("|", "/"),
                after=r["artifacts"]["after"],
            )
        )

    worst = sorted(
        phase_runs,
        key=lambda r: ({"failed": 0, "needs_review": 1, "accepted": 2}[r["verdict"]], r["case"]),
    )[:10]

    report_mode = phase_runs[0]["mode"] if phase_runs else "phase21"
    report_title = {
        "phase3": "Phase 3 Benchmark Report",
        "design": "Design Pipeline (native text) Benchmark Report",
    }.get(report_mode, "Phase 2.1 Benchmark Report")
    cfg = run_config or {}
    report = [
        f"# {report_title}",
        "",
        f"Quality contract v{CONTRACT_VERSION} — rendered-output verdicts. "
        "`accepted` requires every implemented check to pass; `needs_review` and "
        "`failed` are not successes.",
        "",
        f"- Total {report_mode} runs: **{total}**",
        f"- Accepted: **{accepted}** ({pass_rate:.1f}%)",
        f"- Needs review: **{review}**",
        f"- Failed: **{failed}**",
        f"- Legacy v1 layout-metadata pass count (for historical comparison only): "
        f"**{legacy_passed}/{total}**",
    ]
    if baseline_runs:
        report.append(
            "- Baseline mode (template only) accepted: "
            f"**{baseline_accepted}/{len(baseline_runs)}**"
        )
    if cfg:
        report.extend(
            [
                "",
                "## Run configuration",
                "",
                f"- seed: `{cfg.get('seed')}`",
                f"- phase3_n_candidates: `{cfg.get('phase3_n_candidates')}` "
                f"(production default: {cfg.get('phase3_n_candidates_is_production_default')})",
                f"- environment: `{json.dumps(cfg.get('environment', {}), sort_keys=True)}`",
                f"- git commit: `{cfg.get('git_commit')}`",
                f"- wall time: `{cfg.get('wall_time_s')}s`",
            ]
        )
    report.extend(
        [
            "",
            f"## Results table ({report_mode})",
            "",
            "| Case | Size | Verdict | Legacy v1 | Legacy score | Top issues | After |",
            "|---|---:|---|---|---:|---|---|",
            *rows,
            "",
            "## Weakest runs",
            "",
        ]
    )

    for r in worst:
        if r["verdict"] == "accepted":
            continue
        report.append(
            f"- `{r['case']}` `{r['size']}` verdict={r['verdict']} "
            f"issues={'; '.join(i['message'] for i in r['issues'][:3]) or 'n/a'} "
            f"debug={r['artifacts']['layout_debug']}"
        )

    report.append("")
    report.append("## Raw summary JSON")
    summary_json = outdir / "summary.json"
    _write_json(summary_json, {"run_config": cfg, "runs": runs})
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
