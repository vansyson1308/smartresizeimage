"""H3 measurement: how local are campaign revisions, with and without a reference plan.

For every fixture case and one size per orientation, generate a base variant, apply a
revision (copy change on the headline, longer CTA copy, logo asset swap, subheadline
colour change), regenerate with and without the base plan as reference, and measure the
fraction of pixels outside the edited element's scope (its box before and after, padded)
that changed, plus how many other elements moved.

Usage:
    python backend/tools/run_local_edits.py --outdir /tmp/local_edits
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

import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.design.adapter import document_from_elements
from backend.app.design.assets import AssetStore
from backend.app.design.serialize import document_from_dict, document_to_dict
from backend.app.design.variant import VariantBrief, generate_variant
from backend.app.quality import CONTRACT_VERSION, QualityConfig
from backend.app.quality.evaluate import environment_fingerprint
from backend.tools.generate_bench_fixtures import generate_fixtures
from backend.tools.run_ablations import _native_elements
from backend.tools.run_layout_bench import _elements_from_meta

SCOPE_PAD = 36
SIZES = [(1200, 628), (1080, 1080), (1080, 1920)]


@dataclass
class EditRecord:
    case: str
    size: str
    edit: str
    mode: str  # "reference" | "replan"
    out_of_scope_diff: float
    moved_elements: int
    replanned: int
    verdict: str
    elapsed_s: float


def _out_of_scope_diff(before, after, edited: set[str], size: tuple[int, int]) -> float:
    mask = np.ones((size[1], size[0]), dtype=bool)
    for result in (before, after):
        for p in result.plan["placements"]:
            if p["element_id"] in edited:
                x1, y1 = max(0, p["x"] - SCOPE_PAD), max(0, p["y"] - SCOPE_PAD)
                x2 = min(size[0], p["x"] + p["width"] + SCOPE_PAD)
                y2 = min(size[1], p["y"] + p["height"] + SCOPE_PAD)
                mask[y1:y2, x1:x2] = False
    diff = ImageChops.difference(before.image.convert("RGB"), after.image.convert("RGB"))
    changed = np.asarray(diff).max(axis=2) > 0
    return float((changed & mask).sum() / max(1, mask.sum()))


def _moved(before, after, edited: set[str]) -> int:
    b = {
        p["element_id"]: (p["x"], p["y"], p["width"], p["height"])
        for p in before.plan["placements"]
        if p["visible"]
    }
    a = {
        p["element_id"]: (p["x"], p["y"], p["width"], p["height"])
        for p in after.plan["placements"]
        if p["visible"]
    }
    return sum(1 for k in b if k not in edited and a.get(k) != b[k])


def _edits(doc, store):
    """Yield (name, edited_ids, document, brief_kwargs) revisions of ``doc``."""
    texts = [e for e in doc.elements if e.kind == "text" and e.text is not None]
    by_role = {e.role: e for e in texts}
    logos = [e for e in doc.elements if e.role == "logo" and e.asset is not None]
    if "headline" in by_role:
        d = document_from_dict(document_to_dict(doc))
        e = d.element(by_role["headline"].id)
        e.text.replace_text("NEW " + e.text.plain)
        yield "headline_copy", {e.id}, d, {}
    if "cta" in by_role:
        e = by_role["cta"]
        yield "cta_longer", {e.id}, doc, {"text_overrides": {e.id: e.text.plain + " TODAY"}}
    if logos:
        d = document_from_dict(document_to_dict(doc))
        e = d.element(logos[0].id)
        img = store.get(e.asset)
        swapped = Image.new("RGBA", img.size, (40, 40, 40, 255))
        e.asset = store.put(swapped, "logo_swap")
        yield "logo_swap", {e.id}, d, {}
    if "subheadline" in by_role:
        d = document_from_dict(document_to_dict(doc))
        e = d.element(by_role["subheadline"].id)
        for run in e.text.runs:
            run.style.color = "#ffd166"
        yield "sub_colour", {e.id}, d, {}


def run(cases: list[Path], sizes: list[tuple[int, int]]) -> list[EditRecord]:
    records: list[EditRecord] = []
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
            for w, h in sizes:
                base = generate_variant(
                    doc,
                    store,
                    VariantBrief(w, h, name="base"),
                    quality_config=qc,
                    planner="constraints",
                )
                for name, edited, d, kw in _edits(doc, store):
                    for mode in ("reference", "replan"):
                        t0 = time.perf_counter()
                        res = generate_variant(
                            d,
                            store,
                            VariantBrief(w, h, name=name, **kw),
                            quality_config=qc,
                            planner="constraints",
                            reference=base.plan if mode == "reference" else None,
                        )
                        ref_meta = res.plan["planner_meta"].get("reference") or {}
                        records.append(
                            EditRecord(
                                case=case_dir.name,
                                size=f"{w}x{h}",
                                edit=name,
                                mode=mode,
                                out_of_scope_diff=round(
                                    _out_of_scope_diff(base, res, edited, (w, h)), 5
                                ),
                                moved_elements=_moved(base, res, edited),
                                replanned=len(ref_meta.get("replanned", [])),
                                verdict=res.verdict,
                                elapsed_s=round(time.perf_counter() - t0, 3),
                            )
                        )
        print(f"{case_dir.name}: done", flush=True)
    return records


def summarize(records: list[EditRecord]) -> dict:
    out: dict = {}
    for mode in ("reference", "replan"):
        subset = [r for r in records if r.mode == mode]
        if not subset:
            continue
        out[mode] = {
            "runs": len(subset),
            "zero_out_of_scope": sum(1 for r in subset if r.out_of_scope_diff == 0.0),
            "mean_out_of_scope_diff": round(
                sum(r.out_of_scope_diff for r in subset) / len(subset), 5
            ),
            "runs_with_moved_elements": sum(1 for r in subset if r.moved_elements > 0),
            "accepted": sum(1 for r in subset if r.verdict == "accepted"),
            "replanned_runs": sum(1 for r in subset if r.replanned > 0),
            "mean_s": round(sum(r.elapsed_s for r in subset) / len(subset), 3),
        }
        by_edit: dict[str, dict] = {}
        for edit in sorted({r.edit for r in subset}):
            es = [r for r in subset if r.edit == edit]
            by_edit[edit] = {
                "runs": len(es),
                "zero_out_of_scope": sum(1 for r in es if r.out_of_scope_diff == 0.0),
                "runs_with_moved_elements": sum(1 for r in es if r.moved_elements > 0),
            }
        out[mode]["by_edit"] = by_edit
    return out


def build_report(records: list[EditRecord], run_meta: dict) -> str:
    s = summarize(records)
    lines = [
        "# Local edits (H3): out-of-scope change on campaign revisions",
        "",
        f"Contract v{CONTRACT_VERSION}. Same fixtures and sizes for both modes; `reference` "
        "keeps the base variant's plan, `replan` plans from scratch. Scope = the edited "
        f"element's box before and after, padded by {SCOPE_PAD}px.",
        "",
        f"- environment: `{json.dumps(run_meta['environment'], sort_keys=True)}`",
        f"- git commit: `{run_meta.get('git_commit')}`",
        f"- sizes: {', '.join(run_meta['sizes'])}; cases: {run_meta['cases']}",
        "",
        "| Mode | Runs | Zero out-of-scope diff | Mean out-of-scope diff | "
        "Runs with moved elements | Accepted | Re-planned (copy no longer fit) | Mean s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, m in s.items():
        lines.append(
            f"| {mode} | {m['runs']} | {m['zero_out_of_scope']} | "
            f"{m['mean_out_of_scope_diff']:.4f} | {m['runs_with_moved_elements']} | "
            f"{m['accepted']} | {m['replanned_runs']} | {m['mean_s']:.2f} |"
        )
    lines += [
        "",
        "## By edit",
        "",
        "| Mode | Edit | Runs | Zero out-of-scope diff | Runs with moved elements |",
        "|---|---|---:|---:|---:|",
    ]
    for mode, m in s.items():
        for edit, e in m["by_edit"].items():
            lines.append(
                f"| {mode} | {edit} | {e['runs']} | {e['zero_out_of_scope']} | "
                f"{e['runs_with_moved_elements']} |"
            )
    lines += [
        "",
        "## Notes",
        "",
        "- Out-of-scope diff is a pixel measurement on synthetic fixtures; it says whether a "
        "revision touched anything but the edited element, not whether the result is good.",
        "- `Re-planned` counts revisions where the new copy did not fit its previous box and the "
        "element fell back to a fresh plan (reported as `layout_change`).",
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
    parser.add_argument("--outdir", default="backend/tests/fixtures/outputs/local_edits")
    parser.add_argument("--cases", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    fixtures = Path(args.fixtures)
    if len(list(fixtures.glob("case_*/metadata.json"))) < args.cases:
        generate_fixtures(fixtures, cases=args.cases, seed=args.seed)
    cases = sorted(p for p in fixtures.glob("case_*") if (p / "metadata.json").exists())
    records = run(cases, SIZES)
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
        json.dumps({"run": run_meta, "records": [asdict(r) for r in records]}, indent=2)
    )
    (outdir / "report.md").write_text(build_report(records, run_meta))
    print(json.dumps(summarize(records), indent=2))
    print(f"Report: {outdir / 'report.md'}")


if __name__ == "__main__":
    main()
