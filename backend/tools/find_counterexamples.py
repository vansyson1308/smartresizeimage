"""Search for counterexamples: renders the verdict accepts but an independent oracle rejects.

The quality contract judges rendered output with its own checks. This tool re-judges
the same renders with *independent* oracles that share no code path with the planner
or the checks they audit:

    overlap   opaque pixels of two content elements (their rendered masks at the planned
              boxes) overlap by more than 5% of the smaller one, and no allowed_overlap rule
    bounds    a visible content box leaves the canvas by more than 1 px
    order     a hard order_below [a, b] rule (a below b) is violated by the final boxes
    clear     a hard clear_space rule is violated by the final boxes
    subject   the rendered subject/logo region correlates below 0.9 with the master asset

A run whose verdict is ``accepted`` while an oracle fires is a counterexample; runs whose
verdict is ``failed`` while no oracle fires are listed as possible false failures for a
human to look at. Perturbations (copy length, hidden elements, hard rules, directions)
are seeded, so every counterexample can be replayed.

    .venv/bin/python backend/tools/find_counterexamples.py --cases 15 --outdir /tmp/cex
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from backend.app.design.adapter import document_from_elements  # noqa: E402
from backend.app.design.assets import AssetStore  # noqa: E402
from backend.app.design.document import Constraint, Provenance, new_id  # noqa: E402
from backend.app.design.variant import (  # noqa: E402
    VariantBrief,
    _masked_correlation,
    generate_variant,
)
from backend.app.models import BoundingBox  # noqa: E402
from backend.app.quality import QualityConfig  # noqa: E402
from backend.tools.generate_bench_fixtures import generate_fixtures  # noqa: E402
from backend.tools.run_ablations import _native_elements  # noqa: E402
from backend.tools.run_layout_bench import _elements_from_meta, _git_commit  # noqa: E402

CONTENT_ROLES_EXCLUDED = {"background", "background_pattern", "overlay"}
LONG_COPY = " and everything you need for the whole season at prices you will remember"


def _perturb(doc, rng: random.Random, kind: str) -> tuple[dict, dict]:
    """Return (brief kwargs, description) for one perturbation of the master."""
    texts = [e for e in doc.elements if e.kind == "text" and e.text is not None]
    subjects = [e for e in doc.elements if e.role in ("hero_image", "photo", "illustration")]
    logos = [e for e in doc.elements if e.role == "logo"]
    brief: dict = {}
    desc: dict = {"kind": kind}
    if kind == "long_copy" and texts:
        e = rng.choice(texts)
        brief["text_overrides"] = {e.id: e.text.plain + LONG_COPY}
        desc["element"] = e.id
    elif kind == "short_copy" and texts:
        e = rng.choice(texts)
        brief["text_overrides"] = {e.id: e.text.plain.split(" ")[0] or "GO"}
        desc["element"] = e.id
    elif kind == "hide_subject" and subjects:
        brief["hidden_elements"] = [subjects[0].id]
        desc["element"] = subjects[0].id
    elif kind == "hard_order" and len(texts) >= 2:
        a, b = rng.sample(texts, 2)
        doc.add_constraint(Constraint(
            id=new_id("c"), type="order_below", elements=[a.id, b.id], hard=True,
            provenance=Provenance(origin="user", confidence=1.0, notes="counterexample sweep"),
        ))
        desc["rule"] = f"order_below {a.id} < {b.id}"
    elif kind == "hard_clear" and logos:
        doc.add_constraint(Constraint(
            id=new_id("c"), type="clear_space", elements=[logos[0].id],
            params={"ratio": 1.0}, hard=True,
            provenance=Provenance(origin="user", confidence=1.0, notes="counterexample sweep"),
        ))
        desc["rule"] = f"clear_space {logos[0].id} ratio 1.0"
    elif kind == "direction":
        choice = rng.choice(["copy", "subject", "text-left", "text-right", "subject-top",
                             "center"])
        from backend.app.design.grammar import parse_direction

        brief["direction"] = parse_direction(choice)
        desc["direction"] = choice
    return brief, desc


def _masks(result, target) -> dict[str, np.ndarray]:
    """Opaque-pixel masks of visible content elements at their planned boxes."""
    tw, th = target
    out: dict[str, np.ndarray] = {}
    by_id = {e.id: e for e in result.elements}
    for r in result.layout:
        e = by_id.get(r.element_id)
        if e is None or not r.visible or e.image is None:
            continue
        if e.role.value in CONTENT_ROLES_EXCLUDED:
            continue
        b = r.new_bbox
        if b.width < 1 or b.height < 1:
            continue
        alpha = np.asarray(e.image.convert("RGBA").resize((b.width, b.height)), dtype=np.uint8)
        alpha = alpha[..., 3] > 32
        canvas = np.zeros((th, tw), dtype=bool)
        x1, y1 = max(0, b.x), max(0, b.y)
        x2, y2 = min(tw, b.x2), min(th, b.y2)
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = alpha[y1 - b.y:y2 - b.y, x1 - b.x:x2 - b.x]
        out[r.element_id] = canvas
    return out


def _oracles(doc, result, target) -> list[str]:
    tw, th = target
    fired: list[str] = []
    boxes = {r.element_id: r.new_bbox for r in result.layout if r.visible}
    roles = {e.id: e.role for e in doc.elements}
    content = {eid: b for eid, b in boxes.items() if roles.get(eid) not in CONTENT_ROLES_EXCLUDED}
    # bounds
    for eid, b in content.items():
        if b.x < -1 or b.y < -1 or b.x2 > tw + 1 or b.y2 > th + 1:
            fired.append(f"bounds:{eid}")
    # overlap of rendered masks
    masks = _masks(result, target)
    allowed = doc.allowed_overlaps()
    ids = list(masks)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if (a, b) in allowed or (b, a) in allowed:
                continue
            inter = int(np.logical_and(masks[a], masks[b]).sum())
            smaller = max(1, min(int(masks[a].sum()), int(masks[b].sum())))
            if inter / smaller > 0.05:
                fired.append(f"overlap:{a}+{b}:{inter / smaller:.2f}")
    # hard rules on the final boxes
    for c in doc.constraints:
        if not c.enabled or not c.hard:
            continue
        if c.type == "order_below" and len(c.elements) == 2:
            # order_below [a, b]: a sits below b (its top edge at or under b's top edge)
            a, b = content.get(c.elements[0]), content.get(c.elements[1])
            if a is not None and b is not None and a.y < b.y:
                fired.append(f"order:{c.elements[0]}>{c.elements[1]}")
        elif c.type == "clear_space" and c.elements:
            lb = content.get(c.elements[0])
            if lb is None:
                continue
            pad = int(lb.height * float(c.params.get("ratio", 0.5)))
            zone = BoundingBox(lb.x - pad, lb.y - pad, lb.width + 2 * pad, lb.height + 2 * pad)
            for eid, ob in content.items():
                if eid == c.elements[0]:
                    continue
                ix = max(0, min(zone.x2, ob.x2) - max(zone.x, ob.x))
                iy = max(0, min(zone.y2, ob.y2) - max(zone.y, ob.y))
                if ix * iy > 0:
                    fired.append(f"clear:{c.elements[0]}~{eid}")
        elif c.type == "keep_visible" and c.elements and c.elements[0] not in content:
            fired.append(f"visible:{c.elements[0]}")
    # subject correlation (independent of the check: recomputed here)
    by_id = {e.id: e for e in result.elements}
    for eid, b in content.items():
        e = by_id.get(eid)
        de = next((d for d in doc.elements if d.id == eid), None)
        if e is None or de is None or de.kind != "image" or e.image is None:
            continue
        if de.role not in ("hero_image", "photo", "illustration", "icon", "logo"):
            continue
        x1, y1, x2, y2 = max(0, b.x), max(0, b.y), min(tw, b.x2), min(th, b.y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        expected = e.image.convert("RGBA").resize((b.width, b.height), Image.LANCZOS)
        expected = expected.crop((x1 - b.x, y1 - b.y, x2 - b.x, y2 - b.y))
        actual = result.image.convert("RGBA").crop((x1, y1, x2, y2))
        corr = _masked_correlation(expected, actual)
        if corr is not None and corr < 0.9:
            fired.append(f"subject:{eid}:{corr:.2f}")
    return fired


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", default="backend/tests/bench_fixtures")
    ap.add_argument("--cases", type=int, default=15)
    ap.add_argument("--sizes", default="1200x628,1080x1080,1080x1920,300x250")
    ap.add_argument("--perturbations", default="none,long_copy,short_copy,hide_subject,"
                                               "hard_order,hard_clear,direction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="/tmp/autobanner_counterexamples")
    ap.add_argument("--ocr", action="store_true", help="run OCR checks too (slower)")
    args = ap.parse_args()

    fixtures = Path(args.fixtures)
    if not fixtures.exists() or len(list(fixtures.glob("case_*/metadata.json"))) < args.cases:
        generate_fixtures(fixtures, cases=args.cases, seed=args.seed)
    cases = sorted(p for p in fixtures.glob("case_*") if (p / "metadata.json").exists())
    cases = cases[: args.cases]
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    kinds = [k.strip() for k in args.perturbations.split(",") if k.strip()]
    rng = random.Random(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    t_start = time.perf_counter()
    for case_dir in cases:
        meta = json.loads((case_dir / "metadata.json").read_text())
        source = Image.open(case_dir / "input.png").convert("RGBA")
        bg_path = case_dir / "background.png"
        background = Image.open(bg_path).convert("RGBA") if bg_path.exists() else source
        source_size = (meta["source_size"]["width"], meta["source_size"]["height"])
        for kind in kinds:
            for w, h in sizes:
                elements = _native_elements(_elements_from_meta(meta, source, background))
                with tempfile.TemporaryDirectory() as tmp:
                    store = AssetStore(Path(tmp) / "assets")
                    doc = document_from_elements(
                        elements, source_size, store, name=case_dir.name, origin="fixture"
                    )
                    if kind == "none":
                        brief_kw, desc = {}, {"kind": "none"}
                    else:
                        brief_kw, desc = _perturb(doc, rng, kind)
                    result = generate_variant(
                        doc, store, VariantBrief(w, h, name=f"{w}x{h}", **brief_kw),
                        planner="constraints", quality_config=QualityConfig(run_ocr=args.ocr),
                    )
                    fired = _oracles(doc, result, (w, h))
                    issues = [f"{c.check_id}:{c.subject_id}" for c in result.report.issues()][:6]
                    runs.append({
                        "case": case_dir.name, "size": f"{w}x{h}", "perturbation": desc,
                        "verdict": result.verdict, "oracles": fired, "issues": issues,
                        "family": result.plan.get("planner_meta", {}).get("family"),
                    })
    elapsed = time.perf_counter() - t_start

    counter = [r for r in runs if r["verdict"] == "accepted" and r["oracles"]]
    false_fail = [r for r in runs if r["verdict"] == "failed" and not r["oracles"]
                  and not any(i.startswith(("text_fits", "font_coverage")) for i in r["issues"])]
    summary = {
        "runs": len(runs), "elapsed_s": round(elapsed, 1), "git_commit": _git_commit(),
        "verdicts": {v: sum(1 for r in runs if r["verdict"] == v)
                     for v in ("accepted", "needs_review", "failed")},
        "oracle_fired_runs": sum(1 for r in runs if r["oracles"]),
        "counterexamples": len(counter),
        "possible_false_failures": len(false_fail),
        "by_oracle": {},
    }
    for r in runs:
        for f in r["oracles"]:
            key = f.split(":")[0]
            summary["by_oracle"].setdefault(key, {"fired": 0, "accepted": 0})
            summary["by_oracle"][key]["fired"] += 1
            if r["verdict"] == "accepted":
                summary["by_oracle"][key]["accepted"] += 1
    (outdir / "runs.json").write_text(json.dumps(runs, indent=1, ensure_ascii=False))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Counterexample search",
        "",
        f"- runs: {summary['runs']} ({len(cases)} cases × {len(kinds)} perturbations × "
        f"{len(sizes)} sizes), seed {args.seed}, OCR {'on' if args.ocr else 'off'}, "
        f"{summary['elapsed_s']} s, commit `{summary['git_commit']}`",
        f"- verdicts: {summary['verdicts']}",
        f"- runs where an oracle fired: {summary['oracle_fired_runs']}; by oracle "
        f"(fired / of which accepted): "
        + ", ".join(f"{k} {v['fired']}/{v['accepted']}" for k, v in summary["by_oracle"].items()),
        f"- **counterexamples (accepted, oracle fired): {len(counter)}**",
        f"- possible false failures (failed, no oracle, no text-fit/coverage issue): "
        f"{len(false_fail)}",
        "",
    ]
    if counter:
        lines += ["## Counterexamples", ""]
        for r in counter[:60]:
            lines.append(f"- {r['case']} {r['size']} {r['perturbation']} family={r['family']}: "
                         f"{', '.join(r['oracles'])}; checks flagged {r['issues'] or 'nothing'}")
    if false_fail:
        lines += ["", "## Possible false failures", ""]
        for r in false_fail[:30]:
            lines.append(f"- {r['case']} {r['size']} {r['perturbation']}: {r['issues']}")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
