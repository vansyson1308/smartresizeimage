"""Measure a campaign run: R content rows x F formats generated as one job.

The declared workflow is 12 rows x 6 formats = 72 variants. This tool builds a
synthetic layered master (the planner test fixture), submits the campaign through
the real service (joint family planning, per-row consistency checks, durable job
records) and reports wall time, per-variant time, verdicts and family issues.
Synthetic fixture numbers are engineering measurements, not customer validation.

    .venv/bin/python backend/tools/run_campaign.py --rows 12 --formats 6 --outdir /tmp/campaign
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.app.api.service import LOCAL_OWNER, ProjectService  # noqa: E402
from backend.app.design.project import Project  # noqa: E402
from backend.tests.test_planner import _doc  # noqa: E402

FORMATS = [
    {"width": 1200, "height": 628, "name": "Landscape 1200x628"},
    {"width": 1080, "height": 1080, "name": "Square 1080"},
    {"width": 1080, "height": 1350, "name": "Portrait 4:5"},
    {"width": 300, "height": 250, "name": "MREC"},
    {"width": 728, "height": 90, "name": "Leaderboard"},
    {"width": 1080, "height": 1920, "name": "Story 9:16"},
]

MESSAGES = [
    ("Week 1", "SUMMER SALE", "Up to 50% off selected items", "SHOP NOW", "en"),
    ("Week 2", "NEW ARRIVALS", "Fresh styles for the season", "DISCOVER", "en"),
    ("Week 3", "LAST DAYS", "Final markdowns end Sunday", "HURRY", "en"),
    ("Tuần 4", "GIẢM GIÁ HÈ", "Ưu đãi đến 50% cho sản phẩm chọn lọc", "MUA NGAY", "vi"),
    ("Tuần 5", "HÀNG MỚI VỀ", "Phong cách mới cho mùa này", "KHÁM PHÁ", "vi"),
    ("Week 6", "FREE SHIPPING", "On every order over $49 this week", "ORDER TODAY", "en"),
    ("Week 7", "MEMBERS ONLY", "Early access to the autumn drop", "JOIN FREE", "en"),
    ("Woche 8", "SOMMERSCHLUSSVERKAUF", "Bis zu 50% auf ausgewählte Artikel", "JETZT KAUFEN",
     "de"),
    ("Week 9", "BUY ONE GET ONE", "Mix and match across the range", "GRAB IT", "en"),
    ("Tuần 10", "SIÊU KHUYẾN MÃI", "Chỉ trong 48 giờ", "XEM NGAY", "vi"),
    ("Week 11", "BACK IN STOCK", "The bestsellers you asked for", "SHOP THE EDIT", "en"),
    ("Week 12", "THANK YOU", "A little gift for our loyal customers", "CLAIM GIFT", "en"),
]


def build_rows(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        label, head, sub, cta, locale = MESSAGES[i % len(MESSAGES)]
        if i >= len(MESSAGES):
            label = f"{label} ({i // len(MESSAGES) + 1})"
        rows.append({
            "id": f"row{i + 1}", "label": label, "locale": locale,
            "text_overrides": {"headline": head, "sub": sub, "cta": cta},
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--formats", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--outdir", default="/tmp/autobanner_campaign")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        doc, store = _doc(tmp_path / "fixture")
        service = ProjectService(tmp_path / "data", max_workers=args.workers)
        pid = "proj_campaign"
        project = Project.create(service.projects_dir / pid, doc, name="Campaign fixture",
                                 owner=LOCAL_OWNER, brand="Acme")
        project.meta["id"] = pid
        shutil.copytree(store.root, project.assets.root, dirs_exist_ok=True)
        project.save()

        rows = build_rows(args.rows)
        formats = FORMATS[: args.formats]
        t0 = time.perf_counter()
        job = service.request_variants(pid, {"targets": formats, "rows": rows}, owner=LOCAL_OWNER)
        job = service.jobs.wait(job.id, timeout=3600)
        wall = time.perf_counter() - t0
        project = service.get_project(pid, LOCAL_OWNER)
        variants = list(project.variants.values())
        verdicts = Counter(v.verdict for v in variants)
        statuses = Counter(v.status for v in variants)
        family_issues = 0
        overflow = 0
        substituted = 0
        per_row_family: dict[str, set[str]] = {}
        not_accepted: list[dict] = []
        for v in variants:
            detail = project.variant_detail(v.id) or {}
            checks = (detail.get("quality") or {}).get("checks") or []
            if v.verdict != "accepted":
                not_accepted.append({
                    "row": (v.brief.get("row") or {}).get("label"),
                    "format": v.name, "verdict": v.verdict,
                    "issues": [
                        f"{c.get('check_id')}: {c.get('message')}" for c in checks
                        if c.get("status") in ("fail", "needs_review")
                    ][:4],
                })
            family_issues += sum(
                1 for c in checks if str(c.get("check_id", "")).startswith("family_")
                and c.get("status") != "pass"
            )
            plan = detail.get("plan") or {}
            for t in (plan.get("typography") or {}).values():
                overflow += 1 if t.get("overflow") else 0
                substituted += 1 if t.get("font_status") != "available" else 0
            fam = (plan.get("planner_meta") or {}).get("family")
            row = (v.brief.get("row") or {}).get("id")
            if fam and row:
                per_row_family.setdefault(f"{row}:{v.width}x{v.height}", set()).add(str(fam))
        service.shutdown()

    summary = {
        "rows": len(rows), "formats": len(formats), "variants": len(variants),
        "job_status": job.status, "wall_seconds": round(wall, 1),
        "seconds_per_variant": round(wall / max(1, len(variants)), 2),
        "workers": args.workers, "statuses": dict(statuses), "verdicts": dict(verdicts),
        "family_issue_checks": family_issues, "text_overflow_blocks": overflow,
        "substituted_font_blocks": substituted, "not_accepted": not_accepted,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = [
        f"# Campaign run: {summary['rows']} rows x {summary['formats']} formats",
        "",
        f"- variants: {summary['variants']} (job {summary['job_status']})",
        f"- wall time: {summary['wall_seconds']} s with {args.workers} workers "
        f"({summary['seconds_per_variant']} s per variant)",
        f"- statuses: {summary['statuses']}",
        f"- verdicts: {summary['verdicts']}",
        f"- family consistency checks not passing (within rows): {family_issues}",
        f"- text blocks overflowing at minimum size: {overflow}",
        f"- text blocks on a substituted font: {substituted}",
        "",
        "## Variants not accepted automatically",
        "",
    ] + [
        f"- {n['row']} / {n['format']}: {n['verdict']} — " + "; ".join(n["issues"])
        for n in not_accepted
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if job.status == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
