"""Generate deterministic synthetic fixtures for Phase 2.1 benchmark."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SOURCE_SIZE = (1200, 628)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="backend/tests/bench_fixtures")
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_fixtures(outdir: Path, cases: int, seed: int) -> list[str]:
    random.seed(seed)
    np.random.seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    for idx in range(cases):
        scenario = idx % 5
        case_name = f"case_{idx + 1:02d}_{_scenario_name(scenario)}"
        names.append(case_name)

        case_dir = outdir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        img, elements, tags = _build_case(idx, scenario)
        img.save(case_dir / "input.png")
        payload = {
            "case_name": case_name,
            "seed": seed,
            "source_size": {"width": SOURCE_SIZE[0], "height": SOURCE_SIZE[1]},
            "tags": tags,
            "elements": elements,
        }
        (case_dir / "metadata.json").write_text(json.dumps(payload, indent=2))

    return names


def _scenario_name(scenario: int) -> str:
    return {
        0: "hero_headline_cta_logo",
        1: "long_text",
        2: "large_logo_long_cta",
        3: "busy_bg",
        4: "offcenter_hero",
    }[scenario]


def _build_case(idx: int, scenario: int) -> tuple[Image.Image, list[dict], dict]:
    w, h = SOURCE_SIZE
    bg = Image.new("RGBA", SOURCE_SIZE, (86, 126, 164, 255))
    draw = ImageDraw.Draw(bg)

    busy = scenario == 3
    offcenter_hero = scenario == 4

    if busy:
        rng = np.random.default_rng(1000 + idx)
        arr = rng.integers(40, 220, size=(h, w, 3), dtype=np.uint8)
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        bg = Image.fromarray(np.concatenate([arr, alpha], axis=2), mode="RGBA")
        draw = ImageDraw.Draw(bg)
    else:
        for x in range(0, w, 30):
            color = (70 + (x % 90), 110 + (x % 50), 150, 255)
            draw.line((x, 0, (x * 7) % w, h), fill=color, width=2)

    hero_x = 740 if not offcenter_hero else 930
    hero_y = 120
    hero_w, hero_h = 390, 430

    draw.ellipse((hero_x, hero_y, hero_x + hero_w, hero_y + hero_h), fill=(238, 92, 92, 255))

    logo_w, logo_h = (220, 110) if scenario == 2 else (180, 80)
    logo_x, logo_y = 960, 24
    draw.rectangle((logo_x, logo_y, logo_x + logo_w, logo_y + logo_h), fill=(245, 245, 245, 255))
    draw.text((logo_x + 16, logo_y + 30), "LOGO", fill=(22, 22, 22, 255))

    headline = "SUMMER SUPER SALE"
    sub = "Up to 50% off selected items"
    cta = "SHOP NOW"
    if scenario == 1:
        headline = "MEGA CLEARANCE WEEKEND EVENT WITH EXTRA BONUS DISCOUNTS"
        sub = "Limited-time offer on selected collections with free shipping and easy returns"
    if scenario == 2:
        cta = "JOIN MEMBERSHIP & CLAIM YOUR LIMITED REWARD TODAY"

    elements = [
        {
            "id": "bg",
            "name": "Background",
            "role": "background",
            "layer_type": "pixel",
            "bbox": {"x": 0, "y": 0, "width": w, "height": h},
            "priority": 9,
            "z_index": 0,
            "kind": "background",
        },
        {
            "id": "headline",
            "name": "Headline",
            "role": "headline",
            "layer_type": "type",
            "bbox": {"x": 70, "y": 50, "width": 620, "height": 120},
            "priority": 1,
            "z_index": 4,
            "text": headline,
            "kind": "text",
        },
        {
            "id": "sub",
            "name": "Subheadline",
            "role": "subheadline",
            "layer_type": "type",
            "bbox": {"x": 70, "y": 190, "width": 620, "height": 90},
            "priority": 2,
            "z_index": 4,
            "text": sub,
            "kind": "text",
        },
        {
            "id": "cta",
            "name": "CTA",
            "role": "cta",
            "layer_type": "type",
            "bbox": {"x": 70, "y": 305, "width": 350, "height": 95},
            "priority": 2,
            "z_index": 4,
            "text": cta,
            "kind": "text",
        },
        {
            "id": "logo",
            "name": "Logo",
            "role": "logo",
            "layer_type": "pixel",
            "bbox": {"x": logo_x, "y": logo_y, "width": logo_w, "height": logo_h},
            "priority": 1,
            "z_index": 5,
            "kind": "logo",
        },
        {
            "id": "hero",
            "name": "Hero",
            "role": "hero_image",
            "layer_type": "pixel",
            "bbox": {"x": hero_x, "y": hero_y, "width": hero_w, "height": hero_h},
            "priority": 2,
            "z_index": 3,
            "kind": "hero",
        },
    ]

    tags = {
        "busy_background": busy,
        "offcenter_hero": offcenter_hero,
        "scenario": _scenario_name(scenario),
    }
    return bg, elements, tags


def main() -> None:
    args = parse_args()
    names = generate_fixtures(Path(args.outdir), cases=args.cases, seed=args.seed)
    print(f"Generated {len(names)} fixtures in {args.outdir}")


if __name__ == "__main__":
    main()
