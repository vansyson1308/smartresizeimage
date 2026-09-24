"""Rendering must be reproducible across processes (PYTHONHASHSEED varies)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
import hashlib, sys
from backend.app.relayout import ReLayoutEngine
from backend.app.service import flat_banner_anchor_preset
e = ReLayoutEngine(use_ai=False)
e.load_file(sys.argv[1])
r3 = e.relayout_redesign((240, 240), manual_anchors=flat_banner_anchor_preset(e.source_size))
r2 = e.relayout((240, 400))
print(hashlib.sha256(r3.image.tobytes() + r2.image.tobytes()).hexdigest())
"""


def _run(path: str, hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, path],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True, timeout=240,
    )
    return out.stdout.strip().splitlines()[-1]


def test_output_is_identical_across_hash_seeds(tmp_path):
    img = Image.new("RGB", (400, 210), (60, 120, 200))
    d = ImageDraw.Draw(img)
    d.rectangle((20, 20, 180, 80), fill=(255, 255, 255))
    d.ellipse((230, 40, 380, 200), fill=(240, 90, 80))
    path = tmp_path / "b.png"
    img.save(path)
    assert _run(str(path), "1") == _run(str(path), "12345")
