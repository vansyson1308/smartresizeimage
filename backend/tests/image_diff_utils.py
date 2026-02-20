"""Utilities for golden image regression tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops


@dataclass
class DiffResult:
    size_match: bool
    mode_match: bool
    mae: float
    rmse: float
    passed: bool
    message: str
    diff_path: Path | None = None
    actual_path: Path | None = None


def compare_images(
    actual: Image.Image,
    expected: Image.Image,
    output_dir: Path,
    artifact_prefix: str,
    mae_threshold: float = 0.0,
    rmse_threshold: float = 0.0,
) -> DiffResult:
    """Compare two images and write debug artifacts when mismatch occurs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    size_match = actual.size == expected.size
    mode_match = actual.mode == expected.mode

    if not size_match or not mode_match:
        actual_path = output_dir / f"{artifact_prefix}_actual.png"
        diff_path = output_dir / f"{artifact_prefix}_diff.png"
        actual.save(actual_path)

        resized_actual = actual.convert("RGBA").resize(expected.size)
        diff_img = ImageChops.difference(resized_actual, expected.convert("RGBA"))
        diff_img.save(diff_path)

        msg = (
            f"Image metadata mismatch: size {actual.size} vs {expected.size}, "
            f"mode {actual.mode} vs {expected.mode}"
        )
        return DiffResult(
            size_match=size_match,
            mode_match=mode_match,
            mae=float("inf"),
            rmse=float("inf"),
            passed=False,
            message=msg,
            diff_path=diff_path,
            actual_path=actual_path,
        )

    actual_rgba = np.asarray(actual.convert("RGBA"), dtype=np.float32)
    expected_rgba = np.asarray(expected.convert("RGBA"), dtype=np.float32)

    delta = actual_rgba - expected_rgba
    mae = float(np.mean(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(np.square(delta))))

    passed = mae <= mae_threshold and rmse <= rmse_threshold
    diff_path = None
    actual_path = None

    if not passed:
        actual_path = output_dir / f"{artifact_prefix}_actual.png"
        diff_path = output_dir / f"{artifact_prefix}_diff.png"
        actual.save(actual_path)

        diff = np.abs(delta).clip(0, 255).astype(np.uint8)
        Image.fromarray(diff, mode="RGBA").save(diff_path)

    msg = (
        f"mae={mae:.4f} (<= {mae_threshold}), rmse={rmse:.4f} "
        f"(<= {rmse_threshold})"
    )

    return DiffResult(
        size_match=True,
        mode_match=True,
        mae=mae,
        rmse=rmse,
        passed=passed,
        message=msg,
        diff_path=diff_path,
        actual_path=actual_path,
    )
