"""Golden visual regression tests for rendering paths (text-only fixtures)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from backend.app.composition.background import BackgroundExtender
from backend.app.composition.color import BlendMode, composite_pil_over
from backend.app.composition.engine import CompositionEngine
from backend.app.composition.resize import high_quality_resize
from backend.app.enums import ElementRole
from backend.app.models import BoundingBox, DesignElement, LayoutResult
from backend.tests.image_diff_utils import compare_images

OUTPUTS = Path("backend/tests/fixtures/outputs")


GOLDEN_HASHES = {
    "case_a_alpha": "b2b8a87af24de348fc251668c79a61bec5f218c4c6c676cea4970bcacc9d273a",
    "case_b_resize": "05fb17eaef80a6d2d47bbe3e9a5ec0827e9264b305d497cec1abeab6391c85af",
    "case_c_fallback": "e2d2f25359fcb1dc6c526bc734023e976a5814034ddb6c0aab5e3dcfb889aacd",
    "case_d_linear": "0cffde20cf8482c3c99a7eb581c3de78ac3357cb58f479a09d8f784cc1131e4b",
    "case_e_shadow_slight": "e9506814ac5647e6822db99ef5db36d4e26b59547117c26d2dae390b7f5d53a8",
    "case_f_shadow_soft": "ed8c046328663442fe18653028a68b7de051e1c526afd1faf82e512d0129f68e",
    "case_g_shadow_offset": "5882948714dbc8ebbcee6113f18a4bae4789678fb04962f6bb14dd0f027e29c2",
}


def _image_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()


def _gradient_image(size: tuple[int, int], alpha: int = 255) -> Image.Image:
    w, h = size
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 0] = (40 + 180 * xx).astype(np.uint8)
    arr[:, :, 1] = (30 + 170 * yy).astype(np.uint8)
    arr[:, :, 2] = (200 - 120 * xx).astype(np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def _foreground_shape(size: tuple[int, int]) -> Image.Image:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((6, 6, size[0] - 8, size[1] - 10), radius=10, fill=(245, 215, 50, 210))
    draw.ellipse((size[0] // 3, 8, size[0] - 8, size[1] - 14), fill=(220, 90, 80, 160))
    return img


def _render_case_a() -> Image.Image:
    bg = _gradient_image((120, 120), alpha=255)
    fg = _foreground_shape((80, 80))

    engine = CompositionEngine(use_ai_inpainting=False)
    bg_elem = DesignElement(
        id="bg",
        name="bg",
        layer_type="pixel",
        bbox=BoundingBox(0, 0, 120, 120),
        image=bg,
        role=ElementRole.BACKGROUND,
        priority=9,
        z_index=0,
    )
    fg_elem = DesignElement(
        id="fg",
        name="fg",
        layer_type="pixel",
        bbox=BoundingBox(0, 0, 80, 80),
        image=fg,
        role=ElementRole.HERO_IMAGE,
        priority=3,
        z_index=1,
    )
    layout = [
        LayoutResult("bg", BoundingBox(0, 0, 120, 120), 1.0),
        LayoutResult("fg", BoundingBox(20, 20, 80, 80), 1.0),
    ]
    return engine.compose([bg_elem, fg_elem], layout, (120, 120), (120, 120)).image


def _render_case_b() -> Image.Image:
    source = _gradient_image((150, 90), alpha=255)
    return high_quality_resize(source, (64, 64))


def _render_case_c(monkeypatch) -> Image.Image:
    source = _gradient_image((96, 64), alpha=255)

    from backend.app.composition import background as bg_mod

    monkeypatch.setattr(bg_mod, "_cv2_checked", True)
    monkeypatch.setattr(bg_mod, "_cv2_module", None)

    extender = BackgroundExtender(use_ai_inpainting=False)
    return extender.extend(source, (140, 140))


def _render_case_d() -> Image.Image:
    bg = _gradient_image((80, 80), alpha=255)
    fg = _foreground_shape((48, 48))
    return composite_pil_over(bg, fg, (16, 16), use_linear=True).convert("RGB")


def _render_shadow_case(case_name: str) -> Image.Image:
    bg = _gradient_image((140, 120), alpha=255)
    fg = _foreground_shape((70, 50))

    shadow_params = {
        "case_e_shadow_slight": {
            "offset_x": 3,
            "offset_y": 3,
            "blur_radius": 2,
            "opacity": 0.45,
            "color": (0, 0, 0, 255),
        },
        "case_f_shadow_soft": {
            "offset_x": 2,
            "offset_y": 4,
            "blur_radius": 10,
            "opacity": 0.35,
            "color": (0, 0, 0, 255),
        },
        "case_g_shadow_offset": {
            "offset_x": 14,
            "offset_y": 10,
            "blur_radius": 4,
            "opacity": 0.55,
            "color": (20, 20, 20, 255),
        },
    }[case_name]

    engine = CompositionEngine(use_ai_inpainting=False)
    bg_elem = DesignElement(
        id="bg",
        name="bg",
        layer_type="pixel",
        bbox=BoundingBox(0, 0, 140, 120),
        image=bg,
        role=ElementRole.BACKGROUND,
        priority=9,
    )
    fg_elem = DesignElement(
        id="fg",
        name="fg",
        layer_type="pixel",
        bbox=BoundingBox(0, 0, 70, 50),
        image=fg,
        role=ElementRole.HERO_IMAGE,
        priority=3,
        z_index=1,
        effects={"drop_shadow": shadow_params},
    )
    return engine.compose(
        [bg_elem, fg_elem],
        [
            LayoutResult("bg", BoundingBox(0, 0, 140, 120), 1.0),
            LayoutResult("fg", BoundingBox(35, 35, 70, 50), 1.0),
        ],
        (140, 120),
        (140, 120),
    ).image


def test_golden_alpha_compositing_case_a() -> None:
    assert _image_sha256(_render_case_a()) == GOLDEN_HASHES["case_a_alpha"]


def test_golden_gamma_resize_case_b() -> None:
    assert _image_sha256(_render_case_b()) == GOLDEN_HASHES["case_b_resize"]


def test_golden_fallback_extension_case_c(monkeypatch) -> None:
    assert _image_sha256(_render_case_c(monkeypatch)) == GOLDEN_HASHES["case_c_fallback"]


def test_diff_harness_creates_artifact_on_mismatch() -> None:
    expected = _render_case_b()
    wrong = Image.new("RGBA", expected.size, (0, 0, 0, 255))

    result = compare_images(
        wrong,
        expected,
        OUTPUTS,
        artifact_prefix="proof_mismatch_artifact",
        mae_threshold=0.0,
        rmse_threshold=0.0,
    )

    assert not result.passed
    assert result.diff_path is not None and result.diff_path.exists()
    assert result.actual_path is not None and result.actual_path.exists()


def test_golden_linear_compositing_case_d() -> None:
    assert _image_sha256(_render_case_d()) == GOLDEN_HASHES["case_d_linear"]


def test_golden_drop_shadow_slight_case_e() -> None:
    actual = _image_sha256(_render_shadow_case("case_e_shadow_slight"))
    assert actual == GOLDEN_HASHES["case_e_shadow_slight"]


def test_golden_drop_shadow_soft_case_f() -> None:
    actual = _image_sha256(_render_shadow_case("case_f_shadow_soft"))
    assert actual == GOLDEN_HASHES["case_f_shadow_soft"]


def test_golden_drop_shadow_shifted_case_g() -> None:
    actual = _image_sha256(_render_shadow_case("case_g_shadow_offset"))
    assert actual == GOLDEN_HASHES["case_g_shadow_offset"]


def test_golden_blend_modes_mvp_hashes() -> None:
    bg = Image.new("RGBA", (32, 32), (20, 40, 180, 255))
    fg = Image.new("RGBA", (32, 32), (240, 220, 30, 160))

    mode_hashes = {
        BlendMode.NORMAL: "7ce45a82a7b31477d4c6d554143e44bbe34331f7b344ad87f4e5924a9eff3434",
        BlendMode.MULTIPLY: "7f0dd982a33c902aa3e3cf1bc282eefdeb00d40d40d59c73a88b570eccbcba2d",
        BlendMode.SCREEN: "8b24c072f1a37eb0d527790bf8a90bf0d1c10d5d4f7e5f9cddf0b34cabed8e94",
        BlendMode.OVERLAY: "1ee9e3adc907814fdb15f9a96efc513aaa7dcc238cb908f1290b1b134d613e10",
    }

    for mode, digest in mode_hashes.items():
        actual = composite_pil_over(bg, fg, (0, 0), use_linear=True, blend_mode=mode)
        assert _image_sha256(actual) == digest
