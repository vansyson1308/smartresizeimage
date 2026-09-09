"""Gradio web interface for AutoBanner ReLayout Pro."""

from __future__ import annotations

import atexit
import contextlib
import io
import json
import logging
import os
import sys
import tempfile
import traceback
import zipfile

from .enums import ElementRole
from .exceptions import AutoBannerError
from .logging_config import setup_logging
from .relayout import ReLayoutEngine

logger = logging.getLogger("autobanner.main")

# Optional Gradio import
try:
    import gradio as gr

    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    logger.error("Gradio not installed. Run: pip install gradio")

# Temp file management
_temp_files: list[str] = []


def _cleanup_temp_files() -> None:
    """Clean up temporary files."""
    for f in _temp_files:
        with contextlib.suppress(OSError):
            os.unlink(f)
    _temp_files.clear()


atexit.register(_cleanup_temp_files)


# Standard size presets
SIZE_PRESETS = {
    "Instagram Story (1080x1920)": (1080, 1920),
    "Instagram Square (1080x1080)": (1080, 1080),
    "Instagram Portrait (1080x1350)": (1080, 1350),
    "Facebook Cover (1200x630)": (1200, 630),
    "Facebook Post (1200x1200)": (1200, 1200),
    "LinkedIn Post (1200x627)": (1200, 627),
    "YouTube Thumbnail (1280x720)": (1280, 720),
    "Pinterest Pin (1000x1500)": (1000, 1500),
    "Twitter Header (1500x500)": (1500, 500),
    "Billboard (2000x1000)": (2000, 1000),
}


def _use_ai_from_env() -> bool:
    return os.environ.get("AUTOBANNER_USE_AI", "true").strip().lower() == "true"


def _new_session_state() -> dict:
    """Per-browser-session state. Each session owns its own engine and temp files."""
    return {"engine": None, "zip_path": None}


def _remove_file(path: str | None) -> None:
    if path:
        with contextlib.suppress(OSError):
            os.unlink(path)


def _verdict_summary(results: dict) -> str:
    """Render a concise, designer-facing summary of every variant's verdict."""
    if not results:
        return "**Status:** No variants generated"
    labels = {
        "accepted": "Accepted",
        "needs_review": "Needs review",
        "failed": "Failed",
        "not_evaluated": "Not checked",
    }
    lines = []
    for name, result in results.items():
        verdict = result.verdict
        line = f"- **{name}** — {labels.get(verdict, verdict)}"
        if result.quality is not None:
            issues = result.quality.issues()[:3]
            if issues:
                line += ": " + "; ".join(i.message for i in issues)
        lines.append(line)
    counts = {}
    for result in results.values():
        counts[result.verdict] = counts.get(result.verdict, 0) + 1
    header = ", ".join(f"{labels.get(k, k)}: {v}" for k, v in counts.items())
    return f"**Status:** {header}\n" + "\n".join(lines)


def create_interface() -> object:
    """Create Gradio interface for ReLayout Pro.

    The engine is created per browser session (``gr.State``) so concurrent
    users never share parsed elements, role edits or temp files.
    """
    use_ai = _use_ai_from_env()

    with gr.Blocks(
        title="ReLayout Pro - Adaptive Re-Composition",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 98% !important; }
        .gr-button { font-weight: bold !important; }
        .gr-button.primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        .element-table { font-size: 12px; }
        """,
    ) as interface:
        session = gr.State(_new_session_state)

        gr.Markdown(
            """
        # ReLayout Pro - Adaptive Re-Composition Engine

        **Transform your designs to any aspect ratio while preserving layout integrity.**

        ### How it works:
        1. **Upload** your design file (PSD, PNG, JPG, WEBP)
        2. **Review** detected elements and their roles (correct if needed)
        3. **Select** target sizes
        4. **Generate** re-composed versions and read each variant's verdict

        ---
        """
        )

        with gr.Row():
            # Left column - Input & Analysis
            with gr.Column(scale=1):
                gr.Markdown("### Step 1: Upload Design File")

                file_input = gr.File(
                    label="Upload Design File",
                    file_types=[".psd", ".png", ".jpg", ".jpeg", ".webp"],
                    type="filepath",
                )

                analyze_btn = gr.Button("Analyze File", variant="primary")

                gr.Markdown("### Step 2: Review Elements")

                analysis_json = gr.JSON(label="Detected Elements", visible=True)

                preview_image = gr.Image(
                    label="Preview with Bounding Boxes",
                    type="pil",
                    height=300,
                )

                gr.Markdown("### Edit Roles (Optional)")

                with gr.Row():
                    edit_element_id = gr.Textbox(
                        label="Element ID",
                        placeholder="e.g., headline_abc123",
                    )
                    edit_role = gr.Dropdown(
                        choices=[r.value for r in ElementRole],
                        label="New Role",
                    )
                    edit_btn = gr.Button("Update", size="sm")

            # Right column - Output
            with gr.Column(scale=1):
                gr.Markdown("### Step 3: Select Target Sizes")

                size_presets = gr.CheckboxGroup(
                    choices=list(SIZE_PRESETS.keys()),
                    value=[
                        "Instagram Story (1080x1920)",
                        "Instagram Square (1080x1080)",
                    ],
                    label="Preset Sizes",
                )

                with gr.Row():
                    custom_width = gr.Number(value=1200, label="Custom Width")
                    custom_height = gr.Number(value=628, label="Custom Height")
                    custom_name = gr.Textbox(value="Custom", label="Name")

                generation_mode = gr.Radio(
                    choices=["phase21", "phase3"],
                    value="phase21",
                    label="Generation Mode",
                    info="phase21=deterministic relayout, phase3=target-first redesign",
                )
                anchor_preset = gr.Dropdown(
                    choices=["none", "flat_banner_3anchors"],
                    value="none",
                    label="Anchor preset",
                )
                manual_anchors_json = gr.Textbox(
                    label="Manual anchors JSON (for flat images, optional)",
                    placeholder='[{"id":"logo","role":"logo","x":10,"y":10,"width":120,"height":60}]',
                    lines=3,
                )

                generate_btn = gr.Button(
                    "GENERATE ALL LAYOUTS",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("### Step 4: Results")

                gallery = gr.Gallery(
                    label="Generated Layouts",
                    columns=2,
                    height=400,
                    object_fit="contain",
                )

                download_zip = gr.File(label="Download All (ZIP)")

                status = gr.Markdown("**Status:** Ready")

        # Event handlers
        def analyze_file(uploaded_file: str, state: dict) -> tuple:
            state = dict(state or _new_session_state())
            if uploaded_file is None:
                return None, None, "Please upload a design file", state

            try:
                engine = ReLayoutEngine(use_ai=use_ai)
                analysis = engine.load_file(uploaded_file)
                preview = engine.get_preview_image()
                state["engine"] = engine
                return (
                    analysis,
                    preview,
                    f"Loaded {analysis['total_layers']} layers from {analysis['file']}",
                    state,
                )
            except AutoBannerError as e:
                return None, None, f"Error: {str(e)}", state
            except Exception as e:
                traceback.print_exc()
                return None, None, f"Unexpected error: {str(e)}", state

        analyze_btn.click(
            analyze_file,
            inputs=[file_input, session],
            outputs=[analysis_json, preview_image, status, session],
        )

        def update_role(elem_id: str, role: str, state: dict) -> str:
            engine = (state or {}).get("engine")
            if engine is None:
                return "Please load a design file first"
            if elem_id and role:
                success = engine.update_element_role(elem_id, role)
                if success:
                    return f"Updated {elem_id} to {role}"
                return f"Could not update {elem_id}"
            return "Please enter element ID and select role"

        edit_btn.click(
            update_role,
            inputs=[edit_element_id, edit_role, session],
            outputs=[status],
        )

        def generate_layouts(
            presets: list,
            cw: float,
            ch: float,
            cname: str,
            mode: str,
            anchor_preset_selected: str,
            manual_json: str,
            state: dict,
        ) -> tuple:
            state = dict(state or _new_session_state())
            engine = state.get("engine")
            if engine is None or not engine.elements:
                return [], None, "Please load a design file first", state

            try:
                # Only this session's previous ZIP is removed.
                _remove_file(state.get("zip_path"))
                state["zip_path"] = None

                # Build target sizes list
                targets = []

                for size_preset in presets:
                    if size_preset in SIZE_PRESETS:
                        w, h = SIZE_PRESETS[size_preset]
                        name = size_preset.split("(")[0].strip()
                        targets.append((w, h, name))

                # Add custom size
                if cw and ch and cw > 0 and ch > 0:
                    targets.append((int(cw), int(ch), cname or "Custom"))

                if not targets:
                    return [], None, "Please select at least one size", state

                manual_anchors = None
                if anchor_preset_selected == "flat_banner_3anchors" and (
                    not manual_json or not manual_json.strip()
                ):
                    sw, sh = engine.source_size
                    manual_anchors = [
                        {
                            "id": "Mascot",
                            "role": "hero_image",
                            "x": int(0.56 * sw),
                            "y": int(0.22 * sh),
                            "width": int(0.32 * sw),
                            "height": int(0.66 * sh),
                        },
                        {
                            "id": "MainText",
                            "role": "headline",
                            "x": int(0.06 * sw),
                            "y": int(0.10 * sh),
                            "width": int(0.50 * sw),
                            "height": int(0.34 * sh),
                        },
                        {
                            "id": "CTA",
                            "role": "cta",
                            "x": int(0.08 * sw),
                            "y": int(0.64 * sh),
                            "width": int(0.28 * sw),
                            "height": int(0.18 * sh),
                        },
                    ]
                elif manual_json and manual_json.strip():
                    manual_anchors = json.loads(manual_json)
                    if not isinstance(manual_anchors, list):
                        return [], None, "manual anchors JSON must be a list", state

                # Generate
                results = engine.batch_relayout(
                    targets,
                    mode=mode or "phase21",
                    manual_anchors=manual_anchors,
                )

                # Prepare gallery
                gallery_items = []
                for name, result in results.items():
                    gallery_items.append((result.image, f"{name} — {result.verdict}"))

                # Create ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name, result in results.items():
                        img_buffer = io.BytesIO()
                        result.image.save(img_buffer, format="PNG", optimize=True)
                        filename = f"{name.replace(' ', '_')}.png"
                        zf.writestr(filename, img_buffer.getvalue())
                        if result.quality is not None:
                            zf.writestr(
                                f"{name.replace(' ', '_')}.quality.json",
                                json.dumps(result.quality.to_dict(), indent=2),
                            )

                # Save ZIP to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".zip"
                ) as tmp:
                    tmp.write(zip_buffer.getvalue())
                    zip_path = tmp.name
                    state["zip_path"] = zip_path
                    _temp_files.append(zip_path)

                return gallery_items, zip_path, _verdict_summary(results), state

            except AutoBannerError as e:
                return [], None, f"Error: {str(e)}", state
            except Exception as e:
                traceback.print_exc()
                return [], None, f"Unexpected error: {str(e)}", state

        generate_btn.click(
            generate_layouts,
            inputs=[
                size_presets,
                custom_width,
                custom_height,
                custom_name,
                generation_mode,
                anchor_preset,
                manual_anchors_json,
                session,
            ],
            outputs=[gallery, download_zip, status, session],
        )

    return interface


def main() -> None:
    """Main entry point."""
    setup_logging(os.environ.get("AUTOBANNER_LOG_LEVEL", "INFO"))

    logger.info("=" * 70)
    logger.info("RELAYOUT PRO - Adaptive Re-Composition Engine")
    logger.info("=" * 70)

    if not HAS_GRADIO:
        logger.error("Gradio is required. Run: pip install gradio")
        sys.exit(1)

    logger.info("Starting web interface...")

    server_name = os.environ.get(
        "AUTOBANNER_SERVER_NAME", os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    )
    server_port = int(
        os.environ.get("AUTOBANNER_SERVER_PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))
    )
    share = os.environ.get("AUTOBANNER_SHARE", "false").lower() == "true"
    inbrowser = os.environ.get("AUTOBANNER_INBROWSER", "false").lower() == "true"

    try:
        interface = create_interface()
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            inbrowser=inbrowser,
            show_error=True,
        )
    except ValueError as e:
        msg = str(e)
        if "localhost is not accessible" in msg and not share:
            logger.warning(
                "Launch blocked by localhost accessibility check. "
                "Set AUTOBANNER_SHARE=true (or adjust proxy/NO_PROXY) in this environment."
            )
            return
        logger.error("Failed to start: %s", e)
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to start: %s", e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
