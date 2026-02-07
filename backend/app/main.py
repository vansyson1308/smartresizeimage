"""Gradio web interface for AutoBanner ReLayout Pro."""

from __future__ import annotations

import atexit
import contextlib
import io
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


def create_interface() -> object:
    """Create Gradio interface for ReLayout Pro."""
    engine = ReLayoutEngine(use_ai=True)

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
        gr.Markdown(
            """
        # ReLayout Pro - Adaptive Re-Composition Engine

        **Transform your designs to any aspect ratio while preserving layout integrity.**

        ### How it works:
        1. **Upload** your design file (PSD, PNG, JPG, WEBP)
        2. **Review** detected elements and their roles (correct if needed)
        3. **Select** target sizes
        4. **Generate** re-composed versions

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
        def analyze_file(uploaded_file: str) -> tuple:
            if uploaded_file is None:
                return None, None, "Please upload a design file"

            try:
                analysis = engine.load_file(uploaded_file)
                preview = engine.get_preview_image()
                return (
                    analysis,
                    preview,
                    f"Loaded {analysis['total_layers']} layers from {analysis['file']}",
                )
            except AutoBannerError as e:
                return None, None, f"Error: {str(e)}"
            except Exception as e:
                traceback.print_exc()
                return None, None, f"Unexpected error: {str(e)}"

        analyze_btn.click(
            analyze_file,
            inputs=[file_input],
            outputs=[analysis_json, preview_image, status],
        )

        def update_role(elem_id: str, role: str) -> str:
            if elem_id and role:
                success = engine.update_element_role(elem_id, role)
                if success:
                    return f"Updated {elem_id} to {role}"
                return f"Could not update {elem_id}"
            return "Please enter element ID and select role"

        edit_btn.click(
            update_role,
            inputs=[edit_element_id, edit_role],
            outputs=[status],
        )

        def generate_layouts(
            presets: list, cw: float, ch: float, cname: str
        ) -> tuple:
            if not engine.elements:
                return [], None, "Please load a design file first"

            try:
                # Clean previous temp files
                _cleanup_temp_files()

                # Build target sizes list
                targets = []

                for preset in presets:
                    if preset in SIZE_PRESETS:
                        w, h = SIZE_PRESETS[preset]
                        name = preset.split("(")[0].strip()
                        targets.append((w, h, name))

                # Add custom size
                if cw and ch and cw > 0 and ch > 0:
                    targets.append((int(cw), int(ch), cname or "Custom"))

                if not targets:
                    return [], None, "Please select at least one size"

                # Generate
                results = engine.batch_relayout(targets)

                # Prepare gallery
                gallery_items = []
                for name, result in results.items():
                    gallery_items.append((result.image, name))

                # Create ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name, result in results.items():
                        img_buffer = io.BytesIO()
                        result.image.save(img_buffer, format="PNG", optimize=True)
                        filename = f"{name.replace(' ', '_')}.png"
                        zf.writestr(filename, img_buffer.getvalue())

                # Save ZIP to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".zip"
                ) as tmp:
                    tmp.write(zip_buffer.getvalue())
                    zip_path = tmp.name
                    _temp_files.append(zip_path)

                return gallery_items, zip_path, f"Generated {len(results)} layouts!"

            except AutoBannerError as e:
                return [], None, f"Error: {str(e)}"
            except Exception as e:
                traceback.print_exc()
                return [], None, f"Unexpected error: {str(e)}"

        generate_btn.click(
            generate_layouts,
            inputs=[size_presets, custom_width, custom_height, custom_name],
            outputs=[gallery, download_zip, status],
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

    try:
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True,
        )
    except Exception as e:
        logger.error("Failed to start: %s", e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
