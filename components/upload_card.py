"""
==============================================================================
RetinaCare AI - Retinal Image Ingestion & Stage Card
Department of Computer Science & Engineering - GM University
==============================================================================
"""

from __future__ import annotations
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple, Any
from utils.config import SAMPLE_IMAGES_DIR
from utils.html_utils import render_html
from src.preprocessing import get_display_pair
from src.validation import inspect_and_decode_image, check_retinal_fundus_plausibility


def render_upload_card() -> Tuple[Optional[Any], str]:
    """
    Renders the image source selector card with options for verified
    clinical reference samples or custom fundus image uploads.
    
    Returns:
        (image_bytes_or_path, source_label)
    """
    image_source = None
    source_label = ""

    with st.container(border=True):
        render_html(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
                <div>
                    <div class="apple-caption" style="color: var(--accent-blue);">RETINAL IMAGING</div>
                    <div class="apple-card-title" style="margin-bottom: 0;">Upload Retinal Image</div>
                </div>
                <div style="font-size: 0.8rem; color: var(--text-tertiary);">
                    Standard 224×224 RGB
                </div>
            </div>
            """
        )

        source_tab = st.radio(
            label="Input Method",
            options=["Reference Clinical Sample", "Upload Custom Image"],
            horizontal=True,
            label_visibility="collapsed",
            key="input_method_radio"
        )

        if source_tab == "Reference Clinical Sample":
            sample_files = sorted(list(SAMPLE_IMAGES_DIR.glob("*.png")))
            if sample_files:
                sample_options = {
                    f"Stage {f.name.split('_')[1]} — {f.stem.replace('sample_', '').replace('_', ' ').title()}": f
                    for f in sample_files
                }
                selected_name = st.selectbox(
                    label="Choose Reference Retinal Photo",
                    options=list(sample_options.keys()),
                    help="Clinical fundus photographs categorized across the 5-point ICDR severity scale.",
                    key="sample_image_select"
                )
                image_source = sample_options[selected_name]
                source_label = selected_name
            else:
                st.warning("No sample files found in assets directory.")
        else:
            uploaded_file = st.file_uploader(
                label="Upload Retinal Fundus Photo",
                type=["png", "jpg", "jpeg", "bmp"],
                help="High-resolution digital color fundus photography (PNG, JPG, BMP).",
                label_visibility="collapsed",
                key="retinal_file_uploader"
            )
            if uploaded_file is not None:
                image_source = uploaded_file.read()
                source_label = f"Uploaded: {uploaded_file.name}"

    return image_source, source_label


def render_preview_card(image_source: Any, is_processing: bool = False):
    """
    Renders the image inspection card with view modes (Side-by-Side, Original, Normalized),
    retinal plausibility heuristic banner, and active laser beam scanning animation.
    """
    try:
        # Validate and extract preview displays
        rgb = inspect_and_decode_image(image_source)
        orig_disp, prep_disp = get_display_pair(rgb)
        is_plausible, warning_msg = check_retinal_fundus_plausibility(rgb)

        view_mode = st.radio(
            label="Inspection Mode",
            options=["Side-by-Side Comparison", "Original Capture", "Normalized Preprocessed"],
            horizontal=True,
            label_visibility="collapsed",
            key="inspection_mode_radio"
        )

        render_html("<div style='height: 8px;'></div>")

        with st.container(border=True):
            if is_processing:
                render_html(
                    """
                    <div class="retinal-scan-overlay">
                        <div class="retinal-scan-line"></div>
                    </div>
                    """
                )

            if view_mode == "Side-by-Side Comparison":
                col1, col2 = st.columns(2, gap="medium")
                with col1:
                    st.image(orig_disp, use_container_width=True, caption="Digital Color Fundus (Original)")
                with col2:
                    st.image(prep_disp, use_container_width=True, caption="Ben Graham Normalized (224×224)")
            elif view_mode == "Original Capture":
                st.image(orig_disp, width=380, caption="Digital Color Fundus Capture")
            else:
                st.image(prep_disp, width=380, caption="Normalized Input Tensor (224×224)")

        # Gentle anatomical plausibility warning if heuristic triggers
        if not is_plausible and warning_msg:
            render_html(
                f"""
                <div style="margin-top: 10px; padding: 10px 14px; background: rgba(255, 159, 10, 0.08); border-left: 3px solid var(--accent-amber); border-radius: var(--radius-xs); font-size: 0.80rem; color: var(--text-secondary);">
                    <span style="font-weight: 600; color: var(--accent-amber);">Plausibility Notice:</span> {warning_msg}
                </div>
                """
            )

    except Exception as exc:
        st.error(f"Image inspection error: {str(exc)}")
