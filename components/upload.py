"""
RetinaCare AI - Upload Component
Apple-Inspired Retinal Image Ingestion & High-Precision Preview Stage
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Tuple, Any
from utils.config import SAMPLE_IMAGES_DIR, CLASS_SHORT_NAMES
from utils.html_utils import render_html
from src.preprocessing import get_display_pair, validate_image_input


def render_upload_stage() -> Tuple[Optional[Any], str]:
    """
    Renders the image source selector (Upload File or Reference Sample)
    Returns: (image_bytes_or_path, source_label)
    """
    image_source = None
    source_label = ""

    with st.container(border=True):
        render_html(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
                <div>
                    <div class="apple-caption" style="color: var(--accent-blue);">INPUT DATA SOURCE</div>
                    <div class="apple-card-title" style="margin-bottom: 0;">Retinal Fundus Image</div>
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
            label_visibility="collapsed"
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
                    help="Clinical fundus photographs categorized across the 5-point ICDR severity scale."
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
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                image_source = uploaded_file.read()
                source_label = f"Uploaded: {uploaded_file.name}"

    return image_source, source_label


def render_image_preview_stage(image_source: Any, is_processing: bool = False):
    """
    Renders the retinal photograph stage with Apple cinema frame,
    view switcher (Side-by-Side, Original, Ben Graham Enhanced),
    and active laser scan beam if processing.
    """
    try:
        orig_disp, prep_disp = get_display_pair(image_source)

        view_mode = st.radio(
            label="Inspection Mode",
            options=["Side-by-Side Comparison", "Original Capture", "Normalized Preprocessed"],
            horizontal=True,
            label_visibility="collapsed"
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
                    st.image(prep_disp, use_container_width=True, caption="Illumination Normalized (224×224)")
            elif view_mode == "Original Capture":
                st.image(orig_disp, width=380, caption="Digital Color Fundus Capture")
            else:
                st.image(prep_disp, width=380, caption="Ben Graham Normalized Input Tensor (224×224)")

    except Exception as e:
        st.error(f"Image inspection error: {str(e)}")
