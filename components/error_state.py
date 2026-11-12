"""
==============================================================================
RetinaCare AI - Context-Aware Error & Generative Guidance Component
Apple-Inspired Minimalist Alerts & Actionable Resolution Modals
==============================================================================
"""

from __future__ import annotations
import streamlit as st
from typing import Optional, Dict, Any
from utils.html_utils import render_html


def render_context_aware_error(
    error_type: str,
    title: str,
    message: str,
    details: Optional[str] = None,
    action_button_label: Optional[str] = None,
    action_callback: Optional[Any] = None
):
    """
    Renders an Apple-styled context-aware error banner with guidance.
    
    Error Types:
        - 'invalid_image': Corrupted, undersized, or unreadable image file.
        - 'wrong_image_type': Heuristic retinal plausibility warning.
        - 'missing_model': Selected model checkpoint does not exist on disk.
        - 'model_compatibility': Checkpoint exists but fails weight/shape verification.
        - 'class_mapping': Dataset/class order cannot be verified.
        - 'low_confidence': Prediction probability is below confidence threshold.
        - 'generic': General execution failure.
    """
    icons = {
        "invalid_image": "🖼️",
        "wrong_image_type": "👁️",
        "missing_model": "📦",
        "model_compatibility": "⚙️",
        "class_mapping": "📋",
        "low_confidence": "⚠️",
        "generic": "⚠️"
    }
    icon = icons.get(error_type, "⚠️")

    # Error accent styling
    border_color = "rgba(255, 69, 58, 0.4)"
    bg_color = "rgba(255, 69, 58, 0.06)"
    text_accent = "var(--accent-red)"

    if error_type in ("wrong_image_type", "low_confidence"):
        border_color = "rgba(255, 159, 10, 0.4)"
        bg_color = "rgba(255, 159, 10, 0.06)"
        text_accent = "var(--accent-amber)"

    error_html = f"""
    <div class="apple-card" style="border-color: {border_color}; background: {bg_color}; margin-top: 20px; padding: 22px 24px;">
        <div style="display: flex; align-items: flex-start; gap: 14px;">
            <span style="font-size: 1.6rem; line-height: 1;">{icon}</span>
            <div style="flex: 1;">
                <div style="font-size: 1.05rem; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">
                    {title}
                </div>
                <div style="font-size: 0.90rem; color: var(--text-secondary); line-height: 1.5; margin-bottom: 8px;">
                    {message}
                </div>
                {f'''
                <div style="font-size: 0.80rem; color: var(--text-tertiary); font-family: monospace; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: var(--radius-xs); margin-top: 8px;">
                    {details}
                </div>
                ''' if details else ''}
            </div>
        </div>
    </div>
    """
    render_html(error_html)

    # Optional Action Button (e.g. "Try Again" or "Train / Add Model")
    if action_button_label:
        col_l, col_btn, col_r = st.columns([1.2, 1.6, 1.2])
        with col_btn:
            if st.button(action_button_label, type="secondary", use_container_width=True, key=f"btn_err_{error_type}"):
                if action_callback:
                    action_callback()
                st.rerun()


def render_missing_model_error(model_name: str, training_command: str):
    """
    Specifically renders Section 19 requirement:
    'Selected model is currently unavailable.'
    Model: EfficientNetB0
    Action: 'Train / Add Model'
    """
    render_context_aware_error(
        error_type="missing_model",
        title="Selected model is currently unavailable.",
        message=(
            f"The deep learning checkpoint for <strong>{model_name}</strong> has not been found on disk. "
            f"Please choose another trained model or train this architecture before using it for prediction."
        ),
        details=f"Train this model via terminal:\n{training_command}",
        action_button_label="Choose Another Model"
    )


def render_invalid_image_error(user_msg: Optional[str] = None):
    """
    Specifically renders Section 16 & 18 requirements:
    'Unable to analyze this image. Please upload a valid retinal fundus image in JPG, JPEG or PNG format.'
    Button: 'Try Again'
    """
    msg = user_msg or (
        "That image doesn't appear to be usable for analysis. "
        "Try uploading a clear retinal fundus image in JPG, JPEG or PNG format."
    )
    render_context_aware_error(
        error_type="invalid_image",
        title="Unable to analyze this image.",
        message=msg,
        action_button_label="Try Again",
        action_callback=lambda: st.session_state.update({"prediction_result": None})
    )
