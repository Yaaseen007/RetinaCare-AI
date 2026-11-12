"""
==============================================================================
RetinaCare AI - Diagnostic Result Card Component
Apple-Inspired Micro-Interactions, Model Provenance, and Clinical Reveal
==============================================================================
"""

from __future__ import annotations
import streamlit as st
from typing import Dict, Any
from utils.config import CLASS_NAMES, CLASS_SHORT_NAMES
from utils.html_utils import render_html
from components.probability_chart import render_probability_bars


def render_prediction_result_card(pred_data: Dict[str, Any]):
    """
    Renders the Apple-inspired primary result card displaying the highest-confidence
    prediction, model architecture, softmax confidence, low-confidence warning if below threshold,
    animated probability distribution, biomarker interpretations, and educational screening disclaimer.
    """
    if hasattr(pred_data, "model_dump"):
        pred_data = pred_data.model_dump()
    elif hasattr(pred_data, "dict") and callable(pred_data.dict):
        pred_data = pred_data.dict()
    elif not isinstance(pred_data, dict):
        try:
            pred_data = dict(pred_data)
        except Exception:
            pred_data = {}

    model_name = pred_data.get("model_name") or "AI Model"
    dataset_name = pred_data.get("dataset_name") or "APTOS 2019"

    # Safely resolve integer class_id
    raw_cid = pred_data.get("class_id")
    if raw_cid is None:
        raw_cid = pred_data.get("predicted_class_index", 0)
    try:
        class_id = int(raw_cid) if raw_cid is not None else 0
    except (ValueError, TypeError):
        class_id = 0

    # Safely resolve class_name with multiple clinical fallbacks
    class_name = (
        pred_data.get("class_name")
        or pred_data.get("predicted_class_name")
        or pred_data.get("prediction")
    )
    if not class_name:
        if 0 <= class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = f"Stage {class_id}"

    # Safely resolve short_name
    short_name = pred_data.get("short_name")
    if not short_name:
        if 0 <= class_id < len(CLASS_SHORT_NAMES):
            short_name = CLASS_SHORT_NAMES[class_id]
        else:
            short_name = f"Stage {class_id}"

    raw_conf = pred_data.get("confidence_pct")
    try:
        confidence_pct = float(raw_conf) if raw_conf is not None else 0.0
    except (ValueError, TypeError):
        confidence_pct = 0.0

    is_low_conf = bool(pred_data.get("is_low_confidence", False))
    conf_status = pred_data.get("confidence_status") or "Model confidence is derived from vascular feature representations."
    description = pred_data.get("description") or ""
    recommendation = pred_data.get("recommendation") or ""
    prob_dict = pred_data.get("probabilities") or {}
    architecture_type = pred_data.get("architecture_type") or "Deep Convolutional Network"
    training_type = pred_data.get("training_type") or "Transfer Learning"

    raw_thresh = pred_data.get("confidence_threshold", 50.0)
    try:
        conf_threshold = float(raw_thresh) if raw_thresh is not None else 50.0
    except (ValueError, TypeError):
        conf_threshold = 50.0

    # Apple Healthcare Severity Colors
    severity_colors = ["#30D158", "#64D2FF", "#FF9F0A", "#FF851B", "#FF453A"]
    stage_color = severity_colors[class_id] if 0 <= class_id < len(severity_colors) else "#0A84FF"

    # Generate probability bars markup
    prob_bars_html = render_probability_bars(prob_dict, class_id, stage_color)

    # Low-confidence badge / banner
    low_conf_banner_html = ""
    if is_low_conf:
        low_conf_banner_html = f"""
        <div style="margin-bottom: 20px; padding: 14px 18px; background: rgba(255, 159, 10, 0.10); border: 1px solid rgba(255, 159, 10, 0.35); border-radius: var(--radius-sm); display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.35rem;">⚠️</span>
            <div>
                <div style="font-size: 0.82rem; font-weight: 700; color: var(--accent-amber); text-transform: uppercase; letter-spacing: 0.05em;">
                    Low-Confidence Screening Alert ({confidence_pct:.1f}% &lt; {conf_threshold:.0f}%)
                </div>
                <div style="font-size: 0.84rem; color: var(--text-secondary); margin-top: 3px; line-height: 1.45;">
                    {conf_status}
                </div>
            </div>
        </div>
        """

    result_card_html = f"""
    <div class="apple-card apple-result-card" style="margin-top: 24px; border-color: {stage_color}35; box-shadow: 0 12px 36px -6px rgba(0,0,0,0.5), 0 0 24px {stage_color}18;">
        <!-- Header Provenance: Stage Pill & Highest Confidence Tag -->
        <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 18px;">
            <div style="display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: var(--radius-pill); background: {stage_color}18; border: 1px solid {stage_color}45; color: {stage_color}; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;">
                <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: {stage_color};"></span>
                Stage {class_id} / 4 — {short_name}
            </div>
            <div style="display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: var(--radius-pill); background: rgba(10, 132, 255, 0.12); border: 1px solid rgba(10, 132, 255, 0.35); color: var(--accent-blue); font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">
                <span>🏆</span>
                <span>Highest Model Confidence</span>
                <span style="color: var(--text-tertiary);">•</span>
                <span style="color: var(--text-primary);">{model_name}</span>
            </div>
        </div>

        {low_conf_banner_html}

        <!-- Main Title & Large Confidence Score -->
        <div style="display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; margin-bottom: 24px; padding-bottom: 20px; border-bottom: 1px solid var(--border-hairline);">
            <div>
                <h2 style="font-size: clamp(1.6rem, 2.8vw, 2.2rem); font-weight: 700; letter-spacing: -0.03em; color: var(--text-primary); margin: 0 0 4px 0;">
                    {class_name}
                </h2>
                <div style="font-size: 0.88rem; color: var(--text-secondary);">
                    Primary finding evaluated via <strong>{model_name}</strong> ({architecture_type}).
                </div>
            </div>
            <div style="text-align: right; margin-top: 8px;">
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: var(--text-tertiary); letter-spacing: 0.06em;">
                    Model Confidence (Softmax)
                </div>
                <div style="font-size: 2.4rem; font-weight: 800; color: {stage_color}; letter-spacing: -0.03em; line-height: 1;">
                    {confidence_pct:.2f}%
                </div>
                <div style="font-size: 0.70rem; color: var(--text-tertiary); margin-top: 3px;">
                    Top-1 predicted class probability
                </div>
            </div>
        </div>

        <!-- Clinical Biomarkers & Follow-Up Grid -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px;">
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border-hairline); border-radius: var(--radius-md); padding: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: var(--accent-blue); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">
                    Biomarker Interpretation
                </div>
                <div style="font-size: 0.90rem; color: var(--text-secondary); line-height: 1.55; text-align: justify; text-justify: inter-word; text-align-last: left;">
                    {description}
                </div>
            </div>

            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border-hairline); border-radius: var(--radius-md); padding: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: {stage_color}; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">
                    Recommended Follow-Up
                </div>
                <div style="font-size: 0.90rem; color: var(--text-primary); line-height: 1.55; text-align: justify; text-justify: inter-word; text-align-last: left;">
                    {recommendation}
                </div>
            </div>
        </div>

        <!-- Horizontal Probability Distribution -->
        <div style="margin-bottom: 22px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
                <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.06em;">
                    Class Probability Distribution ({model_name})
                </div>
                <div style="font-size: 0.72rem; color: var(--text-tertiary);">
                    Normalized Softmax Probabilities
                </div>
            </div>
            {prob_bars_html}
        </div>

        <!-- Training Dataset & Model Provenance Section -->
        <div style="padding-top: 16px; border-top: 1px solid var(--border-hairline); display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; font-size: 0.78rem;">
            <div>
                <span style="color: var(--text-tertiary); display: block; font-size: 0.70rem; text-transform: uppercase;">Training Dataset</span>
                <span style="color: var(--text-primary); font-weight: 600;">{dataset_name}</span>
            </div>
            <div>
                <span style="color: var(--text-tertiary); display: block; font-size: 0.70rem; text-transform: uppercase;">Architecture</span>
                <span style="color: var(--text-primary); font-weight: 600;">{architecture_type}</span>
            </div>
            <div>
                <span style="color: var(--text-tertiary); display: block; font-size: 0.70rem; text-transform: uppercase;">Training Type</span>
                <span style="color: var(--text-primary); font-weight: 600;">{training_type}</span>
            </div>
            <div>
                <span style="color: var(--text-tertiary); display: block; font-size: 0.70rem; text-transform: uppercase;">Screening Scale</span>
                <span style="color: var(--text-primary); font-weight: 600;">5 Stages (ICDR 0 to 4)</span>
            </div>
        </div>

        <!-- Educational Screening Disclaimer -->
        <div style="margin-top: 18px; padding: 12px 16px; background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border-hairline); border-radius: var(--radius-sm); font-size: 0.76rem; color: var(--text-tertiary); line-height: 1.5;">
            <strong style="color: var(--text-secondary);">Educational Screening Aid Notice:</strong> RetinaCare AI is an assistive deep learning screening technology and does not provide an authorized medical diagnosis. Softmax confidence represents individual model certainty for this specific fundus image and is not a calibrated population probability. Please consult a qualified ophthalmologist for clinical evaluation.
        </div>
    </div>
    """
    render_html(result_card_html)
