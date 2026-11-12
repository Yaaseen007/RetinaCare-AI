"""
==============================================================================
RetinaCare AI - Architecture Telemetry & Multi-Model Readiness Component
Apple-Inspired Minimalist Engine Status Card (No Manual Selection Required)
==============================================================================
"""

from __future__ import annotations
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from utils.html_utils import render_html
from src.model_manager import (
    CANONICAL_MODEL_NAMES,
    get_model_status,
    get_all_models_status
)
from utils.api_client import fetch_models_list, is_backend_alive


def render_model_selector() -> Tuple[str, Dict[str, Any]]:
    """
    Renders the Apple-style Multi-Architecture Readiness card.
    The user does NOT choose a model; all available models run automatically.

    Queries model statuses via FastAPI backend /api/models when available,
    falling back to local registry.

    Returns:
        ("All Available Models", status_dict)
    """
    backend_online = is_backend_alive()
    api_badge = (
        '<span style="display: inline-flex; align-items: center; gap: 5px; color: var(--accent-emerald); font-size: 0.72rem; font-weight: 700; background: rgba(48, 209, 88, 0.12); padding: 3px 8px; border-radius: var(--radius-pill); border: 1px solid rgba(48, 209, 88, 0.35); text-transform: uppercase;">● FASTAPI BACKEND ONLINE</span>'
        if backend_online
        else '<span style="display: inline-flex; align-items: center; gap: 5px; color: var(--text-tertiary); font-size: 0.72rem; font-weight: 600; background: rgba(255, 255, 255, 0.05); padding: 3px 8px; border-radius: var(--radius-pill); border: 1px solid var(--border-hairline); text-transform: uppercase;">○ LOCAL ENGINE ACTIVE</span>'
    )

    # Fetch statuses from FastAPI backend or local registry fallback
    all_statuses = None
    if backend_online:
        try:
            all_statuses = fetch_models_list()
        except Exception:
            all_statuses = None

    if not all_statuses:
        all_statuses = get_all_models_status()

    available_models = [s for s in all_statuses if s.get("is_available", False)]
    total_count = len(all_statuses)
    avail_count = len(available_models)

    # Build model status chips markup
    chips_html = ""
    for s in all_statuses:
        is_ready = s.get("is_available", False)
        dot_color = "#30D158" if is_ready else "#FF9F0A"
        target_role = "WEBSITE MODEL" if s["canonical_name"] == "VGG16" else ("MOBILE APP MODEL" if s["canonical_name"] == "MobileNetV2" else "BENCHMARK")
        label_text = f"READY • {target_role}" if is_ready else "NOT TRAINED"
        chip_bg = "rgba(48, 209, 88, 0.08)" if is_ready else "rgba(255, 255, 255, 0.03)"
        chip_border = "rgba(48, 209, 88, 0.25)" if is_ready else "rgba(255, 255, 255, 0.08)"

        chips_html += f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 7px 12px; background: {chip_bg}; border: 1px solid {chip_border}; border-radius: var(--radius-sm); font-size: 0.78rem;">
            <span style="font-weight: 600; color: var(--text-primary);">{s['display_name']}</span>
            <span style="display: inline-flex; align-items: center; gap: 5px; color: {dot_color}; font-weight: 700; font-size: 0.70rem; letter-spacing: 0.04em;">
                <span style="width: 5px; height: 5px; border-radius: 50%; background: {dot_color}; display: inline-block;"></span>
                {label_text}
            </span>
        </div>
        """

    card_html = f"""
    <div class="apple-card" style="padding: 18px 20px; border-color: var(--border-hairline);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border-hairline);">
            <div>
                <div class="apple-caption" style="color: var(--accent-blue); margin-bottom: 2px;">NEURAL ARCHITECTURE ROUTING</div>
                <h3 style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary); margin: 0;">Dedicated Platform Allocation</h3>
            </div>
            <div>
                {api_badge}
            </div>
        </div>

        <div style="font-size: 0.83rem; color: var(--text-secondary); line-height: 1.5; margin-bottom: 14px;">
            RetinaCare AI architecture employs <strong>VGG16 exclusively</strong> for high-capacity website predictions (90.0% accuracy), and reserves <strong>MobileNetV2</strong> for the Flutter mobile application (88.0% accuracy, 1.0000 ROC-AUC).
        </div>

        <div style="display: flex; flex-direction: column; gap: 6px; margin-bottom: 12px;">
            {chips_html}
        </div>

        <div style="font-size: 0.72rem; color: var(--text-tertiary); line-height: 1.4; border-top: 1px solid var(--border-hairline); padding-top: 10px;">
            ℹ️ No manual model selection needed. All available models execute during analysis, followed by ranked alternative comparisons.
        </div>
    </div>
    """
    render_html(card_html)

    return "All Available Models", {
        "is_available": avail_count > 0,
        "available_count": avail_count,
        "total_count": total_count,
        "all_statuses": all_statuses
    }
