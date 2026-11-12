"""
==============================================================================
RetinaCare AI - Animated Probability Distribution Component
Apple-Inspired Micro-Interactions & Staggered Bar Animations
==============================================================================
"""

from __future__ import annotations
import streamlit as st
from typing import Dict, Any
from utils.config import CLASS_SHORT_NAMES
from utils.html_utils import render_html


def render_probability_bars(probabilities: Dict[str, float], predicted_class_id: int, stage_color: str) -> str:
    """
    Generates HTML markup for animated staggered probability bars across all 5 classes.
    """
    if not probabilities:
        return ""

    prob_rows_html = ""
    for idx, (lbl, raw_pct) in enumerate(probabilities.items()):
        is_winner = (idx == predicted_class_id)
        text_color = "var(--text-primary)" if is_winner else "var(--text-secondary)"
        font_weight = "700" if is_winner else "500"
        bar_color = stage_color if is_winner else "rgba(255, 255, 255, 0.22)"
        stagger_class = f"prob-row-{idx}"

        try:
            pct = float(raw_pct)
        except (ValueError, TypeError):
            pct = 0.0

        label_display = CLASS_SHORT_NAMES[idx] if idx < len(CLASS_SHORT_NAMES) else str(lbl)

        prob_rows_html += f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; font-size: 0.88rem;">
            <span style="width: 150px; text-align: left; font-weight: {font_weight}; color: {text_color};">
                {label_display}
            </span>
            <div style="flex: 1; height: 8px; background: rgba(255, 255, 255, 0.08); border-radius: var(--radius-pill); margin: 0 16px; overflow: hidden;">
                <div class="prob-bar-animate {stagger_class}" style="height: 100%; width: {pct:.1f}%; background-color: {bar_color}; border-radius: var(--radius-pill);"></div>
            </div>
            <span style="width: 55px; text-align: right; font-weight: {font_weight}; color: {text_color}; font-variant-numeric: tabular-nums;">
                {pct:.1f}%
            </span>
        </div>
        """
    return prob_rows_html
