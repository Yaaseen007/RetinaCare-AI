"""
RetinaCare AI - Metrics Component
Apple-Inspired Minimalist Telemetry Grid for Model Performance
"""

import json
import streamlit as st
from typing import Dict, Any, Optional
from utils.config import METRICS_JSON_PATH
from utils.html_utils import render_html
from utils.api_client import fetch_metrics_via_api, is_backend_alive


def load_evaluation_metrics() -> Optional[Dict[str, Any]]:
    """Loads evaluation metrics from FastAPI /api/model-metrics or local results/metrics.json."""
    if is_backend_alive():
        try:
            api_metrics = fetch_metrics_via_api()
            if api_metrics and api_metrics.get("primary_model_metrics"):
                return api_metrics["primary_model_metrics"]
        except Exception:
            pass

    if METRICS_JSON_PATH.is_file():
        try:
            with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def render_performance_metrics_grid():
    """
    Renders the spacious Apple metric cards: Accuracy, Precision, Recall, F1 Score.
    """
    metrics = load_evaluation_metrics()

    if metrics:
        acc = f"{metrics['accuracy'] * 100:.1f}%" if "accuracy" in metrics else "N/A"
        prec = f"{metrics['macro_precision'] * 100:.1f}%" if "macro_precision" in metrics else "N/A"
        rec = f"{metrics['macro_recall'] * 100:.1f}%" if "macro_recall" in metrics else "N/A"
        f1 = f"{metrics['macro_f1'] * 100:.1f}%" if "macro_f1" in metrics else "N/A"
    else:
        acc, prec, rec, f1 = "N/A", "N/A", "N/A", "N/A"

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        render_html(
            f"""
            <div class="apple-card" style="padding: 22px 20px;">
                <div class="apple-caption" style="color: var(--accent-blue);">TEST ACCURACY</div>
                <div style="font-size: 2.6rem; font-weight: 700; letter-spacing: -0.04em; color: var(--text-primary); margin: 6px 0;">
                    {acc}
                </div>
                <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.4; text-align: justify; text-justify: inter-word; text-align-last: left;">
                    Overall validation correctness across test samples
                </div>
            </div>
            """
        )

    with col2:
        render_html(
            f"""
            <div class="apple-card" style="padding: 22px 20px;">
                <div class="apple-caption" style="color: #64D2FF;">MACRO PRECISION</div>
                <div style="font-size: 2.6rem; font-weight: 700; letter-spacing: -0.04em; color: var(--text-primary); margin: 6px 0;">
                    {prec}
                </div>
                <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.4; text-align: justify; text-justify: inter-word; text-align-last: left;">
                    Positive predictive value across all clinical stages
                </div>
            </div>
            """
        )

    with col3:
        render_html(
            f"""
            <div class="apple-card" style="padding: 22px 20px;">
                <div class="apple-caption" style="color: var(--accent-emerald);">MACRO RECALL</div>
                <div style="font-size: 2.6rem; font-weight: 700; letter-spacing: -0.04em; color: var(--text-primary); margin: 6px 0;">
                    {rec}
                </div>
                <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.4; text-align: justify; text-justify: inter-word; text-align-last: left;">
                    Sensitivity across microvascular severity classes
                </div>
            </div>
            """
        )

    with col4:
        render_html(
            f"""
            <div class="apple-card" style="padding: 22px 20px;">
                <div class="apple-caption" style="color: var(--accent-amber);">MACRO F1 SCORE</div>
                <div style="font-size: 2.6rem; font-weight: 700; letter-spacing: -0.04em; color: var(--text-primary); margin: 6px 0;">
                    {f1}
                </div>
                <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.4; text-align: justify; text-justify: inter-word; text-align-last: left;">
                    Harmonic mean balance of precision and recall
                </div>
            </div>
            """
        )
