"""
==============================================================================
RetinaCare AI - Multi-Model Benchmark Comparison Component
Apple-Inspired Comparative Matrix, Agreement Calculation & Ethical Labelling
==============================================================================
"""

from __future__ import annotations
import streamlit as st
from typing import Any, List, Dict
from utils.html_utils import render_html
from utils.api_client import compare_via_api, is_backend_alive, BackendConnectionError
from src.prediction import compare_all_available_models, compute_model_agreement


def render_model_comparison_section(image_source: Any):
    """
    Renders the interactive 'Compare Available Models' component.
    Executes comparison via FastAPI backend (/api/compare) with fallback.
    """
    render_html(
        """
        <div style="margin-top: 28px; padding-top: 24px; border-top: 1px solid var(--border-hairline);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <div>
                    <div class="apple-caption" style="color: var(--accent-blue);">CROSS-ARCHITECTURE VALIDATION</div>
                    <h3 style="font-size: 1.25rem; font-weight: 700; color: var(--text-primary); margin: 0 0 4px 0;">
                        Compare AI Models
                    </h3>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">
                        Evaluate this retinal fundus photograph across all available trained deep learning architectures simultaneously via the FastAPI backend.
                    </div>
                </div>
            </div>
        </div>
        """
    )

    if "comparison_results" not in st.session_state:
        st.session_state["comparison_results"] = None

    c_spacer_l, c_btn_col, c_spacer_r = st.columns([1.0, 1.8, 1.0])
    with c_btn_col:
        run_comparison = st.button(
            "⚡ Compare Available Models",
            type="secondary",
            use_container_width=True,
            key="btn_run_model_comparison"
        )

    if run_comparison:
        with st.spinner("Submitting to FastAPI backend (/api/compare) for multi-model evaluation..."):
            try:
                resp = compare_via_api(image_source)
                results = resp.get("results", [])
            except BackendConnectionError:
                results = compare_all_available_models(image_source)
            st.session_state["comparison_results"] = results
            st.rerun()

    results = st.session_state.get("comparison_results")
    if results is not None:
        render_comparison_results_card(results)


def render_comparison_results_card(results: List[Dict[str, Any]]):
    """
    Renders the comparison table card and model agreement banner.
    """
    agreement = compute_model_agreement(results)

    # Table rows HTML
    rows_html = ""
    for r in results:
        model_name = r["model_name"]
        status = r["status"]
        prediction = r["prediction"]
        conf_pct = r["confidence_pct"]

        if status == "Ready" and conf_pct is not None:
            status_badge = "<span style='color: var(--accent-emerald); font-weight: 600;'>● READY</span>"
            pred_cell = f"<span style='font-weight: 600; color: var(--text-primary);'>{prediction}</span>"
            conf_cell = f"<span style='font-weight: 700; color: var(--accent-blue); font-variant-numeric: tabular-nums;'>{conf_pct:.2f}%</span>"
        elif status == "Not Available":
            status_badge = "<span style='color: var(--text-tertiary); font-weight: 500;'>○ NOT TRAINED</span>"
            pred_cell = "<span style='color: var(--text-tertiary); font-style: italic;'>Not Available</span>"
            conf_cell = "<span style='color: var(--text-tertiary);'>—</span>"
        else:
            status_badge = "<span style='color: var(--accent-red); font-weight: 600;'>✕ ERROR</span>"
            pred_cell = "<span style='color: var(--accent-red);'>Inference Error</span>"
            conf_cell = "<span style='color: var(--text-tertiary);'>—</span>"

        rows_html += f"""
        <tr style="border-bottom: 1px solid var(--border-subtle); font-size: 0.88rem;">
            <td style="padding: 12px 14px; font-weight: 600; color: var(--text-primary);">{model_name}</td>
            <td style="padding: 12px 14px;">{status_badge}</td>
            <td style="padding: 12px 14px;">{pred_cell}</td>
            <td style="padding: 12px 14px; text-align: right;">{conf_cell}</td>
        </tr>
        """

    # Model Agreement Banner
    agreement_html = ""
    if agreement["available_count"] > 0:
        agreement_html = f"""
        <div style="margin-top: 18px; padding: 14px 18px; background: rgba(10, 132, 255, 0.08); border: 1px solid rgba(10, 132, 255, 0.28); border-radius: var(--radius-sm);">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.15rem;">🎯</span>
                    <div>
                        <div style="font-size: 0.76rem; font-weight: 700; color: var(--accent-blue); text-transform: uppercase; letter-spacing: 0.06em;">
                            Model Prediction Agreement
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-primary); margin-top: 2px;">
                            {agreement['agreement_label']}
                        </div>
                    </div>
                </div>
                <div style="font-size: 0.76rem; color: var(--text-secondary); max-width: 320px; line-height: 1.4; text-align: right;">
                    Preliminary screening alignment across available models. <strong>This is NOT a medical consensus.</strong>
                </div>
            </div>
        </div>
        """

    table_card_html = f"""
    <div class="apple-card" style="margin-top: 16px; padding: 20px 24px;">
        <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px;">
            Cross-Model Diagnostic Matrix (5 Architectures)
        </div>
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead>
                    <tr style="border-bottom: 1px solid var(--border-hairline); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-tertiary);">
                        <th style="padding: 8px 14px;">Architecture</th>
                        <th style="padding: 8px 14px;">Status</th>
                        <th style="padding: 8px 14px;">Prediction</th>
                        <th style="padding: 8px 14px; text-align: right;">Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        {agreement_html}
    </div>
    """
    render_html(table_card_html)
