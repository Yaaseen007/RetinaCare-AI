"""
RetinaCare AI - Charts & Diagnostic Visualizations Component
Apple-Inspired Minimalist Visual Analytics for Model Performance
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
from utils.config import (
    TRAINING_HISTORY_PLOT_PATH,
    CONFUSION_MATRIX_PATH,
    RESULTS_DIR,
    CLASS_SHORT_NAMES
)
from utils.html_utils import render_html
from utils.api_client import fetch_metrics_via_api, is_backend_alive
from src.evaluate import get_comparison_dataframe

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def render_performance_charts():
    """
    Renders training dynamics and confusion matrix with Apple-inspired minimalist visual language.
    """
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        with st.container(border=True):
            render_html(
                """
                <div style="margin-bottom: 16px;">
                    <div class="apple-caption" style="color: var(--accent-blue); margin-bottom: 6px;">TRAINING DYNAMICS</div>
                    <div class="apple-card-title">Accuracy Convergence & Loss</div>
                    <div style="font-size: 0.86rem; color: var(--text-secondary); text-align: justify; text-justify: inter-word; text-align-last: left;">
                        Cross-entropy loss and top-1 accuracy trajectory across fine-tuning epochs.
                    </div>
                </div>
                """
            )

            history_json_path = RESULTS_DIR / "training_history.json"
            if history_json_path.is_file() and HAS_PLOTLY:
                try:
                    with open(history_json_path, "r", encoding="utf-8") as f:
                        hist_data = json.load(f)
                    epochs = list(range(1, len(hist_data.get("accuracy", [])) + 1))
                    t_acc = hist_data.get("accuracy", [])
                    v_acc = hist_data.get("val_accuracy", [])

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=epochs, y=t_acc, mode='lines', name='Train Acc',
                        line=dict(color='#0A84FF', width=2.4)
                    ))
                    fig.add_trace(go.Scatter(
                        x=epochs, y=v_acc, mode='lines', name='Val Acc',
                        line=dict(color='#64D2FF', width=2.2, dash='dot')
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(title=dict(text='Epoch', font=dict(color='#86868B', size=11)),
                                   showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                                   zeroline=False, tickfont=dict(color='#86868B', size=10)),
                        yaxis=dict(title=dict(text='Accuracy', font=dict(color='#86868B', size=11)),
                                   showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                                   zeroline=False, tickfont=dict(color='#86868B', size=10),
                                   range=[0.5, 1.0]),
                        margin=dict(l=20, r=20, t=10, b=20),
                        height=280,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                    font=dict(color='#F5F5F7', size=11))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    if TRAINING_HISTORY_PLOT_PATH.is_file():
                        st.image(str(TRAINING_HISTORY_PLOT_PATH), use_container_width=True)
            elif TRAINING_HISTORY_PLOT_PATH.is_file():
                st.image(str(TRAINING_HISTORY_PLOT_PATH), use_container_width=True)
            else:
                st.info("Training history plot will be displayed here after model evaluation.")

    with c2:
        with st.container(border=True):
            render_html(
                """
                <div style="margin-bottom: 16px;">
                    <div class="apple-caption" style="color: #64D2FF; margin-bottom: 6px;">DISCRIMINATION MATRIX</div>
                    <div class="apple-card-title">Normalized Confusion Matrix</div>
                    <div style="font-size: 0.86rem; color: var(--text-secondary); text-align: justify; text-justify: inter-word; text-align-last: left;">
                        Observed clinical ground truth vs. predicted classification frequencies.
                    </div>
                </div>
                """
            )

            if CONFUSION_MATRIX_PATH.is_file():
                st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
            else:
                st.info("Confusion matrix visualization unavailable. Run model evaluation to generate authentic results.")


def render_model_comparison_table():
    """
    Renders deep learning architecture benchmarks with translucent table styling.
    Queries FastAPI /api/model-metrics when backend is alive, falling back to local results.
    """
    with st.container(border=True):
        render_html(
            """
            <div style="margin-bottom: 16px;">
                <div class="apple-caption" style="color: var(--accent-blue); margin-bottom: 6px;">BENCHMARK EVALUATION</div>
                <div class="apple-card-title">Deep Learning Architecture Comparison</div>
                <div style="font-size: 0.86rem; color: var(--text-secondary); text-align: justify; text-justify: inter-word; text-align-last: left;">
                    Comparative evaluation across convolutional neural backbones on the Diabetic Retinopathy test cohort.
                </div>
            </div>
            """
        )

        comp_df = None
        if is_backend_alive():
            try:
                metrics_resp = fetch_metrics_via_api()
                if metrics_resp and "model_comparison" in metrics_resp:
                    comp_data = metrics_resp["model_comparison"]
                    rows = []
                    for model_name, metrics_dict in comp_data.items():
                        row = {"Model Architecture": model_name}
                        row.update(metrics_dict)
                        rows.append(row)
                    if rows:
                        comp_df = pd.DataFrame(rows)
            except Exception:
                comp_df = None

        if comp_df is None:
            comp_df = get_comparison_dataframe()

        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        render_html(
            """
            <div style="font-size: 0.78rem; color: var(--text-tertiary); margin-top: 12px; text-align: justify; text-justify: inter-word; text-align-last: left;">
                Scientific Transparency: Architectures not evaluated locally on your device are strictly designated as <code>Not evaluated</code>.
            </div>
            """
        )
