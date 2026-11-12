"""
==============================================================================
RetinaCare AI - Automated Multi-Model Prediction Orchestrator
Apple-Inspired Spatial Execution, Highest-Confidence Selection & Transitions
==============================================================================
"""

from __future__ import annotations
import time
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Any, Optional, List
from utils.config import CLASS_NAMES, CLASS_SHORT_NAMES
from utils.html_utils import render_html
from utils.api_client import (
    compare_via_api,
    predict_via_api,
    is_backend_alive,
    BackendConnectionError,
    BackendAPIError
)
from src.prediction import (
    compare_all_available_models,
    validate_and_rank_model_predictions,
    run_model_inference,
    CONFIDENCE_THRESHOLD_PCT
)
from src.validation import ImageValidationError
from components.result_card import render_prediction_result_card
from components.alternative_viewer import render_alternative_predictions_viewer
from components.comparison import render_comparison_results_card
from components.error_state import (
    render_context_aware_error,
    render_missing_model_error,
    render_invalid_image_error
)

__all__ = [
    "run_automated_analysis_pipeline",
    "run_ai_analysis_pipeline",
    "append_to_session_history",
    "render_prediction_result",
    "render_prediction_history",
    "reset_analysis_state",
    "render_empty_state",
    "render_error_state"
]


def run_automated_analysis_pipeline(image_source: Any) -> Dict[str, Any]:
    """
    Executes automated multi-model AI diagnostic pipeline across all available backbones:
    - User does not choose a model beforehand.
    - Evaluates all available models concurrently via FastAPI /api/compare (or local fallback).
    - Displays clean analysis state with subtle animated retinal scan and skeleton placeholder.
    - Validates all probability distributions (finite, normalized, 5 classes, argmax correspondence).
    - Automatically selects highest-confidence prediction as primary result.
    - Sorts alternative predictions by descending predicted-class confidence.
    """
    status_placeholder = st.empty()
    skeleton_placeholder = st.empty()

    # 1. Authentic loading indicator with scanning animation
    with status_placeholder.container():
        render_html(
            """
            <div style="display: flex; align-items: center; justify-content: center; gap: 10px; padding: 12px 22px; background: rgba(10, 132, 255, 0.10); border-radius: var(--radius-pill); border: 1px solid rgba(10, 132, 255, 0.30); font-size: 0.88rem; font-weight: 600; color: var(--accent-blue); text-align: center; margin: 12px 0;">
                <span class="pulse-indicator" style="background: var(--accent-blue); box-shadow: 0 0 10px var(--accent-blue);"></span>
                <span>Executing multi-model inference across all 5 deep learning architectures...</span>
            </div>
            """
        )

    # 2. Elegant skeleton placeholder while waiting for genuine inference
    with skeleton_placeholder.container():
        render_html(
            """
            <div class="apple-card" style="margin-top: 20px; padding: 24px; border-color: rgba(255,255,255,0.06);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px;">
                    <div class="skeleton-loader" style="height: 24px; width: 140px; border-radius: 9999px;"></div>
                    <div class="skeleton-loader" style="height: 24px; width: 180px; border-radius: 9999px;"></div>
                </div>
                <div class="skeleton-loader" style="height: 38px; width: 60%; margin-bottom: 22px;"></div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 22px;">
                    <div class="skeleton-loader" style="height: 74px; border-radius: 12px;"></div>
                    <div class="skeleton-loader" style="height: 74px; border-radius: 12px;"></div>
                </div>
                <div class="skeleton-loader" style="height: 90px; width: 100%; border-radius: 12px;"></div>
            </div>
            """
        )

    # 3. Dedicated website inference via VGG16 (FastAPI backend or local engine fallback)
    try:
        from utils.api_client import predict_web_via_api
        vgg_pred = predict_web_via_api(image_source)
        raw_results = [vgg_pred]
    except BackendConnectionError:
        # Fallback to local inference engine if backend process is not active
        vgg_pred = run_model_inference(model_name="VGG16", image_input=image_source)
        raw_results = [vgg_pred]
    except BackendAPIError as api_err:
        status_placeholder.empty()
        skeleton_placeholder.empty()
        if api_err.status_code == 400:
            raise ImageValidationError(api_err.detail)
        else:
            raise RuntimeError(api_err.detail)

    # Clear loading elements as soon as real results arrive
    status_placeholder.empty()
    skeleton_placeholder.empty()

    # 4. Strict mathematical verification & ranking
    ranked_info = validate_and_rank_model_predictions(raw_results, confidence_threshold=CONFIDENCE_THRESHOLD_PCT)

    if not ranked_info["has_valid_prediction"]:
        raise RuntimeError(
            "None of the available deep learning models produced a valid probability distribution for this image."
        )

    primary = ranked_info["primary_result"]
    alternatives = ranked_info["alternative_results"]
    unranked = ranked_info["unranked_or_failed"]

    # 5. Store in volatile session state
    st.session_state["primary_prediction"] = primary
    st.session_state["alternative_predictions"] = alternatives
    st.session_state["unranked_or_failed"] = unranked
    st.session_state["comparison_results"] = raw_results
    st.session_state["prediction_result"] = primary  # For history and backward compatibility
    st.session_state["scroll_to_results"] = True

    # Append primary finding to history
    append_to_session_history(primary)

    return primary


def run_ai_analysis_pipeline(model_name: str, image_source: Any) -> Dict[str, Any]:
    """
    Backwards-compatible single-model execution helper.
    """
    try:
        pred_data = predict_via_api(model_name=model_name, image_source=image_source)
    except BackendConnectionError:
        pred_data = run_model_inference(model_name=model_name, image_input=image_source)
    except BackendAPIError as api_err:
        if api_err.status_code == 400:
            raise ImageValidationError(api_err.detail)
        raise RuntimeError(api_err.detail)

    append_to_session_history(pred_data)
    st.session_state["prediction_result"] = pred_data
    return pred_data


def append_to_session_history(pred_data: Dict[str, Any]):
    """Appends prediction record to volatile session-based history."""
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []

    history_item = {
        "timestamp": pred_data.get("timestamp", datetime.now().strftime("%H:%M:%S")),
        "model_name": pred_data.get("model_name", "AI Model"),
        "prediction": pred_data.get("class_name", pred_data.get("prediction", "Unknown")),
        "short_name": pred_data.get("short_name", "Stage"),
        "confidence_pct": pred_data.get("confidence_pct", 0.0),
        "severity_color": pred_data.get("severity_color", "#0A84FF")
    }
    st.session_state["prediction_history"].append(history_item)


def render_prediction_result(pred_data: Dict[str, Any]):
    """
    Renders the complete diagnostic results suite:
    1. Smooth focus anchor
    2. Primary Result Card (Highest model confidence)
    3. Animated Alternative Predictions Viewer
    4. Cross-Architecture Consensus Table
    """
    # Smooth focus anchor
    render_html("<div id='analysis-results' style='position: relative; top: -20px;'></div>")

    # Client-side smooth scroll trigger on first reveal
    if st.session_state.get("scroll_to_results"):
        st.session_state["scroll_to_results"] = False
        components.html(
            """
            <script>
            (function() {
                try {
                    const el = window.parent.document.getElementById('analysis-results');
                    if (el) {
                        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                } catch(e) {}
            })();
            </script>
            """,
            height=0,
            width=0
        )

    # 1. Primary Result Card (Highest Softmax Confidence)
    primary = st.session_state.get("primary_prediction") or pred_data
    if hasattr(primary, "model_dump"):
        primary = primary.model_dump()
    elif hasattr(primary, "dict") and callable(primary.dict):
        primary = primary.dict()
    elif not isinstance(primary, dict):
        try:
            primary = dict(primary)
        except Exception:
            primary = {}

    # Guarantee class_name and short_name in primary
    if not primary.get("class_name"):
        raw_c = primary.get("class_id", primary.get("predicted_class_index", 0))
        try:
            cid = int(raw_c) if raw_c is not None else 0
        except (ValueError, TypeError):
            cid = 0
        cname = (
            primary.get("predicted_class_name")
            or primary.get("prediction")
            or (CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"Stage {cid}")
        )
        primary["class_name"] = cname
        primary["predicted_class_name"] = cname
    if not primary.get("short_name"):
        raw_c = primary.get("class_id", 0)
        try:
            cid = int(raw_c) if raw_c is not None else 0
        except (ValueError, TypeError):
            cid = 0
        primary["short_name"] = CLASS_SHORT_NAMES[cid] if 0 <= cid < len(CLASS_SHORT_NAMES) else f"Stage {cid}"

    render_prediction_result_card(primary)

    # 2. Animated Alternative Predictions Viewer
    alts = st.session_state.get("alternative_predictions", [])
    unranked = st.session_state.get("unranked_or_failed", [])
    if alts or unranked:
        render_alternative_predictions_viewer(
            alternative_results=alts,
            primary_result=primary,
            unranked_or_failed=unranked
        )

    # 3. Cross-Architecture Matrix & Consensus Banner
    comp_results = st.session_state.get("comparison_results")
    if comp_results:
        render_html("<div style='height: 20px;'></div>")
        render_comparison_results_card(comp_results)


def render_prediction_history():
    """
    Renders lightweight session-based prediction history table (Section 24).
    """
    history = st.session_state.get("prediction_history", [])
    if not history:
        return

    render_html(
        """
        <div style="margin-top: 32px; padding-top: 20px; border-top: 1px solid var(--border-hairline);">
            <div class="apple-caption" style="color: var(--text-tertiary); margin-bottom: 4px;">VOLATILE SESSION TELEMETRY</div>
            <h4 style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin: 0 0 12px 0;">
                Prediction History (Current Session)
            </h4>
        </div>
        """
    )

    rows_html = ""
    for item in reversed(history[-5:]):  # Show last 5
        rows_html += f"""
        <tr style="border-bottom: 1px solid var(--border-subtle); font-size: 0.85rem;">
            <td style="padding: 10px 14px; color: var(--text-tertiary); font-family: monospace;">{item['timestamp']}</td>
            <td style="padding: 10px 14px; font-weight: 600; color: var(--text-primary);">{item['model_name']}</td>
            <td style="padding: 10px 14px;">
                <span style="display: inline-flex; align-items: center; gap: 6px; color: {item['severity_color']}; font-weight: 600;">
                    <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: {item['severity_color']};"></span>
                    {item['short_name']}
                </span>
            </td>
            <td style="padding: 10px 14px; text-align: right; font-weight: 700; color: var(--text-primary); font-variant-numeric: tabular-nums;">
                {item['confidence_pct']:.2f}%
            </td>
        </tr>
        """

    history_card_html = f"""
    <div class="apple-card" style="padding: 16px 20px;">
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead>
                    <tr style="border-bottom: 1px solid var(--border-hairline); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-tertiary);">
                        <th style="padding: 6px 14px;">Time</th>
                        <th style="padding: 6px 14px;">Model</th>
                        <th style="padding: 6px 14px;">Prediction</th>
                        <th style="padding: 6px 14px; text-align: right;">Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    """
    render_html(history_card_html)


def reset_analysis_state(keep_model: bool = True):
    """
    Executes smooth reset of analysis state.
    Clears image, primary prediction, alternative predictions, comparison results, and processing flags.
    """
    st.session_state["prediction_result"] = None
    st.session_state["primary_prediction"] = None
    st.session_state["alternative_predictions"] = None
    st.session_state["unranked_or_failed"] = None
    st.session_state["prediction_source_id"] = None
    st.session_state["comparison_results"] = None
    st.session_state["is_analyzing"] = False
    st.session_state["scroll_to_results"] = False
    if not keep_model:
        st.session_state["selected_model_choice"] = None


def render_empty_state():
    """Renders the Apple minimal empty state when no image is loaded."""
    render_html(
        """
        <div class="apple-card" style="padding: 64px 24px; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <div style="font-size: 3rem; margin-bottom: 16px; opacity: 0.85;">👁️</div>
            <h3 style="font-size: 1.4rem; font-weight: 600; letter-spacing: -0.02em; color: var(--text-primary); margin: 0 0 8px 0; text-align: center;">
                Your retinal analysis starts here.
            </h3>
            <p style="font-size: 0.95rem; color: var(--text-secondary); max-width: 440px; margin: 0 auto; line-height: 1.5; text-align: center !important; text-align-last: center !important; display: block;">
                Select a verified reference sample or upload a digital color fundus photograph to inspect microvasculature.
            </p>
        </div>
        """
    )


def render_error_state(error_message: str):
    """Legacy fallback error banner."""
    render_context_aware_error(
        error_type="generic",
        title="Analysis Notice",
        message=error_message
    )
