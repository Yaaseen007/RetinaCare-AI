"""
==============================================================================
RetinaCare AI - Intelligent Retinal Health Screening
Apple-Inspired Minimalist Healthcare & Multi-Model Deep Learning Interface
Engineered to recreate Apple macOS / iOS / iPadOS spatial design & materials
==============================================================================
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Suppress verbose TensorFlow C++ and oneDNN runtime output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Project root resolution & external libraries
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_PYLIB = Path("E:/pylib")
if _PYLIB.exists() and str(_PYLIB) not in sys.path:
    sys.path.insert(0, str(_PYLIB))

import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="RetinaCare AI — Intelligence for Retinal Health",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from utils.config import (
    BEST_MODEL_PATH,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_SHORT_NAMES,
    MEDICAL_DISCLAIMER
)
from utils.html_utils import render_html
from src.model_manager import (
    CANONICAL_MODEL_NAMES,
    get_model_status,
    get_all_models_status,
    load_model_instance,
    MissingModelError,
    ModelCompatibilityError
)
from src.class_mapping import (
    DATASET_NAME,
    ClassMappingVerificationError
)
from src.validation import (
    ImageValidationError,
    CorruptedImageError
)
from src.prediction import (
    run_model_inference,
    compare_all_available_models,
    compute_model_agreement,
    CONFIDENCE_THRESHOLD_PCT
)
from components.navigation import render_navigation, navigate_to
from components.hero import render_hero
from components.model_selector import render_model_selector
from components.upload_card import render_upload_card, render_preview_card
from components.prediction import (
    run_automated_analysis_pipeline,
    run_ai_analysis_pipeline,
    render_prediction_result,
    render_prediction_history,
    reset_analysis_state,
    render_empty_state
)
from components.comparison import render_model_comparison_section
from components.error_state import (
    render_context_aware_error,
    render_missing_model_error,
    render_invalid_image_error
)
from components.metrics import render_performance_metrics_grid
from components.charts import render_performance_charts, render_model_comparison_table
from components.about import render_about_section
from components.footer import render_footer


# ----------------------------------------------------------------------------
# CSS Design System Injection (Apple Theme & Motion Physics)
# ----------------------------------------------------------------------------
def inject_apple_styles():
    """Reads and injects Apple Design System CSS and animations."""
    theme_css_path = PROJECT_ROOT / "styles" / "theme.css"
    anim_css_path = PROJECT_ROOT / "styles" / "animations.css"

    combined_css = "<style>\n"
    if theme_css_path.is_file():
        combined_css += theme_css_path.read_text(encoding="utf-8") + "\n"
    if anim_css_path.is_file():
        combined_css += anim_css_path.read_text(encoding="utf-8") + "\n"
    combined_css += "</style>"

    st.markdown(combined_css, unsafe_allow_html=True)


inject_apple_styles()


# ----------------------------------------------------------------------------
# Cached Predictor Engine Helper
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def warm_model_cache(model_name: str):
    """Pre-loads and caches only the specified neural model checkpoint."""
    try:
        return load_model_instance(model_name)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Session State Initialization
# ----------------------------------------------------------------------------
if "current_nav" not in st.session_state:
    st.session_state["current_nav"] = "Home"
if "selected_model_choice" not in st.session_state:
    st.session_state["selected_model_choice"] = None
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "prediction_source_id" not in st.session_state:
    st.session_state["prediction_source_id"] = None
if "is_analyzing" not in st.session_state:
    st.session_state["is_analyzing"] = False
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []
if "comparison_results" not in st.session_state:
    st.session_state["comparison_results"] = None


# ----------------------------------------------------------------------------
# Apple Floating Navigation Bar
# ----------------------------------------------------------------------------
active_nav = render_navigation()


# ============================================================================
# SECTION 1: HOME
# ============================================================================
if active_nav == "Home":
    render_hero()


# ============================================================================
# SECTION 2: MULTI-MODEL PREDICTION & RETINAL SCREENING
# ============================================================================
elif active_nav == "Prediction":
    render_html(
        """
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-bottom: 28px;">
            <div class="apple-caption" style="color: var(--accent-blue); margin-bottom: 8px;">CLINICAL DECISION SUPPORT</div>
            <h2 class="apple-section-headline" style="text-align: center !important; margin: 0 0 8px 0;">Retinal Screening & Analysis</h2>
            <p class="apple-section-subhead" style="text-align: center !important; text-align-last: center !important; margin: 0 auto !important; max-width: 620px; display: block;">
                Ingest digital color fundus photography and compute automated multi-model deep learning inference with ranked alternative perspectives.
            </p>
        </div>
        """
    )

    col_input, col_stage = st.columns([1.08, 1.92], gap="large")

    with col_input:
        # STEP 1: Automated Architecture Readiness Telemetry (No manual selection needed)
        selected_model, model_status = render_model_selector()

        render_html("<div style='height: 16px;'></div>")

        # STEP 2: Retinal Fundus Image Source (Upload Area / Sample)
        image_source, source_label = render_upload_card()

        # STEP 25: Reset Control ("New Analysis")
        if image_source is not None or st.session_state.get("prediction_result") is not None:
            render_html("<div style='height: 12px;'></div>")
            if st.button("🔄 New Analysis", type="secondary", use_container_width=True, key="btn_new_analysis"):
                reset_analysis_state(keep_model=True)
                st.rerun()

    with col_stage:
        if image_source is not None:
            # Check if image source changed, smoothly reset prediction
            if st.session_state.get("prediction_source_id") != source_label:
                st.session_state["prediction_result"] = None
                st.session_state["primary_prediction"] = None
                st.session_state["alternative_predictions"] = None
                st.session_state["unranked_or_failed"] = None
                st.session_state["comparison_results"] = None
                st.session_state["prediction_source_id"] = source_label

            # STEP 3 & 4: Image Validation & High-Precision Preview
            render_preview_card(image_source, is_processing=st.session_state["is_analyzing"])

            render_html("<div style='height: 20px;'></div>")

            # STEP 5: Automated Analysis Action (Runs all available architectures)
            btn_spacer_l, btn_col, btn_spacer_r = st.columns([1.0, 1.8, 1.0])
            with btn_col:
                analyze_clicked = st.button(
                    "⚡ Analyze Retinal Scan (All Models)",
                    type="primary",
                    disabled=st.session_state["is_analyzing"],
                    use_container_width=True,
                    key="btn_analyze_action"
                )

            # Execute Automated Multi-Model Prediction Workflow
            if analyze_clicked and not st.session_state["is_analyzing"]:
                try:
                    st.session_state["is_analyzing"] = True
                    # Evaluates all available architectures, validates probabilities, selects highest-confidence
                    pred_output = run_automated_analysis_pipeline(image_source)
                    st.session_state["is_analyzing"] = False
                    st.rerun()
                except ImageValidationError as err:
                    st.session_state["is_analyzing"] = False
                    render_invalid_image_error(str(err))
                except Exception as err:
                    st.session_state["is_analyzing"] = False
                    render_context_aware_error(
                        error_type="generic",
                        title="Screening Pipeline Anomaly",
                        message=f"We couldn't complete the multi-model analysis: {str(err)}. Please try again."
                    )

            # STEP 6: Result Section (Primary result, animated alternatives, comparison table)
            if st.session_state.get("prediction_result") is not None:
                render_prediction_result(st.session_state["prediction_result"])

            # Section 24: Prediction History
            render_prediction_history()

        else:
            render_empty_state()
            render_prediction_history()


# ============================================================================
# SECTION 3: MODEL PERFORMANCE & BENCHMARKS
# ============================================================================
elif active_nav == "Performance":
    render_html(
        """
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-bottom: 32px;">
            <div class="apple-caption" style="color: var(--accent-blue); margin-bottom: 8px;">VALIDATION TELEMETRY</div>
            <h2 class="apple-section-headline" style="text-align: center !important; margin: 0 0 8px 0;">Model Performance</h2>
            <p class="apple-section-subhead" style="text-align: center !important; text-align-last: center !important; margin: 0 auto !important; max-width: 600px; display: block;">
                Empirical diagnostic metrics, training convergence history, and benchmark evaluations across neural backbones.
            </p>
        </div>
        """
    )

    # 4 Spacious Metric Cards
    render_performance_metrics_grid()

    render_html("<div style='height: 28px;'></div>")

    # Training Curves & Confusion Matrix
    render_performance_charts()

    # Benchmark Comparison Table
    render_model_comparison_table()


# ============================================================================
# SECTION 4: ABOUT ARCHITECTURE & CLINICAL ETHICS
# ============================================================================
elif active_nav == "About":
    render_about_section()


# Apple Healthcare Minimal Footer
render_footer()
