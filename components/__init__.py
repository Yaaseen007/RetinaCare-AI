"""
RetinaCare AI - Apple Design System Components Package
"""

from components.navigation import render_navigation, navigate_to
from components.hero import render_hero
from components.model_selector import render_model_selector
from components.upload_card import render_upload_card, render_preview_card
from components.prediction import (
    run_ai_analysis_pipeline,
    render_prediction_result,
    render_prediction_history,
    reset_analysis_state,
    render_empty_state,
    render_error_state
)
from components.result_card import render_prediction_result_card
from components.comparison import render_model_comparison_section
from components.probability_chart import render_probability_bars
from components.error_state import (
    render_context_aware_error,
    render_missing_model_error,
    render_invalid_image_error
)
from components.metrics import render_performance_metrics_grid
from components.charts import render_performance_charts, render_model_comparison_table
from components.about import render_about_section
from components.footer import render_footer

__all__ = [
    "render_navigation",
    "navigate_to",
    "render_hero",
    "render_model_selector",
    "render_upload_card",
    "render_preview_card",
    "run_ai_analysis_pipeline",
    "render_prediction_result",
    "render_prediction_history",
    "reset_analysis_state",
    "render_empty_state",
    "render_error_state",
    "render_prediction_result_card",
    "render_model_comparison_section",
    "render_probability_bars",
    "render_context_aware_error",
    "render_missing_model_error",
    "render_invalid_image_error",
    "render_performance_metrics_grid",
    "render_performance_charts",
    "render_model_comparison_table",
    "render_about_section",
    "render_footer"
]
