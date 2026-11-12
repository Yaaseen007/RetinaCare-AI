"""
RetinaCare AI - About Component
Apple-Inspired System Architecture, Problem Formulation, and Clinical Ethics Grid
"""

import streamlit as st
from utils.config import MEDICAL_DISCLAIMER
from utils.html_utils import render_html


def render_about_section():
    """
    Renders the About section with symmetrical Apple cards, generous whitespace,
    and precise typography.
    """
    render_html(
        """
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-bottom: 36px;">
            <div class="apple-caption" style="color: var(--accent-blue); margin-bottom: 8px;">RESEARCH ARCHITECTURE</div>
            <h2 class="apple-section-headline" style="text-align: center !important; margin: 0 0 8px 0;">About RetinaCare AI</h2>
            <p class="apple-section-subhead" style="text-align: center !important; text-align-last: center !important; margin: 0 auto !important; max-width: 600px; display: block;">
                A deep learning decision support framework engineered for high-throughput preliminary screening of diabetic microvascular eye disease.
            </p>
        </div>
        """
    )

    # Row 1: Problem Statement & Clinical Workflow
    r1_col1, r1_col2 = st.columns(2, gap="medium")
    with r1_col1:
        render_html(
            """
            <div class="apple-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div class="apple-caption" style="color: var(--accent-blue); margin-bottom: 8px;">EPIDEMIOLOGY</div>
                    <div class="apple-card-title">Problem Statement</div>
                    <p class="apple-card-body" style="text-align: justify; text-justify: inter-word; text-align-last: left;">
                        Diabetic Retinopathy (DR) is the primary driver of preventable blindness among working-age adults globally. In its early microvascular stages, diabetic retinopathy progresses asymptomatically. Over 90% of severe vision loss cases can be prevented with regular, automated fundus screening and timely clinical referral.
                    </p>
                </div>
                <div style="margin-top: 20px; font-size: 0.82rem; font-weight: 600; color: var(--accent-blue);">
                    Global Public Health Challenge
                </div>
            </div>
            """
        )

    with r1_col2:
        render_html(
            """
            <div class="apple-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div class="apple-caption" style="color: #64D2FF; margin-bottom: 8px;">ENGINEERING PIPELINE</div>
                    <div class="apple-card-title">End-to-End Clinical Workflow</div>
                    <div class="apple-card-body" style="line-height: 1.65; text-align: justify; text-justify: inter-word; text-align-last: left;">
                        <strong>1. Ingestion:</strong> Standard RGB fundus photograph acquisition.<br>
                        <strong>2. Masking:</strong> Circular aperture cropping to isolate retinal field.<br>
                        <strong>3. Normalization:</strong> Ben Graham weighted Gaussian subtraction.<br>
                        <strong>4. Extraction:</strong> Deep convolutional transfer representations.<br>
                        <strong>5. Assessment:</strong> Calibrated 5-stage ICDR clinical classification.
                    </div>
                </div>
                <div style="margin-top: 20px; font-size: 0.82rem; font-weight: 600; color: #64D2FF;">
                    Standardized Processing Pipeline
                </div>
            </div>
            """
        )

    render_html("<div style='height: 20px;'></div>")

    # Row 2: Objectives & Dataset / Ethics
    r2_col1, r2_col2 = st.columns(2, gap="medium")
    with r2_col1:
        render_html(
            """
            <div class="apple-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div class="apple-caption" style="color: var(--accent-emerald); margin-bottom: 8px;">OBJECTIVES</div>
                    <div class="apple-card-title">System Objectives</div>
                    <div class="apple-card-body" style="line-height: 1.65; text-align: justify; text-justify: inter-word; text-align-last: left;">
                        • Eliminate illumination variance across heterogeneous clinic cameras.<br>
                        • Address clinical class imbalance via balanced focal and cross-entropy weighting.<br>
                        • Provide transparent probability distributions for all 5 disease stages.<br>
                        • Guarantee complete privacy with strict in-memory execution.
                    </div>
                </div>
                <div style="margin-top: 20px; font-size: 0.82rem; font-weight: 600; color: var(--accent-emerald);">
                    Academic Engineering Milestones
                </div>
            </div>
            """
        )

    with r2_col2:
        render_html(
            """
            <div class="apple-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div class="apple-caption" style="color: var(--accent-amber); margin-bottom: 8px;">DATASET & ETHICS</div>
                    <div class="apple-card-title">Training Cohorts & Zero-Retention</div>
                    <p class="apple-card-body" style="text-align: justify; text-justify: inter-word; text-align-last: left;">
                        Trained on benchmark retinal repositories (APTOS 2019 Blindness Detection, EyePACS, and IDRiD) with verified expert ophthalmologist annotations.<br><br>
                        <strong>Privacy-by-Design:</strong> Zero persistent storage of patient photographs. Every uploaded tensor is processed in volatile RAM and immediately discarded.
                    </p>
                </div>
                <div style="margin-top: 20px; font-size: 0.82rem; font-weight: 600; color: var(--accent-amber);">
                    HIPAA / GDPR Aligned Architecture
                </div>
            </div>
            """
        )

    render_html("<div style='height: 20px;'></div>")

    # Row 3: Technology Stack & Clinical Disclaimer
    r3_col1, r3_col2 = st.columns(2, gap="medium")
    with r3_col1:
        render_html(
            """
            <div class="apple-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div class="apple-caption" style="color: var(--accent-blue); margin-bottom: 8px;">INFRASTRUCTURE</div>
                    <div class="apple-card-title">Technology Stack</div>
                    <div class="apple-card-body" style="line-height: 1.65; text-align: justify; text-justify: inter-word; text-align-last: left;">
                        <strong>Deep Learning:</strong> Python 3.12, TensorFlow 2.15, Keras, Scikit-Learn<br>
                        <strong>Computer Vision:</strong> OpenCV, Pillow (PIL), NumPy<br>
                        <strong>Visualizations:</strong> Plotly, Matplotlib, Seaborn<br>
                        <strong>Web Presentation:</strong> Streamlit with Apple spatial design system
                    </div>
                </div>
                <div style="margin-top: 20px; font-size: 0.82rem; font-weight: 600; color: var(--accent-blue);">
                    Modern Python & Deep Learning Stack
                </div>
            </div>
            """
        )

    with r3_col2:
        render_html(
            f"""
            <div class="apple-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between; border-color: rgba(10, 132, 255, 0.3);">
                <div>
                    <div class="apple-caption" style="color: var(--accent-blue); margin-bottom: 8px;">REGULATORY COMPLIANCE</div>
                    <div class="apple-card-title">Clinical Disclaimer</div>
                    <p class="apple-card-body" style="text-align: justify; text-justify: inter-word; text-align-last: left;">
                        {MEDICAL_DISCLAIMER}
                    </p>
                </div>
                <div style="margin-top: 20px; font-size: 0.82rem; font-weight: 600; color: var(--accent-blue);">
                    Academic Research & Educational Scope
                </div>
            </div>
            """
        )
