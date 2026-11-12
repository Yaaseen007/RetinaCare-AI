"""
RetinaCare AI - Hero Component
Cinematic, Minimalist Apple Hero with Staggered Entrance and Physical Action Triggers
"""

import streamlit as st
from utils.config import MEDICAL_DISCLAIMER
from utils.html_utils import render_html
from components.navigation import navigate_to


def render_hero():
    """Renders the Apple cinematic hero section and interactive feature blocks."""
    # Ambient Ethereal Glow / Hero Stage
    render_html(
        """
        <div style="text-align: center; padding: 48px 20px 24px 20px; position: relative;">
            <div style="position: absolute; top: 0%; left: 50%; transform: translateX(-50%); width: 600px; height: 320px; background: radial-gradient(ellipse at center, rgba(10, 132, 255, 0.12) 0%, rgba(10, 132, 255, 0.02) 60%, transparent 80%); pointer-events: none; z-index: 0;"></div>
            <div style="position: relative; z-index: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
                <div class="hero-animate-badge" style="margin-bottom: 20px;">
                    <span style="display: inline-flex; align-items: center; gap: 8px; padding: 6px 16px; border-radius: 9999px; background: var(--surface-card); border: 1px solid var(--border-hairline); backdrop-filter: var(--glass-blur); font-size: 0.76rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--accent-blue);">
                        <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--accent-blue);"></span>
                        DEEP LEARNING OPHTHALMOLOGY
                    </span>
                </div>
                <h1 class="apple-hero-headline hero-animate-heading" style="text-align: center !important; margin: 0 0 16px 0;">
                    Intelligence for<br>
                    <span class="apple-gradient-text">Retinal Health.</span>
                </h1>
                <p class="apple-hero-subhead hero-animate-subhead" style="text-align: center !important; text-align-last: center !important; margin: 0 auto 32px auto !important; max-width: 680px; display: block;">
                    Analyze retinal fundus images with a deep learning model designed for preliminary diabetic retinopathy screening.
                </p>
            </div>
        </div>
        """
    )

    # Primary and Secondary Action Buttons
    c_left, c_actions, c_right = st.columns([1.2, 1.6, 1.2])
    with c_actions:
        render_html('<div class="hero-animate-actions">')
        btn1, btn2 = st.columns(2, gap="small")
        with btn1:
            if st.button("Analyze Image", type="primary", use_container_width=True):
                navigate_to("Prediction")
        with btn2:
            if st.button("Explore Performance", type="secondary", use_container_width=True):
                navigate_to("Performance")
        render_html('</div>')

    render_html("<div style='height: 48px;'></div>")

    # Staggered 3 Feature Cards
    render_html('<div class="hero-animate-features">')
    fc1, fc2, fc3 = st.columns(3, gap="medium")

    with fc1:
        render_html(
            """
            <div class="apple-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div class="apple-caption" style="color: var(--accent-blue); margin-bottom: 12px;">01 / DEEP VISION</div>
                    <div class="apple-card-title">Transfer Learning Architecture</div>
                    <p class="apple-card-body" style="text-align: justify; text-justify: inter-word; text-align-last: left;">
                        Fine-tuned MobileNetV2 and deep convolutional backbones optimized to isolate microaneurysms and intraretinal microvascular abnormalities.
                    </p>
                </div>
                <div style="margin-top: 24px; font-size: 0.82rem; font-weight: 600; color: var(--accent-blue); display: flex; align-items: center; gap: 6px;">
                    Neural Architecture <span>→</span>
                </div>
            </div>
            """
        )

    with fc2:
        render_html(
            """
            <div class="apple-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div class="apple-caption" style="color: #64D2FF; margin-bottom: 12px;">02 / PREPROCESSING</div>
                    <div class="apple-card-title">Graham Local Contrast</div>
                    <p class="apple-card-body" style="text-align: justify; text-justify: inter-word; text-align-last: left;">
                        Eliminates lighting artifacts and peripheral vignetting via weighted Gaussian subtraction, accentuating subtle vascular pathology.
                    </p>
                </div>
                <div style="margin-top: 24px; font-size: 0.82rem; font-weight: 600; color: #64D2FF; display: flex; align-items: center; gap: 6px;">
                    Biomarker Enhancement <span>→</span>
                </div>
            </div>
            """
        )

    with fc3:
        render_html(
            """
            <div class="apple-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div class="apple-caption" style="color: var(--accent-emerald); margin-bottom: 12px;">03 / CLINICAL STANDARDS</div>
                    <div class="apple-card-title">5-Stage ICDR Grading</div>
                    <p class="apple-card-body" style="text-align: justify; text-justify: inter-word; text-align-last: left;">
                        Outputs a calibrated posterior probability distribution from Stage 0 (Healthy) through Stage 4 (Proliferative Diabetic Retinopathy).
                    </p>
                </div>
                <div style="margin-top: 24px; font-size: 0.82rem; font-weight: 600; color: var(--accent-emerald); display: flex; align-items: center; gap: 6px;">
                    Severity Staging <span>→</span>
                </div>
            </div>
            """
        )
    render_html('</div>')

    # Clinical Research Notice
    render_html("<div style='height: 32px;'></div>")
    render_html(
        f"""
        <div style="max-width: 860px; margin: 0 auto; background: var(--surface-card); border: 1px solid var(--border-hairline); border-radius: var(--radius-md); padding: 18px 26px; font-size: 0.85rem; color: var(--text-secondary); text-align: justify; text-justify: inter-word; text-align-last: left; line-height: 1.55;">
            <span style="color: var(--accent-blue); font-weight: 600;">Clinical Advisory:</span> {MEDICAL_DISCLAIMER}
        </div>
        """
    )
