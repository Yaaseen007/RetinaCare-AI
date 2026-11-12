"""
RetinaCare AI - Footer Component
Apple-Inspired Minimalist Healthcare AI Footer & Compliance Badges
"""

import streamlit as st
from utils.html_utils import render_html


def render_footer():
    """Renders the Apple minimal footer with academic and compliance metadata."""
    render_html("<div style='height: 64px;'></div>")
    render_html(
        """
        <div style="border-top: 1px solid var(--border-hairline); padding-top: 32px; padding-bottom: 32px; text-align: center;">
            <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px; max-width: var(--content-max-width); margin: 0 auto;">
                <div style="text-align: left;">
                    <div style="font-size: 0.95rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em;">
                        RetinaCare AI
                    </div>
                    <div style="font-size: 0.80rem; color: var(--text-secondary); margin-top: 2px;">
                        Deep Learning for Diabetic Retinopathy Screening • Department of CSE, GM University
                    </div>
                </div>

                <div style="display: flex; align-items: center; gap: 20px; font-size: 0.82rem; color: var(--text-tertiary);">
                    <span>Zero-Retention Privacy</span>
                    <span>•</span>
                    <span>ICDR 5-Stage Standard</span>
                    <span>•</span>
                    <span>Academic Research Only</span>
                </div>
            </div>

            <div style="margin: 20px auto 0 auto !important; font-size: 0.74rem; color: var(--text-quaternary); max-width: 780px; line-height: 1.5; text-align: center !important; text-align-last: center !important; display: block;">
                This system is an academic research prototype engineered for preliminary screening education. It is not an FDA/CE-cleared diagnostic device. Always consult a licensed ophthalmologist for comprehensive dilated eye examination.
            </div>
        </div>
        """
    )
