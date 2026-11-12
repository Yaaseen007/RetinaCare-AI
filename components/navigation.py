"""
RetinaCare AI - Navigation Component
Apple macOS / iOS Floating Segmented Navigation System
"""

import streamlit as st
from typing import List, Callable
from utils.html_utils import render_html

NAV_ITEMS = ["Home", "Prediction", "Performance", "About"]


def render_navigation() -> str:
    """
    Renders the floating frosted-glass navigation bar.
    Returns the currently active section name ('Home', 'Prediction', 'Performance', 'About').
    """
    if "current_nav" not in st.session_state:
        st.session_state["current_nav"] = "Home"
    if "nav_radio" not in st.session_state:
        st.session_state["nav_radio"] = st.session_state["current_nav"]
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "dark"

    # Floating Navigation Container
    render_html('<div class="apple-nav-container">')
    
    col_brand, col_nav, col_meta = st.columns([1.2, 2.6, 0.8], vertical_alignment="center")

    with col_brand:
        render_html(
            """
            <div class="apple-nav-brand-group">
                <div style="width: 28px; height: 28px; border-radius: 8px; background: linear-gradient(135deg, #0A84FF 0%, #0071E3 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(10, 132, 255, 0.4);">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="3"></circle>
                        <path d="M2.5 12c2.5-5 5.5-8 9.5-8s7 3 9.5 8c-2.5 5-5.5 8-9.5 8s-7-3-9.5-8z"></path>
                    </svg>
                </div>
                <div>
                    <span class="apple-nav-brand-title">RetinaCare</span>
                    <span class="apple-nav-pill-badge" style="margin-left: 6px;">AI</span>
                </div>
            </div>
            """
        )

    with col_nav:
        def on_nav_change():
            st.session_state["current_nav"] = st.session_state["nav_radio"]

        st.radio(
            label="Navigation Tabs",
            options=NAV_ITEMS,
            key="nav_radio",
            horizontal=True,
            label_visibility="collapsed",
            on_change=on_nav_change
        )

    with col_meta:
        # Subtle status indicator
        render_html(
            """
            <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
                <span class="pulse-indicator" style="background: #30D158; box-shadow: 0 0 8px #30D158;"></span>
                <span style="font-size: 0.78rem; font-weight: 500; color: var(--text-secondary); letter-spacing: -0.01em;">Online</span>
            </div>
            """
        )

    render_html('</div>')
    
    return st.session_state["current_nav"]


def navigate_to(section_name: str):
    """Programmatically switches the active navigation tab with smooth state update."""
    if section_name in NAV_ITEMS:
        st.session_state["current_nav"] = section_name
        st.session_state["nav_radio"] = section_name
        st.rerun()
