"""
HTML Rendering Utilities for RetinaCare AI.
Prevents Streamlit CommonMark parsers from treating indented HTML as code blocks.
"""

from __future__ import annotations
import streamlit as st


def clean_html(html_str: str) -> str:
    """
    Strips leading and trailing whitespace from every line of an HTML string.
    Eliminates blank lines that can trigger CommonMark 4-space indented code blocks.
    """
    if not html_str:
        return ""
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())


def render_html(html_str: str):
    """
    Renders pure HTML safely via st.markdown without accidental <pre><code> code blocks.
    """
    cleaned = clean_html(html_str)
    if cleaned:
        st.markdown(cleaned, unsafe_allow_html=True)
