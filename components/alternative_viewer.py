"""
==============================================================================
RetinaCare AI - Animated Alternative Predictions Viewer
Apple-Inspired Spatial Carousel, Micro-Transitions, Keyboard & Auto-Rotation
==============================================================================
"""

from __future__ import annotations
import json
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Any, List, Optional
from utils.html_utils import render_html


def render_alternative_predictions_viewer(
    alternative_results: List[Dict[str, Any]],
    primary_result: Optional[Dict[str, Any]] = None,
    unranked_or_failed: Optional[List[Dict[str, Any]]] = None
):
    """
    Renders the polished animated alternative-predictions viewer directly
    beneath the primary result card.

    Features:
    - Smooth 60fps directional slide and fade transitions (Left/Right)
    - Prev / Next controls and dot indicators
    - Compact model tabs for instant navigation
    - Optional auto-rotation mode (disabled by default, 4s interval, pause on hover)
    - Full keyboard accessibility (Arrow Left/Right, visible focus rings)
    - Displays predicted class, confidence delta vs primary, and full 5-class distribution
    - Respects prefers-reduced-motion
    - Fully isolated: switching alternatives NEVER modifies the primary result card
    """
    if not alternative_results:
        # If only 1 model is available or trained
        render_html(
            """
            <div class="apple-card" style="margin-top: 20px; padding: 18px 22px; text-align: center; border-color: var(--border-hairline);">
                <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
                    Alternative Architectures
                </div>
                <div style="font-size: 0.90rem; color: var(--text-secondary);">
                    No additional trained models available for comparison. Train other backbones to evaluate multiple perspectives.
                </div>
            </div>
            """
        )
        return

    # Prepare JSON payload for the client-side viewer
    clean_alternatives = []
    primary_class_id = primary_result.get("class_id") if primary_result else None
    primary_conf = primary_result.get("confidence_pct", 0.0) if primary_result else 0.0

    severity_colors = ["#30D158", "#64D2FF", "#FF9F0A", "#FF851B", "#FF453A"]

    for alt in alternative_results:
        cid = alt.get("class_id", 0)
        c_pct = alt.get("confidence_pct", 0.0)
        delta = round(c_pct - primary_conf, 2)
        agrees = (primary_class_id is not None and cid == primary_class_id)

        clean_alternatives.append({
            "model_name": alt.get("model_name", "Alternative Model"),
            "architecture_type": alt.get("architecture_type", "Convolutional Network"),
            "backbone": alt.get("backbone", ""),
            "class_id": cid,
            "class_name": alt.get("class_name", alt.get("prediction", "Unknown")),
            "short_name": alt.get("short_name", f"Stage {cid}"),
            "confidence_pct": round(c_pct, 2),
            "confidence_delta": delta,
            "agrees_with_primary": agrees,
            "severity_color": severity_colors[cid] if 0 <= cid < len(severity_colors) else "#0A84FF",
            "latency_ms": alt.get("latency_ms", 0.0),
            "probabilities": alt.get("probabilities", {}),
            "description": alt.get("description", ""),
            "recommendation": alt.get("recommendation", "")
        })

    alts_json = json.dumps(clean_alternatives).replace("</", "<\\/")

    # Unavailable or failed architectures summary markup
    unranked_markup = ""
    if unranked_or_failed:
        badges = []
        for uf in unranked_or_failed:
            m_name = uf.get("model_name", "Unknown")
            status = uf.get("status", "Not Available")
            msg = uf.get("error_msg", "Not trained")
            if status == "Not Available":
                badges.append(f"<span style='display:inline-flex;align-items:center;gap:4px;padding:3px 8px;border-radius:12px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);font-size:0.72rem;color:var(--text-tertiary);'>○ {m_name} (Not Trained)</span>")
            else:
                badges.append(f"<span style='display:inline-flex;align-items:center;gap:4px;padding:3px 8px;border-radius:12px;background:rgba(255,69,58,0.1);border:1px solid rgba(255,69,58,0.3);font-size:0.72rem;color:#FF453A;'>✕ {m_name} (Error)</span>")
        
        if badges:
            unranked_markup = f"""
            <div style="margin-top: 14px; padding: 10px 16px; background: rgba(255,255,255,0.02); border-radius: var(--radius-sm); border: 1px solid var(--border-hairline); display: flex; flex-wrap: wrap; align-items: center; gap: 8px;">
                <span style="font-size: 0.72rem; font-weight: 700; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.05em;">Other Architectures:</span>
                {' '.join(badges)}
            </div>
            """

    # Section header
    render_html(
        f"""
        <div style="margin-top: 32px; padding-top: 20px; border-top: 1px solid var(--border-hairline);">
            <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px;">
                <div>
                    <div class="apple-caption" style="color: var(--accent-blue); margin-bottom: 4px;">COMPARATIVE MULTI-MODEL PERSPECTIVES</div>
                    <h3 style="font-size: 1.25rem; font-weight: 700; color: var(--text-primary); margin: 0 0 4px 0;">
                        Alternative Model Predictions
                    </h3>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">
                        Ranked by descending softmax confidence. Browse other architectures to compare visual lesion interpretations.
                    </div>
                </div>
            </div>
        </div>
        """
    )

    # Self-contained, accessible, smooth HTML5/CSS3/ES6 component
    carousel_component_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {{
            --bg-card: rgba(28, 28, 30, 0.75);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-hairline: rgba(255, 255, 255, 0.05);
            --accent-blue: #0A84FF;
            --accent-emerald: #30D158;
            --accent-amber: #FF9F0A;
            --accent-red: #FF453A;
            --text-primary: #FFFFFF;
            --text-secondary: rgba(255, 255, 255, 0.72);
            --text-tertiary: rgba(255, 255, 255, 0.44);
            --radius-md: 14px;
            --radius-sm: 10px;
            --radius-pill: 9999px;
            --font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: var(--font-family);
        }}

        body {{
            background: transparent;
            color: var(--text-primary);
            overflow: hidden;
            user-select: none;
            -webkit-font-smoothing: antialiased;
        }}

        .alt-container {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 18px 22px;
            box-shadow: 0 8px 28px -4px rgba(0, 0, 0, 0.45);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            display: flex;
            flex-direction: column;
            gap: 16px;
            max-width: 100%;
        }}

        /* Navigation Header & Controls */
        .alt-header {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--border-hairline);
        }}

        .model-tabs {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 6px;
        }}

        .model-tab-btn {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-pill);
            color: var(--text-secondary);
            font-size: 0.76rem;
            font-weight: 600;
            padding: 5px 12px;
            cursor: pointer;
            transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
            outline: none;
        }}

        .model-tab-btn:hover {{
            background: rgba(255, 255, 255, 0.10);
            color: var(--text-primary);
            border-color: rgba(255, 255, 255, 0.18);
        }}

        .model-tab-btn.active {{
            background: rgba(10, 132, 255, 0.16);
            border-color: rgba(10, 132, 255, 0.45);
            color: var(--accent-blue);
            box-shadow: 0 0 12px rgba(10, 132, 255, 0.22);
        }}

        .model-tab-btn:focus-visible {{
            box-shadow: 0 0 0 2px var(--accent-blue);
        }}

        .controls-cluster {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        /* Auto-rotate Toggle */
        .auto-rotate-box {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.74rem;
            color: var(--text-tertiary);
            cursor: pointer;
            padding: 4px 10px;
            border-radius: var(--radius-pill);
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-hairline);
            transition: all 0.2s ease;
        }}
        .auto-rotate-box:hover {{
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.06);
        }}
        .auto-rotate-box input {{
            accent-color: var(--accent-blue);
            cursor: pointer;
            width: 13px;
            height: 13px;
        }}

        .nav-buttons {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: rgba(255, 255, 255, 0.04);
            border-radius: var(--radius-pill);
            padding: 2px 4px;
            border: 1px solid var(--border-hairline);
        }}

        .nav-btn {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            width: 28px;
            height: 28px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            line-height: 1;
            transition: all 0.18s ease;
            outline: none;
        }}

        .nav-btn:hover {{
            background: rgba(255, 255, 255, 0.12);
            color: var(--text-primary);
        }}

        .nav-btn:active {{
            transform: scale(0.92);
        }}

        .nav-btn:focus-visible {{
            box-shadow: 0 0 0 2px var(--accent-blue);
        }}

        .pos-indicator {{
            font-size: 0.74rem;
            color: var(--text-tertiary);
            font-variant-numeric: tabular-nums;
            padding: 0 4px;
        }}

        /* Card Content Viewport */
        .card-viewport {{
            position: relative;
            min-height: 330px;
            overflow: hidden;
            width: 100%;
        }}

        .alt-card {{
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            gap: 14px;
            opacity: 0;
            pointer-events: none;
            transform: translateX(40px);
            transition: transform 0.36s cubic-bezier(0.16, 1, 0.3, 1),
                        opacity 0.36s cubic-bezier(0.16, 1, 0.3, 1);
            will-change: transform, opacity;
        }}

        .alt-card.active {{
            position: relative;
            opacity: 1;
            pointer-events: auto;
            transform: translateX(0);
        }}

        .alt-card.slide-left-out {{
            opacity: 0;
            transform: translateX(-40px);
        }}

        .alt-card.slide-right-out {{
            opacity: 0;
            transform: translateX(40px);
        }}

        /* Card Elements */
        .card-top {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }}

        .stage-pill {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 4px 12px;
            border-radius: var(--radius-pill);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .stage-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
        }}

        .agreement-badge {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: var(--radius-pill);
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-hairline);
        }}

        .hero-row {{
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            justify-content: space-between;
            gap: 10px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-hairline);
        }}

        .model-title {{
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }}

        .model-subtitle {{
            font-size: 0.82rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }}

        .conf-cluster {{
            text-align: right;
        }}

        .conf-number {{
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1;
            font-variant-numeric: tabular-nums;
        }}

        .conf-delta {{
            font-size: 0.72rem;
            font-weight: 600;
            margin-top: 3px;
        }}

        /* Probability Bars */
        .bars-container {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .bar-row {{
            display: grid;
            grid-template-columns: 140px 1fr 50px;
            align-items: center;
            gap: 10px;
            font-size: 0.78rem;
        }}

        .bar-label {{
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .bar-label.active {{
            color: var(--text-primary);
            font-weight: 600;
        }}

        .bar-track {{
            height: 6px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: var(--radius-pill);
            overflow: hidden;
            position: relative;
        }}

        .bar-fill {{
            height: 100%;
            border-radius: var(--radius-pill);
            transition: width 0.5s cubic-bezier(0.16, 1, 0.3, 1);
            will-change: width;
        }}

        .bar-val {{
            text-align: right;
            font-variant-numeric: tabular-nums;
            color: var(--text-tertiary);
            font-size: 0.74rem;
        }}

        .bar-val.active {{
            color: var(--text-primary);
            font-weight: 700;
        }}

        /* Footer Telemetry */
        .card-footer {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--border-hairline);
            font-size: 0.72rem;
            color: var(--text-tertiary);
        }}

        /* Reduced Motion Accessibility */
        @media (prefers-reduced-motion: reduce) {{
            .alt-card {{
                transition: opacity 0.01ms !important;
                transform: none !important;
            }}
            .bar-fill {{
                transition: none !important;
            }}
        }}
    </style>
    </head>
    <body>
    <div class="alt-container" id="carouselWrapper">
        <!-- Navigation Header -->
        <div class="alt-header">
            <div class="model-tabs" id="modelTabs" role="tablist" aria-label="Alternative Models"></div>
            
            <div class="controls-cluster">
                <label class="auto-rotate-box" title="Auto-advance through alternative models every 4 seconds (pauses on hover)">
                    <input type="checkbox" id="autoRotateToggle" aria-label="Toggle Auto-Rotation">
                    <span>⏱️ Auto-rotate</span>
                </label>
                
                <div class="nav-buttons">
                    <button class="nav-btn" id="btnPrev" aria-label="Previous Alternative Model" title="Previous model (Left Arrow)">‹</button>
                    <span class="pos-indicator" id="posIndicator">1 / 1</span>
                    <button class="nav-btn" id="btnNext" aria-label="Next Alternative Model" title="Next model (Right Arrow)">›</button>
                </div>
            </div>
        </div>

        <!-- Viewport -->
        <div class="card-viewport" id="cardViewport"></div>
    </div>

    <script>
    (function() {{
        const models = {alts_json};
        if (!models || models.length === 0) return;

        let currentIndex = 0;
        let autoRotateTimer = null;
        let isHovered = false;

        const tabsContainer = document.getElementById('modelTabs');
        const viewport = document.getElementById('cardViewport');
        const posIndicator = document.getElementById('posIndicator');
        const btnPrev = document.getElementById('btnPrev');
        const btnNext = document.getElementById('btnNext');
        const autoRotateToggle = document.getElementById('autoRotateToggle');
        const wrapper = document.getElementById('carouselWrapper');

        const classNames = [
            "No DR (Stage 0)",
            "Mild DR (Stage 1)",
            "Moderate DR (Stage 2)",
            "Severe DR (Stage 3)",
            "Proliferative DR (Stage 4)"
        ];

        // Build Tabs & Cards DOM
        models.forEach((m, idx) => {{
            // Tab button
            const tabBtn = document.createElement('button');
            tabBtn.className = 'model-tab-btn' + (idx === 0 ? ' active' : '');
            tabBtn.textContent = `${{m.model_name}} (${{m.confidence_pct}}%)`;
            tabBtn.setAttribute('role', 'tab');
            tabBtn.setAttribute('aria-selected', idx === 0 ? 'true' : 'false');
            tabBtn.setAttribute('aria-controls', 'card-' + idx);
            tabBtn.id = 'tab-' + idx;
            tabBtn.addEventListener('click', () => goToIndex(idx));
            tabsContainer.appendChild(tabBtn);

            // Card
            const card = document.createElement('div');
            card.className = 'alt-card' + (idx === 0 ? ' active' : '');
            card.id = 'card-' + idx;
            card.setAttribute('role', 'tabpanel');
            card.setAttribute('aria-labelledby', 'tab-' + idx);

            const agreementTag = m.agrees_with_primary
                ? `<span class="agreement-badge" style="color: var(--accent-emerald); border-color: rgba(48,209,88,0.3);">● Agrees with Primary (Stage ${{m.class_id}})</span>`
                : `<span class="agreement-badge" style="color: var(--accent-amber); border-color: rgba(255,159,10,0.3);">▲ Differs from Primary (Stage ${{m.class_id}})</span>`;

            const deltaColor = m.confidence_delta >= 0 ? 'var(--accent-emerald)' : 'var(--text-tertiary)';
            const deltaSign = m.confidence_delta > 0 ? '+' : '';

            // Probability bars HTML
            let barsHtml = '';
            for (let i = 0; i < 5; i++) {{
                const rawProb = m.probabilities ? (m.probabilities[classNames[i]] || m.probabilities[Object.keys(m.probabilities)[i]] || 0) : 0;
                const probVal = Math.round(rawProb * 10) / 10;
                const isActive = (i === m.class_id);
                const barColor = isActive ? m.severity_color : 'rgba(255,255,255,0.18)';
                barsHtml += `
                    <div class="bar-row">
                        <span class="bar-label ${{isActive ? 'active' : ''}}">${{classNames[i]}}</span>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: ${{probVal}}%; background: ${{barColor}};"></div>
                        </div>
                        <span class="bar-val ${{isActive ? 'active' : ''}}">${{probVal}}%</span>
                    </div>
                `;
            }}

            card.innerHTML = `
                <div class="card-top">
                    <div class="stage-pill" style="background: ${{m.severity_color}}18; border: 1px solid ${{m.severity_color}}45; color: ${{m.severity_color}};">
                        <span class="stage-dot" style="background: ${{m.severity_color}};"></span>
                        Stage ${{m.class_id}} / 4 — ${{m.short_name}}
                    </div>
                    ${{agreementTag}}
                </div>

                <div class="hero-row">
                    <div>
                        <div class="model-title">${{m.model_name}}</div>
                        <div class="model-subtitle">${{m.architecture_type}}</div>
                    </div>
                    <div class="conf-cluster">
                        <div class="conf-number" style="color: ${{m.severity_color}};">${{m.confidence_pct}}%</div>
                        <div class="conf-delta" style="color: ${{deltaColor}};">${{deltaSign}}${{m.confidence_delta}}% vs Primary</div>
                    </div>
                </div>

                <div class="bars-container">
                    ${{barsHtml}}
                </div>

                <div class="card-footer">
                    <span>Forward Latency: ~${{m.latency_ms}} ms</span>
                    <span>Evaluation Metric: Normalized Softmax Output</span>
                </div>
            `;

            viewport.appendChild(card);
        }});

        function updateUI(direction = 1) {{
            const cards = viewport.querySelectorAll('.alt-card');
            const tabs = tabsContainer.querySelectorAll('.model-tab-btn');

            cards.forEach((c, idx) => {{
                if (idx === currentIndex) {{
                    c.className = 'alt-card active';
                }} else if (idx < currentIndex) {{
                    c.className = 'alt-card slide-left-out';
                }} else {{
                    c.className = 'alt-card slide-right-out';
                }}
            }});

            tabs.forEach((t, idx) => {{
                t.className = 'model-tab-btn' + (idx === currentIndex ? ' active' : '');
                t.setAttribute('aria-selected', idx === currentIndex ? 'true' : 'false');
            }});

            posIndicator.textContent = `${{currentIndex + 1}} / ${{models.length}}`;
        }}

        function goToIndex(newIdx) {{
            const dir = newIdx >= currentIndex ? 1 : -1;
            currentIndex = (newIdx + models.length) % models.length;
            updateUI(dir);
        }}

        function nextModel() {{
            goToIndex(currentIndex + 1);
        }}

        function prevModel() {{
            goToIndex(currentIndex - 1);
        }}

        // Auto-rotation logic
        function restartAutoRotate() {{
            if (autoRotateTimer) clearInterval(autoRotateTimer);
            if (autoRotateToggle.checked && !isHovered) {{
                autoRotateTimer = setInterval(() => {{
                    nextModel();
                }}, 4000);
            }}
        }}

        btnNext.addEventListener('click', () => {{
            nextModel();
            restartAutoRotate();
        }});

        btnPrev.addEventListener('click', () => {{
            prevModel();
            restartAutoRotate();
        }});

        autoRotateToggle.addEventListener('change', () => {{
            restartAutoRotate();
        }});

        wrapper.addEventListener('mouseenter', () => {{
            isHovered = true;
            if (autoRotateTimer) clearInterval(autoRotateTimer);
        }});

        wrapper.addEventListener('mouseleave', () => {{
            isHovered = false;
            restartAutoRotate();
        }});

        // Keyboard navigation
        window.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight') {{
                nextModel();
                restartAutoRotate();
            }} else if (e.key === 'ArrowLeft') {{
                prevModel();
                restartAutoRotate();
            }}
        }});

        // Initialize UI
        updateUI(1);
    }})();
    </script>
    </body>
    </html>
    """

    # Embed with perfect calculated height without iframe scrollbars
    components.html(carousel_component_html, height=440, scrolling=False)

    if unranked_markup:
        render_html(unranked_markup)
