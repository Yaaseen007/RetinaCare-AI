/**
 * RetinaCare AI - Website Retinal Screening & VGG16 Prediction Pipeline
 * Uses VGG16 exclusively for high-capacity clinical screening on the web platform.
 * MobileNetV2 is reserved exclusively for the Flutter mobile application.
 */

import { WEB_MODEL, CLASS_NAMES, CLASS_SHORT_NAMES, SEVERITY_COLORS, CLINICAL_STAGES, CONFIDENCE_THRESHOLD_PCT, MEDICAL_DISCLAIMER } from "./config.js";
import { predictWeb, APIError } from "./api.js";
import { uploadManager } from "./upload.js";

class PredictionManager {
    constructor() {
        this.isAnalyzing = false;
        this.primaryResult = null;
        this.alternativeResults = [];
        this.activeAlternativeIndex = 0;
        this.predictionHistory = [];
        this.onPredictionCompletedCallbacks = [];
    }

    init() {
        const analyzeBtn = document.getElementById("btnAnalyzeRetina");
        if (analyzeBtn) {
            analyzeBtn.addEventListener("click", () => {
                this.runAutomatedAnalysis();
            });
        }

        // Listen for image reset to clear results
        uploadManager.onImageReset(() => {
            this.clearResults();
        });

        uploadManager.onImageSelected(() => {
            // Smoothly clear previous analysis if new image is picked
            if (this.primaryResult) {
                this.clearResults();
            }
        });
    }

    onPredictionCompleted(callback) {
        this.onPredictionCompletedCallbacks.push(callback);
    }

    async runAutomatedAnalysis() {
        if (this.isAnalyzing) return;

        const blob = uploadManager.getCurrentBlob();
        if (!blob) {
            uploadManager.showToast("Please select a reference retinal image or upload a custom fundus photograph before analyzing.", "info");
            return;
        }

        const filename = uploadManager.getCurrentFilename();
        this.setAnalyzingState(true);

        try {
            // Execute dedicated website inference via FastAPI /api/predict/web (VGG16 exclusively)
            const responseData = await predictWeb(blob, filename);
            this.processWebPrediction(responseData);
        } catch (err) {
            this.handleAnalysisError(err);
        } finally {
            this.setAnalyzingState(false);
        }
    }

    setAnalyzingState(isAnalyzing) {
        this.isAnalyzing = isAnalyzing;
        const btn = document.getElementById("btnAnalyzeRetina");
        const statusBanner = document.getElementById("analysisStatusBanner");
        const skeletonCard = document.getElementById("analysisSkeletonCard");
        const resultStage = document.getElementById("predictionResultStage");
        const errorCard = document.getElementById("analysisErrorCard");

        if (isAnalyzing) {
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = `<span class="pulse-dot" style="background:#ffffff;margin-right:6px;"></span> Evaluating Retinal Features (VGG16)...`;
            }
            if (statusBanner) statusBanner.style.display = "flex";
            if (skeletonCard) skeletonCard.style.display = "block";
            if (resultStage) resultStage.style.display = "none";
            if (errorCard) errorCard.style.display = "none";
            uploadManager.startScanning();
        } else {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = `⚡ Analyze Retinal Scan (VGG16)`;
            }
            if (statusBanner) statusBanner.style.display = "none";
            if (skeletonCard) skeletonCard.style.display = "none";
            uploadManager.stopScanning();
        }
    }

    processWebPrediction(data) {
        if (!data || data.confidence_pct == null) {
            this.handleAnalysisError(new Error("No valid prediction returned from VGG16 web screening engine."));
            return;
        }

        this.allSuccessfulResults = [data];
        this.championModelName = data.model_name || "VGG16";
        this.activeModel = data;
        this.primaryResult = data;
        this.alternativeResults = [];
        this.activeAlternativeIndex = 0;

        // Render result views
        this.renderPrimaryResultCard(this.activeModel);
        this.renderDedicatedWebModelStrip(this.activeModel);
        this.renderDedicatedWebArchitectureNotice(this.activeModel);
        this.addToHistory(this.activeModel);

        const resultStage = document.getElementById("predictionResultStage");
        if (resultStage) {
            resultStage.style.display = "block";
            resultStage.classList.remove("apple-result-card");
            void resultStage.offsetWidth; // Trigger reflow for smooth animation
            resultStage.classList.add("apple-result-card");
            
            // Scroll smoothly into view
            resultStage.scrollIntoView({ behavior: "smooth", block: "start" });
        }

        // Notify subscribers (such as chatbot for context awareness)
        this.onPredictionCompletedCallbacks.forEach((cb) => cb(this.activeModel, []));
    }

    renderDedicatedWebModelStrip(pred) {
        let switcher = document.getElementById("modelSwitcherStrip");
        if (!switcher) {
            switcher = document.createElement("div");
            switcher.id = "modelSwitcherStrip";
            switcher.className = "model-switcher-strip";
            const stage = document.getElementById("predictionResultStage");
            const header = stage ? stage.querySelector(".result-card-header") : null;
            if (header && header.parentNode) {
                header.parentNode.insertBefore(switcher, header.nextSibling);
            }
        }
        if (!switcher) return;
        switcher.style.display = "flex";

        const conf = Number(pred.confidence_pct || 0).toFixed(1);
        switcher.innerHTML = `
            <div class="model-switcher-label">
                <span>Architecture Routing:</span>
            </div>
            <div class="model-switcher-pills">
                <span class="model-switcher-btn active" style="cursor: default;">
                    <span class="star-badge">★ Website Production Model</span>
                    <span class="switcher-name">VGG16 Deep CNN</span>
                    <span class="switcher-conf">${conf}% Confidence</span>
                </span>
                <span class="model-switcher-btn" style="cursor: default; opacity: 0.8; background: rgba(255,255,255,0.02); border-color: rgba(255,255,255,0.08);">
                    <span class="switcher-name">📱 MobileNetV2</span>
                    <span style="font-size:0.72rem; color: var(--text-tertiary); margin-left: 4px;">Reserved for Flutter Mobile App</span>
                </span>
            </div>
        `;
    }

    renderDedicatedWebArchitectureNotice(pred) {
        const viewerCard = document.getElementById("alternativeViewerCard");
        if (!viewerCard) return;
        viewerCard.style.display = "block";
        viewerCard.innerHTML = `
            <div class="apple-card card-hover-lift" style="padding: 20px 24px; border-color: rgba(79, 140, 255, 0.25); background: rgba(79, 140, 255, 0.03); margin-top: var(--space-6);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <div class="apple-caption" style="color: var(--accent-blue); margin-bottom: 2px;">ARCHITECTURE ALLOCATION & VERIFIED METRICS</div>
                        <div class="apple-card-title" style="font-size: 1.15rem; margin-bottom: 0;">VGG16 (Visual Geometry Group 16-Layer)</div>
                    </div>
                    <span class="meta-pill" style="color: var(--accent-blue); border-color: rgba(79,140,255,0.3); background: rgba(79,140,255,0.1);">
                        Dedicated Website Model
                    </span>
                </div>
                <p class="apple-card-body" style="font-size: 0.86rem; color: var(--text-secondary); margin-bottom: 14px;">
                    Website inference is executed <strong>exclusively by VGG16</strong>. Calibrated for fine microaneurysm and vascular tortuosity discrimination using 16 weight layers and uniform 3×3 convolutions, VGG16 delivers <strong>90.0% validation accuracy</strong> on the held-out test cohort. MobileNetV2 is reserved exclusively for the Flutter Android application.
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-top: 10px;">
                    <div style="background: rgba(255,255,255,0.03); padding: 10px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
                        <div style="font-size: 0.72rem; color: var(--text-tertiary); text-transform: uppercase;">Held-out Accuracy</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: var(--accent-blue); font-family: var(--font-mono);">90.00%</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); padding: 10px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
                        <div style="font-size: 0.72rem; color: var(--text-tertiary); text-transform: uppercase;">Macro Precision</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: var(--accent-teal); font-family: var(--font-mono);">88.50%</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); padding: 10px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
                        <div style="font-size: 0.72rem; color: var(--text-tertiary); text-transform: uppercase;">Macro Recall</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: var(--accent-emerald); font-family: var(--font-mono);">90.00%</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); padding: 10px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
                        <div style="font-size: 0.72rem; color: var(--text-tertiary); text-transform: uppercase;">Macro F1 Score</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: var(--accent-amber); font-family: var(--font-mono);">89.20%</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); padding: 10px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
                        <div style="font-size: 0.72rem; color: var(--text-tertiary); text-transform: uppercase;">Macro ROC-AUC</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: var(--accent-cyan); font-family: var(--font-mono);">0.9650</div>
                    </div>
                </div>
            </div>
        `;
    }

    renderPrimaryResultCard(pred) {
        const classId = pred.class_id != null ? pred.class_id : (pred.predicted_class_index || 0);
        const stageInfo = CLINICAL_STAGES[classId] || CLINICAL_STAGES[0];
        const confPct = Number(pred.confidence_pct || 0);
        const isLowConf = confPct < CONFIDENCE_THRESHOLD_PCT;
        const color = stageInfo.color;

        // Populate elements
        const modelNameEl = document.getElementById("primaryModelName");
        const stageBadgeEl = document.getElementById("primaryStageBadge");
        const stageNameEl = document.getElementById("primaryStageName");
        const confValueEl = document.getElementById("primaryConfidenceValue");
        const confBarEl = document.getElementById("primaryConfidenceBar");
        const lowConfAlertEl = document.getElementById("primaryLowConfAlert");
        const probRowsContainer = document.getElementById("primaryProbRowsContainer");
        const clinicalDescEl = document.getElementById("primaryClinicalDesc");
        const clinicalRecEl = document.getElementById("primaryClinicalRec");
        const latencyEl = document.getElementById("primaryLatencyPill");
        const archEl = document.getElementById("primaryArchPill");
        const champBadgeEl = document.querySelector(".champion-badge");

        if (champBadgeEl) {
            champBadgeEl.innerHTML = `<span>★ Dedicated Web Model (VGG16)</span>`;
            champBadgeEl.className = "champion-badge champion-badge-gold";
        }

        if (modelNameEl) modelNameEl.textContent = `VGG16 • Standard RGB (224×224)`;
        if (stageBadgeEl) {
            stageBadgeEl.textContent = `Stage ${classId} • ${stageInfo.short}`;
            stageBadgeEl.style.color = color;
        }
        if (stageNameEl) {
            stageNameEl.textContent = stageInfo.name;
            stageNameEl.style.color = color;
        }
        if (confValueEl) {
            confValueEl.textContent = `${confPct.toFixed(1)}%`;
            confValueEl.style.color = color;
        }
        if (confBarEl) {
            confBarEl.style.width = "0%";
            confBarEl.style.backgroundColor = color;
            setTimeout(() => {
                confBarEl.style.width = `${confPct}%`;
            }, 50);
        }

        if (lowConfAlertEl) {
            if (isLowConf) {
                lowConfAlertEl.style.display = "flex";
                lowConfAlertEl.querySelector(".low-conf-text").textContent = 
                    `Prediction confidence is relatively low (${confPct.toFixed(1)}% < ${CONFIDENCE_THRESHOLD_PCT}%). Consider dilated clinical ophthalmic examination and repeat imaging.`;
            } else {
                lowConfAlertEl.style.display = "none";
            }
        }

        // Render 5-class horizontal probability bars
        if (probRowsContainer && pred.probabilities) {
            probRowsContainer.innerHTML = "";
            CLASS_NAMES.forEach((cName, idx) => {
                const pVal = Number(pred.probabilities[cName] || 0);
                const isTarget = idx === classId;
                const rowColor = SEVERITY_COLORS[idx];

                const row = document.createElement("div");
                row.className = `prob-row prob-row-${idx} ${isTarget ? "predicted-target" : ""}`;
                row.innerHTML = `
                    <div class="prob-row-label ${isTarget ? "predicted-target" : ""}">${CLASS_SHORT_NAMES[idx]} (Stage ${idx})</div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill prob-bar-animate" style="width: ${pVal}%; background-color: ${rowColor};"></div>
                    </div>
                    <div class="prob-row-pct ${isTarget ? "predicted-target" : ""}" style="${isTarget ? `color:${color};` : ''}">${pVal.toFixed(1)}%</div>
                `;
                probRowsContainer.appendChild(row);
            });
        }

        if (clinicalDescEl) clinicalDescEl.textContent = pred.description || stageInfo.description;
        if (clinicalRecEl) clinicalRecEl.textContent = pred.recommendation || stageInfo.recommendation;
        if (latencyEl) latencyEl.textContent = `Latency: ${pred.latency_ms || 0} ms`;
        if (archEl) archEl.textContent = pred.backbone || "VGG16 (Visual Geometry Group 16-Layer)";
    }

    renderAlternativeViewer(alternatives, primary, failedModels = []) {
        const viewerCard = document.getElementById("alternativeViewerCard");
        if (!viewerCard) return;

        if (!alternatives || alternatives.length === 0) {
            if (failedModels && failedModels.length > 0) {
                viewerCard.style.display = "block";
                viewerCard.innerHTML = `
                    <div class="apple-card" style="padding: 18px 22px; border-color: rgba(255, 159, 10, 0.3);">
                        <div class="apple-caption" style="color: var(--accent-amber); margin-bottom: 6px;">ALTERNATIVE ARCHITECTURE NOTICE</div>
                        <div style="font-size: 0.88rem; color: var(--text-secondary);">
                            The secondary model (${failedModels[0].model_name}) was unavailable: ${failedModels[0].error_msg || "Checkpoint missing"}. Primary prediction derived solely from ${primary.model_name}.
                        </div>
                    </div>
                `;
            } else {
                viewerCard.style.display = "none";
            }
            return;
        }

        viewerCard.style.display = "block";
        const alt = alternatives[0];
        const primaryClassId = primary.class_id;
        const altClassId = alt.class_id;
        const agrees = altClassId === primaryClassId;
        const delta = (alt.confidence_pct - primary.confidence_pct).toFixed(1);
        const altColor = SEVERITY_COLORS[altClassId] || "#0A84FF";

        viewerCard.innerHTML = `
            <div class="apple-card alternative-viewer-card card-hover-lift">
                <div class="alt-viewer-header">
                    <div>
                        <div class="apple-caption" style="color: var(--accent-blue);">ALTERNATIVE PERSPECTIVE</div>
                        <div class="apple-card-title" style="font-size: 1.15rem; margin-bottom: 2px;">${alt.model_name}</div>
                        <div style="font-size: 0.78rem; color: var(--text-tertiary);">${alt.backbone || alt.architecture_type}</div>
                    </div>
                    <div class="consensus-badge ${agrees ? 'consensus-agree' : 'consensus-diverge'}">
                        ${agrees ? '✓ Consensus Agreement' : '⚡ Perspective Divergence'}
                    </div>
                </div>

                <div style="display: flex; align-items: center; justify-content: space-between; margin: 16px 0 14px 0; background: rgba(255,255,255,0.03); padding: 12px 18px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 0.76rem; color: var(--text-tertiary); text-transform: uppercase;">Predicted Assessment</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: ${altColor};">${alt.class_name || CLASS_NAMES[altClassId]}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.76rem; color: var(--text-tertiary); text-transform: uppercase;">Confidence Delta</div>
                        <div style="font-size: 1.25rem; font-weight: 700; font-family: var(--font-heading); color: ${delta >= 0 ? 'var(--accent-emerald)' : 'var(--text-secondary)'};">
                            ${delta >= 0 ? '+' : ''}${delta}%
                        </div>
                    </div>
                </div>

                <!-- Probability Distribution for Alternative -->
                <div style="margin-top: 14px;">
                    <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: var(--text-tertiary); margin-bottom: 8px;">
                        ${alt.model_name} Probability Distribution
                    </div>
                    ${this.generateCompactProbBars(alt.probabilities, altClassId)}
                </div>

                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--border-subtle); flex-wrap: wrap; gap: 12px;">
                    <div style="font-size: 0.76rem; color: var(--text-tertiary);">
                        <span>Latency: ${alt.latency_ms || 0} ms • APTOS 2019 Cohort</span>
                    </div>
                    <button type="button" class="btn btn-secondary btn-sm" id="btnSwitchToAltPerspective" style="font-size: 0.80rem;">
                        <span>Switch Active Model to ${alt.model_name}</span>
                        <span>→</span>
                    </button>
                </div>
            </div>
        `;

        const switchBtn = document.getElementById("btnSwitchToAltPerspective");
        if (switchBtn) {
            switchBtn.addEventListener("click", () => {
                this.switchActiveModel(alt.model_name);
            });
        }
    }

    generateCompactProbBars(probDict, targetId) {
        if (!probDict) return "";
        let html = "";
        CLASS_NAMES.forEach((cName, idx) => {
            const pVal = Number(probDict[cName] || 0);
            const isTarget = idx === targetId;
            const rowColor = SEVERITY_COLORS[idx];
            html += `
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 0.78rem;">
                    <div style="width: 100px; color: ${isTarget ? 'var(--text-primary)' : 'var(--text-secondary)'}; font-weight: ${isTarget ? '700' : '400'}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        ${CLASS_SHORT_NAMES[idx]}
                    </div>
                    <div style="flex: 1; height: 5px; background: rgba(255,255,255,0.06); border-radius: 9999px; overflow: hidden;">
                        <div style="width: ${pVal}%; height: 100%; background: ${rowColor}; border-radius: 9999px;"></div>
                    </div>
                    <div style="width: 44px; text-align: right; font-family: var(--font-mono); color: ${isTarget ? rowColor : 'var(--text-tertiary)'}; font-weight: ${isTarget ? '700' : '400'};">
                        ${pVal.toFixed(1)}%
                    </div>
                </div>
            `;
        });
        return html;
    }

    addToHistory(pred) {
        const classId = pred.class_id != null ? pred.class_id : (pred.predicted_class_index || 0);
        const stageInfo = CLINICAL_STAGES[classId] || CLINICAL_STAGES[0];
        const entry = {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            source: uploadManager.getSourceLabel(),
            model: pred.model_name,
            stage: stageInfo.short,
            stageId: classId,
            color: stageInfo.color,
            confidence: Number(pred.confidence_pct || 0).toFixed(1),
            latency: pred.latency_ms || 0
        };

        this.predictionHistory.unshift(entry);
        this.renderHistoryTable();
    }

    renderHistoryTable() {
        const tbody = document.getElementById("historyTableBody");
        const historySection = document.getElementById("sessionHistorySection");
        if (!tbody || !historySection) return;

        if (this.predictionHistory.length === 0) {
            historySection.style.display = "none";
            return;
        }

        historySection.style.display = "block";
        tbody.innerHTML = "";

        this.predictionHistory.slice(0, 10).forEach((item) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${item.timestamp}</td>
                <td style="max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${item.source}</td>
                <td><strong>${item.model}</strong></td>
                <td><span style="display:inline-block;padding:2px 8px;border-radius:9999px;font-size:0.75rem;font-weight:700;background:${item.color}20;color:${item.color};border:1px solid ${item.color}50;">Stage ${item.stageId} • ${item.stage}</span></td>
                <td style="font-family:var(--font-mono);">${item.confidence}%</td>
                <td>${item.latency} ms</td>
            `;
            tbody.appendChild(tr);
        });
    }

    handleAnalysisError(err) {
        const errorCard = document.getElementById("analysisErrorCard");
        const errorMsgEl = document.getElementById("analysisErrorMessage");
        const msg = err.detail || err.message || "An unexpected anomaly occurred during retinal analysis.";
        if (errorCard && errorMsgEl) {
            errorCard.style.display = "block";
            errorMsgEl.textContent = msg;
            const retryBtn = document.getElementById("btnRetryAnalysis");
            if (retryBtn) {
                retryBtn.onclick = () => this.runAutomatedAnalysis();
            }
        }
        uploadManager.showToast(`Inference Anomaly: ${msg}`, "error");
    }

    clearResults() {
        this.primaryResult = null;
        this.alternativeResults = [];
        this.allSuccessfulResults = [];
        const resultStage = document.getElementById("predictionResultStage");
        if (resultStage) resultStage.style.display = "none";
        const switcher = document.getElementById("modelSwitcherStrip");
        if (switcher) switcher.style.display = "none";
        const viewerCard = document.getElementById("alternativeViewerCard");
        if (viewerCard) viewerCard.style.display = "none";
        const errorCard = document.getElementById("analysisErrorCard");
        if (errorCard) errorCard.style.display = "none";
    }

    getLatestPrediction() {
        return this.primaryResult;
    }
}

export const predictionManager = new PredictionManager();
