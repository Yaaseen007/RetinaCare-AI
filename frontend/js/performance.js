/**
 * RetinaCare AI - Model Performance & Clinical Benchmarking Dashboard
 * Strictly scoped to retained models: VGG16 and MobileNetV2.
 * Real metrics loaded from FastAPI telemetry or verified evaluation assets.
 */

import { RETAINED_MODELS } from "./config.js";
import { fetchModelMetrics } from "./api.js";

class PerformanceDashboard {
    constructor() {
        this.metricsData = null;
        this.trainingChart = null;
    }

    async init() {
        // Load data on init or when view becomes active
        await this.loadDashboardData();

        // Listen for view changes or theme changes to redraw charts cleanly
        window.addEventListener("viewchange", (e) => {
            if (e.detail && e.detail.view === "performance") {
                this.renderTrainingDynamicsChart();
            }
        });

        window.addEventListener("themechange", () => {
            this.renderTrainingDynamicsChart();
        });
    }

    async loadDashboardData() {
        try {
            const data = await fetchModelMetrics();
            this.metricsData = data;
            this.renderMetricCards(data.primary_model_metrics);
            this.renderBenchmarkTable(data.model_comparison);
            this.renderClassificationBreakdown(data.primary_model_metrics);
            this.renderTrainingDynamicsChart();
        } catch (err) {
            console.warn("Unable to load performance metrics:", err);
        }
    }

    renderMetricCards(primaryMetrics) {
        if (!primaryMetrics) return;

        const accEl = document.getElementById("metricTestAccuracy");
        const precEl = document.getElementById("metricMacroPrecision");
        const recEl = document.getElementById("metricMacroRecall");
        const f1El = document.getElementById("metricMacroF1");

        if (accEl) accEl.textContent = `${(primaryMetrics.accuracy * 100).toFixed(1)}%`;
        if (precEl) precEl.textContent = `${(primaryMetrics.macro_precision * 100).toFixed(1)}%`;
        if (recEl) recEl.textContent = `${(primaryMetrics.macro_recall * 100).toFixed(1)}%`;
        if (f1El) f1El.textContent = `${(primaryMetrics.macro_f1 * 100).toFixed(1)}%`;
    }

    renderBenchmarkTable(comparisonData) {
        const tbody = document.getElementById("benchmarkTableBody");
        if (!tbody || !comparisonData) return;

        tbody.innerHTML = "";

        // Strictly filter to VGG16 and MobileNetV2
        RETAINED_MODELS.forEach((modelKey) => {
            // Find key in comparisonData case-insensitively
            const foundKey = Object.keys(comparisonData).find(
                (k) => k.toLowerCase() === modelKey.toLowerCase()
            );

            if (!foundKey) return;
            const rowData = comparisonData[foundKey];
            const isVGG = foundKey === "VGG16";
            const isMobile = foundKey === "MobileNetV2";

            const tr = document.createElement("tr");
            tr.className = "champion-row";

            tr.innerHTML = `
                <td>
                    <div style="font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 8px;">
                        <span>${foundKey}</span>
                        ${isVGG ? '<span style="font-size:0.68rem;padding:2px 8px;border-radius:9999px;background:rgba(79,140,255,0.15);color:#4F8CFF;border:1px solid rgba(79,140,255,0.3);letter-spacing:0.02em;">Website Dedicated Model (90.0% Top Acc)</span>' : ''}
                        ${isMobile ? '<span style="font-size:0.68rem;padding:2px 8px;border-radius:9999px;background:rgba(120,220,232,0.15);color:#78DCE8;border:1px solid rgba(120,220,232,0.3);letter-spacing:0.02em;">Flutter Mobile App Model (88.0% Acc)</span>' : ''}
                    </div>
                </td>
                <td style="font-family: var(--font-mono); font-weight: 700;">${rowData["Accuracy"] || "N/A"}</td>
                <td style="font-family: var(--font-mono);">${rowData["Precision"] || "N/A"}</td>
                <td style="font-family: var(--font-mono);">${rowData["Recall"] || "N/A"}</td>
                <td style="font-family: var(--font-mono);">${rowData["F1 Score"] || "N/A"}</td>
                <td style="font-family: var(--font-mono);">${rowData["ROC-AUC"] || "N/A"}</td>
                <td>${rowData["Training Time"] || "N/A"}</td>
                <td>
                    <span style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:9999px;font-size:0.75rem;font-weight:600;background:rgba(66,214,164,0.12);color:#42D6A4;border:1px solid rgba(66,214,164,0.3);">
                        <span class="pulse-dot" style="background:#42D6A4;width:6px;height:6px;border-radius:50%;display:inline-block;box-shadow:0 0 8px #42D6A4;"></span>
                        ${isVGG ? 'Active (Web)' : 'Active (Mobile)'}
                    </span>
                </td>
            `;
            tbody.appendChild(tr);
        });
    }

    renderClassificationBreakdown(primaryMetrics) {
        const tableBody = document.getElementById("classificationReportBody");
        if (!tableBody || !primaryMetrics || !primaryMetrics.classification_report) return;

        tableBody.innerHTML = "";
        const report = primaryMetrics.classification_report;

        const classLabels = [
            "No DR (0)",
            "Mild (1)",
            "Moderate (2)",
            "Severe (3)",
            "Proliferative (4)"
        ];

        classLabels.forEach((lbl) => {
            if (report[lbl]) {
                const item = report[lbl];
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td style="font-weight:600;">${lbl}</td>
                    <td style="font-family:var(--font-mono);">${(item.precision * 100).toFixed(1)}%</td>
                    <td style="font-family:var(--font-mono);">${(item.recall * 100).toFixed(1)}%</td>
                    <td style="font-family:var(--font-mono);">${(item["f1-score"] * 100).toFixed(1)}%</td>
                    <td style="font-family:var(--font-mono);color:var(--text-tertiary);">${item.support}</td>
                `;
                tableBody.appendChild(tr);
            }
        });
    }

    async renderTrainingDynamicsChart() {
        const canvas = document.getElementById("trainingDynamicsCanvas");
        if (!canvas) return;

        // Fetch real training history JSON
        let historyData = null;
        try {
            const res = await fetch("assets/data/training_history.json");
            if (res.ok) {
                historyData = await res.json();
            }
        } catch {
            return;
        }

        if (!historyData || !historyData.accuracy) return;

        const epochs = historyData.accuracy.map((_, i) => `Epoch ${i + 1}`);
        const isDark = document.documentElement.getAttribute("data-theme") !== "light";

        const textColor = isDark ? "#A8B3C7" : "#556480";
        const gridColor = isDark ? "rgba(255, 255, 255, 0.06)" : "rgba(11, 18, 32, 0.07)";

        // Destroy existing Chart.js instance if any
        if (this.trainingChart) {
            this.trainingChart.destroy();
            this.trainingChart = null;
        }

        // Check if Chart.js is loaded
        if (typeof Chart !== "undefined") {
            const ctx = canvas.getContext("2d");
            this.trainingChart = new Chart(ctx, {
                type: "line",
                data: {
                    labels: epochs,
                    datasets: [
                        {
                            label: "Train Accuracy",
                            data: historyData.accuracy.map((v) => Number((v * 100).toFixed(1))),
                            borderColor: "#4F8CFF",
                            backgroundColor: "rgba(79, 140, 255, 0.10)",
                            borderWidth: 2.5,
                            pointRadius: 3,
                            pointHoverRadius: 6,
                            tension: 0.35,
                            fill: true
                        },
                        {
                            label: "Val Accuracy",
                            data: historyData.val_accuracy.map((v) => Number((v * 100).toFixed(1))),
                            borderColor: "#78DCE8",
                            borderDash: [5, 4],
                            backgroundColor: "transparent",
                            borderWidth: 2.2,
                            pointRadius: 3,
                            pointHoverRadius: 6,
                            tension: 0.35
                        },
                        {
                            label: "Train Loss",
                            data: historyData.loss.map((v) => Number(v.toFixed(2))),
                            borderColor: "#F4BD64",
                            backgroundColor: "transparent",
                            borderWidth: 1.8,
                            pointRadius: 2,
                            yAxisID: "y1",
                            tension: 0.35
                        },
                        {
                            label: "Val Loss",
                            data: historyData.val_loss.map((v) => Number(v.toFixed(2))),
                            borderColor: "#FF727C",
                            borderDash: [3, 3],
                            backgroundColor: "transparent",
                            borderWidth: 1.8,
                            pointRadius: 2,
                            yAxisID: "y1",
                            tension: 0.35
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: "index",
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            position: "top",
                            align: "end",
                            labels: {
                                color: isDark ? "#F5F7FC" : "#0B1220",
                                font: { size: 11, family: "Inter, -apple-system, sans-serif", weight: "500" },
                                boxWidth: 12,
                                usePointStyle: true,
                                pointStyle: "circle"
                            }
                        },
                        tooltip: {
                            backgroundColor: isDark ? "rgba(16, 27, 48, 0.94)" : "rgba(255, 255, 255, 0.96)",
                            titleColor: isDark ? "#F5F7FC" : "#0B1220",
                            bodyColor: isDark ? "#A8B3C7" : "#556480",
                            borderColor: isDark ? "rgba(255, 255, 255, 0.12)" : "rgba(11, 18, 32, 0.12)",
                            borderWidth: 1,
                            padding: 12,
                            boxPadding: 6,
                            cornerRadius: 12
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: gridColor },
                            ticks: { color: textColor, font: { size: 10 } }
                        },
                        y: {
                            type: "linear",
                            position: "left",
                            min: 30,
                            max: 100,
                            title: { display: true, text: "Accuracy (%)", color: textColor, font: { size: 11 } },
                            grid: { color: gridColor },
                            ticks: { color: textColor, font: { size: 10 } }
                        },
                        y1: {
                            type: "linear",
                            position: "right",
                            min: 0,
                            max: 2.0,
                            title: { display: true, text: "Cross-Entropy Loss", color: textColor, font: { size: 11 } },
                            grid: { drawOnChartArea: false },
                            ticks: { color: textColor, font: { size: 10 } }
                        }
                    }
                }
            });
        } else {
            this.renderNativeCanvasFallback(canvas, historyData, isDark);
        }
    }

    renderNativeCanvasFallback(canvas, historyData, isDark) {
        const ctx = canvas.getContext("2d");
        const w = (canvas.width = canvas.parentElement ? canvas.parentElement.clientWidth : 500);
        const h = (canvas.height = 260);

        ctx.clearRect(0, 0, w, h);
        const padL = 40, padR = 20, padT = 30, padB = 30;
        const plotW = w - padL - padR;
        const plotH = h - padT - padB;

        // Grid lines
        ctx.strokeStyle = isDark ? "rgba(255,255,255,0.06)" : "rgba(11,18,32,0.06)";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padT + (plotH / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padL, y);
            ctx.lineTo(w - padR, y);
            ctx.stroke();

            // Label
            ctx.fillStyle = isDark ? "#A8B3C7" : "#556480";
            ctx.font = "10px Inter, -apple-system, sans-serif";
            ctx.textAlign = "right";
            ctx.fillText(`${100 - i * 20}%`, padL - 6, y + 3);
        }

        const count = historyData.accuracy.length;
        if (count < 2) return;

        // Draw helper
        const drawCurve = (data, minVal, maxVal, color) => {
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            data.forEach((val, i) => {
                const x = padL + (i / (count - 1)) * plotW;
                const normalized = (val - minVal) / (maxVal - minVal);
                const y = padT + plotH * (1 - normalized);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        };

        // Train accuracy & Val accuracy
        drawCurve(historyData.accuracy.map(v => v * 100), 20, 100, "#4F8CFF");
        drawCurve(historyData.val_accuracy.map(v => v * 100), 20, 100, "#78DCE8");

        // Legend
        ctx.font = "11px Inter, -apple-system, sans-serif";
        ctx.textAlign = "left";
        ctx.fillStyle = "#4F8CFF";
        ctx.fillText("● Train Acc", padL + 10, 16);
        ctx.fillStyle = "#78DCE8";
        ctx.fillText("● Val Acc", padL + 100, 16);
    }
}

export const performanceDashboard = new PerformanceDashboard();
