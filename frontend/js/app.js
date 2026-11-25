/**
 * RetinaCare AI - Application Entry Point & Telemetry Supervisor
 * Integrates theme, navigation, ingestion, prediction, performance metrics, and assistant.
 */

import { themeManager } from "./theme.js";
import { navigationManager } from "./navigation.js";
import { magneticDock } from "./magnetic-dock.js";
import { uploadManager } from "./upload.js";
import { predictionManager } from "./prediction.js";
import { performanceDashboard } from "./performance.js";
import { chatbotManager } from "./chatbot.js";
import { checkHealth } from "./api.js";
import { getApiBaseUrl, setApiBaseUrl } from "./config.js";

class App {
    constructor() {
        this.isBackendOnline = false;
        this.healthInterval = null;
    }

    async init() {
        console.log("Initializing RetinaCare AI Web Frontend...");

        // Initialize submodules
        themeManager.init();
        navigationManager.init();
        magneticDock.init();
        uploadManager.init();
        predictionManager.init();
        performanceDashboard.init();
        chatbotManager.init();

        // Wire Theme Toggle Button
        const themeBtn = document.getElementById("themeToggleBtn");
        if (themeBtn) {
            themeBtn.addEventListener("click", () => {
                themeManager.toggle();
            });
        }

        // Wire Backend URL configuration in Settings Modal
        this.initBackendSettings();

        // Periodic Health Check
        await this.pollBackendHealth();
        this.healthInterval = setInterval(() => this.pollBackendHealth(), 15000);
    }

    initBackendSettings() {
        const urlInput = document.getElementById("backendUrlInput");
        const saveBtn = document.getElementById("saveBackendUrlBtn");
        const testBtn = document.getElementById("testBackendUrlBtn");
        const resetBtn = document.getElementById("resetBackendUrlBtn");

        if (urlInput) {
            urlInput.value = getApiBaseUrl();
        }

        if (saveBtn && urlInput) {
            saveBtn.addEventListener("click", async () => {
                const newUrl = urlInput.value.trim();
                if (!newUrl) {
                    uploadManager.showToast("Please enter a valid backend URL", "error");
                    return;
                }
                setApiBaseUrl(newUrl);
                uploadManager.showToast(`Backend API URL updated: ${newUrl}`, "success");
                await this.pollBackendHealth();
            });
        }

        if (testBtn && urlInput) {
            testBtn.addEventListener("click", async () => {
                testBtn.disabled = true;
                testBtn.textContent = "Testing...";
                const startTime = performance.now();
                try {
                    const health = await checkHealth(5000);
                    const elapsed = Math.round(performance.now() - startTime);
                    uploadManager.showToast(`Connected successfully in ${elapsed}ms! (${health.status || 'OK'})`, "success");
                    this.updateBackendStatusBadge(true, elapsed);
                } catch (err) {
                    uploadManager.showToast(`Connection failed: ${err.message}`, "error");
                    this.updateBackendStatusBadge(false);
                } finally {
                    testBtn.disabled = false;
                    testBtn.textContent = "Test";
                }
            });
        }

        if (resetBtn && urlInput) {
            resetBtn.addEventListener("click", (e) => {
                e.preventDefault();
                try {
                    localStorage.removeItem("retinacare_api_url");
                } catch {}
                delete window.__RETINACARE_API_URL__;
                const defUrl = getApiBaseUrl();
                urlInput.value = defUrl;
                setApiBaseUrl(defUrl);
                uploadManager.showToast("Reset to default backend URL", "info");
                this.pollBackendHealth();
            });
        }
    }

    updateBackendStatusBadge(isOnline, latencyMs = null) {
        const badge = document.getElementById("backendStatusBadge");
        const desc = document.getElementById("backendStatusDesc");
        const currentUrl = getApiBaseUrl();

        if (desc) {
            desc.innerHTML = `Active target: <code>${currentUrl}</code>`;
        }

        if (badge) {
            if (isOnline) {
                const latencyText = latencyMs ? ` (${latencyMs}ms)` : "";
                badge.style.color = "var(--accent-emerald)";
                badge.style.background = "rgba(48,209,88,0.12)";
                badge.style.borderColor = "rgba(48,209,88,0.3)";
                badge.textContent = `● Online${latencyText}`;
            } else {
                badge.style.color = "var(--accent-rose)";
                badge.style.background = "rgba(255,69,58,0.12)";
                badge.style.borderColor = "rgba(255,69,58,0.3)";
                badge.textContent = "✕ Offline / Sleeping";
            }
        }
    }

    async pollBackendHealth() {
        try {
            const start = performance.now();
            await checkHealth(4000);
            const elapsed = Math.round(performance.now() - start);
            this.isBackendOnline = true;
            this.updateBackendStatusBadge(true, elapsed);
        } catch {
            this.isBackendOnline = false;
            this.updateBackendStatusBadge(false);
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const app = new App();
    app.init();
});
