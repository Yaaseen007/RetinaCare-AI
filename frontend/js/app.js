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

        // Periodic Health Check
        await this.pollBackendHealth();
        this.healthInterval = setInterval(() => this.pollBackendHealth(), 12000);
    }

    async pollBackendHealth() {
        try {
            await checkHealth(3000);
            this.isBackendOnline = true;
        } catch {
            this.isBackendOnline = false;
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const app = new App();
    app.init();
});
