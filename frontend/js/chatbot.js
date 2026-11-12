/**
 * RetinaCare AI Assistant - Intelligent Clinical Decision Support Chatbot
 * Floating bottom-left assistant with context-aware retinal analysis explanations,
 * ICDR stage guidance, model diagnostics, and educational disclaimers.
 */

import { CLINICAL_STAGES, CLASS_NAMES, CLASS_SHORT_NAMES, MEDICAL_DISCLAIMER } from "./config.js";
import { predictionManager } from "./prediction.js";

class ChatbotManager {
    constructor() {
        this.isOpen = false;
        this.currentPrediction = null;
        this.messages = [];
        this.isProcessing = false;
        this.initialized = false;
    }

    init() {
        if (this.initialized) return;
        this.initialized = true;

        const triggerBtn = document.getElementById("chatbotTriggerBtn");
        const closeBtn = document.getElementById("chatCloseBtn");
        const sendBtn = document.getElementById("chatSendBtn");
        const inputField = document.getElementById("chatInputField");

        if (triggerBtn) {
            triggerBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                this.toggleWindow();
            });
        }
        if (closeBtn) {
            closeBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                this.closeWindow();
            });
        }
        if (sendBtn && inputField) {
            sendBtn.addEventListener("click", () => this.handleSendMessage());
            inputField.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    this.handleSendMessage();
                }
            });
        }

        // Close on Escape or click outside
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && this.isOpen) {
                this.closeWindow();
            }
        });
        document.addEventListener("click", (e) => {
            const win = document.getElementById("chatbotWindow");
            const trigger = document.getElementById("chatbotTriggerBtn");
            if (this.isOpen && win && trigger && !win.contains(e.target) && !trigger.contains(e.target)) {
                this.closeWindow();
            }
        });

        // Listen for new predictions to enable context-aware answers
        predictionManager.onPredictionCompleted((primary, alternatives) => {
            this.currentPrediction = primary;
            this.updateContextChip();
        });

        // Wire suggested chips
        this.bindChips();

        // Initial welcome message
        this.appendMessage("assistant", "Hello! I am your RetinaCare Clinical Assistant. I can explain your retinal screening predictions, the 5 ICDR Diabetic Retinopathy stages, Ben Graham preprocessing, and model confidence metrics. How can I help you today?");
    }

    toggleWindow() {
        if (this.isOpen) {
            this.closeWindow();
        } else {
            this.openWindow();
        }
    }

    openWindow() {
        this.isOpen = true;
        const win = document.getElementById("chatbotWindow");
        if (win) {
            win.classList.add("open");
            const input = document.getElementById("chatInputField");
            if (input) setTimeout(() => input.focus(), 150);
        }
    }

    closeWindow() {
        this.isOpen = false;
        const win = document.getElementById("chatbotWindow");
        if (win) {
            win.classList.remove("open");
        }
    }

    updateContextChip() {
        const contextChip = document.getElementById("chipCurrentResult");
        if (contextChip && this.currentPrediction) {
            const stage = this.currentPrediction.class_id != null ? this.currentPrediction.class_id : 0;
            contextChip.textContent = `Explain my ${this.currentPrediction.model_name} (Stage ${stage}) scan`;
            contextChip.style.display = "inline-block";
        }
    }

    bindChips() {
        document.querySelectorAll(".chat-chip").forEach((chip) => {
            chip.addEventListener("click", () => {
                const query = chip.getAttribute("data-query") || chip.textContent.trim();
                this.submitUserQuery(query);
            });
        });
    }

    submitUserQuery(queryText) {
        if (this.isProcessing || !queryText.trim()) return;

        this.isProcessing = true;
        this.appendMessage("user", queryText);

        const input = document.getElementById("chatInputField");
        const sendBtn = document.getElementById("chatSendBtn");
        if (input) {
            input.value = "";
            input.disabled = true;
        }
        if (sendBtn) sendBtn.disabled = true;

        // Show typing indicator
        this.showTypingIndicator();

        setTimeout(() => {
            this.hideTypingIndicator();
            const reply = this.generateClinicalResponse(queryText);
            this.appendMessage("assistant", reply);
            this.isProcessing = false;
            if (input) {
                input.disabled = false;
                input.focus();
            }
            if (sendBtn) sendBtn.disabled = false;
        }, 500);
    }

    handleSendMessage() {
        const input = document.getElementById("chatInputField");
        if (!input) return;
        const text = input.value.trim();
        if (!text) return;
        this.submitUserQuery(text);
    }

    formatMarkdown(text) {
        if (!text) return "";
        let formatted = text
            // Escape HTML entities
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            // Bold **text**
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            // Italic *text*
            .replace(/\*(.+?)\*/g, "<em>$1</em>")
            // Inline code `code`
            .replace(/`([^`]+)`/g, "<code class=\"chat-inline-code\">$1</code>");

        // Format bullet points
        const lines = formatted.split("\n");
        const processedLines = lines.map((line) => {
            const trimmed = line.trim();
            if (trimmed.startsWith("• ") || trimmed.startsWith("- ")) {
                return `<div class="chat-bullet-row"><span class="chat-bullet">•</span><span>${trimmed.substring(2)}</span></div>`;
            }
            return line;
        });

        return processedLines.join("<br>");
    }

    appendMessage(role, text) {
        const container = document.getElementById("chatMessagesContainer");
        if (!container) return;

        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const htmlContent = role === "assistant" ? this.formatMarkdown(text) : document.createTextNode(text).textContent;

        const bubble = document.createElement("div");
        bubble.className = `chat-bubble chat-bubble-${role} fade-in-up`;
        bubble.innerHTML = `
            <div class="chat-bubble-content">${htmlContent}</div>
            <div class="chat-timestamp">${timeStr}</div>
        `;

        container.appendChild(bubble);
        container.scrollTop = container.scrollHeight;
    }

    showTypingIndicator() {
        const container = document.getElementById("chatMessagesContainer");
        if (!container) return;
        if (document.getElementById("chatTypingIndicator")) return;

        const ind = document.createElement("div");
        ind.id = "chatTypingIndicator";
        ind.className = "chat-bubble chat-bubble-assistant fade-in-up";
        ind.style.color = "var(--text-tertiary)";
        ind.innerHTML = `
            <div style="display:flex;align-items:center;gap:6px;">
                <span class="pulse-dot" style="background:var(--accent-blue);box-shadow:0 0 8px var(--accent-blue);"></span>
                <span style="font-size:0.82rem;font-style:italic;">RetinaCare Assistant is formulating response...</span>
            </div>
        `;
        container.appendChild(ind);
        container.scrollTop = container.scrollHeight;
    }

    hideTypingIndicator() {
        const ind = document.getElementById("chatTypingIndicator");
        if (ind) ind.remove();
    }

    /**
     * Context-aware clinical decision engine with strict guardrails
     */
    generateClinicalResponse(query) {
        const q = query.toLowerCase();

        // 1. Context-Aware Scan Explanation
        if (q.includes("my scan") || q.includes("current result") || q.includes("explain my") || q.includes("this image")) {
            if (!this.currentPrediction) {
                return "You haven't run an analysis yet. Select a reference fundus image or upload a scan, click 'Analyze Retinal Scan', and I'll provide an in-depth interpretation of the predicted biomarkers.";
            }

            const p = this.currentPrediction;
            const cid = p.class_id != null ? p.class_id : (p.predicted_class_index || 0);
            const stage = CLINICAL_STAGES[cid] || CLINICAL_STAGES[0];
            const conf = Number(p.confidence_pct || 0).toFixed(1);

            return `**Analysis Summary:** Your scan was assessed by **${p.model_name}** as **Stage ${cid}: ${stage.name}** with a Softmax confidence of **${conf}%**.\n\n` +
                   `**Clinical Findings:** ${p.description || stage.description}\n\n` +
                   `**Recommended Next Step:** ${p.recommendation || stage.recommendation}\n\n` +
                   `*Note: A ${conf}% confidence indicates the model's posterior probability on this image, not historical test accuracy.*`;
        }

        // 2. 5 ICDR Stages
        if (q.includes("5 stages") || q.includes("stages") || q.includes("grades") || q.includes("icdr") || q.includes("scale")) {
            return "The International Clinical Diabetic Retinopathy (ICDR) severity scale defines 5 disease stages:\n\n" +
                   "• **Stage 0 (No DR):** No visible vascular lesions. Annual screening recommended.\n" +
                   "• **Stage 1 (Mild DR):** Microaneurysms only. 6–12 month follow-up.\n" +
                   "• **Stage 2 (Moderate DR):** Multiple microaneurysms, dot/blot hemorrhages, or venous beading. 3–6 month referral.\n" +
                   "• **Stage 3 (Severe DR):** Follows the 4-2-1 rule (>20 hemorrhages in 4 quadrants, venous beading in 2, or IRMA in 1). High risk of vision loss; 2–4 week urgent referral.\n" +
                   "• **Stage 4 (Proliferative DR):** Neovascularization or vitreous/preretinal hemorrhage. High imminent blindness risk; 24–48 hour immediate referral for PRP or anti-VEGF.";
        }

        // 3. Stage-specific questions
        if (q.includes("stage 0") || q.includes("no dr")) {
            return `**Stage 0 — No Diabetic Retinopathy:** ${CLINICAL_STAGES[0].description} Recommendation: ${CLINICAL_STAGES[0].recommendation}`;
        }
        if (q.includes("stage 1") || q.includes("mild")) {
            return `**Stage 1 — Mild Non-Proliferative DR:** ${CLINICAL_STAGES[1].description} Recommendation: ${CLINICAL_STAGES[1].recommendation}`;
        }
        if (q.includes("stage 2") || q.includes("moderate")) {
            return `**Stage 2 — Moderate Non-Proliferative DR:** ${CLINICAL_STAGES[2].description} Recommendation: ${CLINICAL_STAGES[2].recommendation}`;
        }
        if (q.includes("stage 3") || q.includes("severe")) {
            return `**Stage 3 — Severe Non-Proliferative DR:** ${CLINICAL_STAGES[3].description} Recommendation: ${CLINICAL_STAGES[3].recommendation}`;
        }
        if (q.includes("stage 4") || q.includes("proliferative")) {
            return `**Stage 4 — Proliferative Diabetic Retinopathy:** ${CLINICAL_STAGES[4].description} Recommendation: ${CLINICAL_STAGES[4].recommendation}`;
        }

        // 4. Preprocessing & Ben Graham
        if (q.includes("graham") || q.includes("preprocessing") || q.includes("normalization") || q.includes("contrast") || q.includes("aperture")) {
            return "Our standardization pipeline uses the **Ben Graham Method**:\n\n" +
                   "1. **Circular Aperture Masking:** Detects the dark boundaries of the fundus camera and centers the spherical retina.\n" +
                   "2. **Local Contrast Normalization:** Subtracts a heavily Gaussian-blurred version of the image to remove illumination variance, camera flash gradients, and vignetting across different clinics.\n" +
                   "3. **Standardized 224×224 Resizing:** Scales vascular structures consistently for deep learning tensor inputs.";
        }

        // 5. Model Comparisons & Platform Allocation (VGG16 for Web vs MobileNetV2 for Mobile)
        if (q.includes("vgg") || q.includes("mobilenet") || q.includes("difference") || q.includes("compare models") || q.includes("which model")) {
            return "RetinaCare AI utilizes a dedicated dual-model architecture:\n\n" +
                   "• **VGG16 (Website Engine):** Classic 16-layer deep convolutional network dedicated exclusively to the web platform. Reaches **90.0% test accuracy** with **0.9650 ROC-AUC** using uniform 3×3 convolutions for foundational multi-scale retinal feature representation.\n" +
                   "• **MobileNetV2 (Flutter Mobile App Engine):** Ultra-efficient inverted residual architecture reserved exclusively for the Flutter Android application. Reaches **88.0% test accuracy** with **1.0000 ROC-AUC** and ultra-low latency (~30-50ms), optimized for on-device mobile screening.\n\n" +
                   "This architectural separation ensures high-capacity deep feature extraction on the web and fast, lightweight execution on mobile devices.";
        }

        // 6. Confidence vs Accuracy
        if (q.includes("confidence") || q.includes("accuracy") || q.includes("threshold")) {
            return "**Confidence vs. Accuracy:**\n\n" +
                   "• **Prediction Confidence:** The Softmax probability output of the neural network for a specific uploaded scan (e.g. 88.5% confidence for Stage 2).\n" +
                   "• **Held-out Test Accuracy:** The empirical classification correctness of the model measured across the entire validation cohort (e.g. 90.0% for VGG16).\n\n" +
                   "If model confidence falls below our **50% threshold**, an alert is flagged advising repeat imaging and clinical examination.";
        }

        // 7. Clinical referral guidance
        if (q.includes("referral") || q.includes("doctor") || q.includes("ophthalmologist") || q.includes("specialist") || q.includes("when to see")) {
            return "Referral timing depends on the identified disease stage:\n\n" +
                   "• **Stage 0:** Annual routine dilated eye examination.\n" +
                   "• **Stage 1 (Mild):** Eye exam in 6 to 12 months with primary metabolic management.\n" +
                   "• **Stage 2 (Moderate):** Formal ophthalmology referral within 3 to 6 months.\n" +
                   "• **Stage 3 (Severe):** Urgent ophthalmology referral within 2 to 4 weeks.\n" +
                   "• **Stage 4 (Proliferative):** Emergency ophthalmology referral within 24 to 48 hours for immediate intervention.";
        }

        // 8. Strict Guardrails for Out-of-Scope / Off-Topic Queries
        const ophthalmicTerms = ["retina", "eye", "fundus", "vision", "dr", "diabetic", "diabetic retinopathy", "cataract", "glaucoma", "macular", "vessel", "camera", "aptos", "model", "vgg", "mobilenet", "stage", "grade", "confidence", "accuracy", "graham", "bleeding", "aneurysm", "laser", "vegf"];
        const isOphthalmic = ophthalmicTerms.some((t) => q.includes(t));

        if (!isOphthalmic) {
            return "I am the RetinaCare Clinical Decision Assistant, specialized exclusively in retinal health, diabetic retinopathy screening, and our deep learning model outputs. Please ask a question related to retinal imaging, the 5 ICDR disease stages, model confidence, or clinical screening guidelines.";
        }

        // Default clinical answer
        return "Diabetic Retinopathy is a microvascular complication of diabetes characterized by progressive capillary damage, ischemia, and potential neovascularization in the retina. Regular fundus screening allows early detection before irreversible vision loss occurs. How can I assist you with your retinal analysis?";
    }
}

export const chatbotManager = new ChatbotManager();
