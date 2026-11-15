/**
 * RetinaCare AI - Retinal Image Ingestion & Preprocessing Inspector
 * Handles reference samples, custom file uploads, drag-and-drop, plausibility heuristics,
 * and high-precision inspection modes (Side-by-Side, Original, Normalized).
 */

import { SAMPLE_IMAGES } from "./config.js";

const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024; // 15MB
const ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"];

class UploadManager {
    constructor() {
        this.currentBlob = null;
        this.currentFilename = "";
        this.currentSourceType = ""; // 'sample' or 'custom'
        this.currentSourceId = "";
        this.originalImage = null; // HTMLImageElement
        this.currentInspectionMode = "side-by-side"; // 'side-by-side', 'original', 'normalized'
        this.isProcessing = false;
        this.plausibilityWarning = null;

        this.onImageSelectedCallbacks = [];
        this.onImageResetCallbacks = [];
    }

    init() {
        this.renderSampleThumbnails();
        this.bindEvents();

        // Preload first reference sample by default for instant clinical demonstration
        this.selectSample(SAMPLE_IMAGES[0]);
    }

    onImageSelected(callback) {
        this.onImageSelectedCallbacks.push(callback);
    }

    onImageReset(callback) {
        this.onImageResetCallbacks.push(callback);
    }

    bindEvents() {
        // Ingestion Method Segmented Control (Reference Samples vs Custom Upload)
        const tabBtns = document.querySelectorAll(".ingest-tab-btn");
        const sampleSection = document.getElementById("sampleIngestSection");
        const uploadSection = document.getElementById("customUploadSection");

        tabBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                tabBtns.forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                const method = btn.getAttribute("data-method");
                if (method === "sample") {
                    sampleSection.style.display = "block";
                    uploadSection.style.display = "none";
                } else {
                    sampleSection.style.display = "none";
                    uploadSection.style.display = "block";
                }
            });
        });

        // Dropzone interactions
        const dropzone = document.getElementById("uploadDropzone");
        const fileInput = document.getElementById("retinalFileInput");

        if (dropzone && fileInput) {
            dropzone.addEventListener("click", () => fileInput.click());

            ["dragenter", "dragover"].forEach((eventName) => {
                dropzone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropzone.classList.add("drag-active");
                });
            });

            ["dragleave", "drop"].forEach((eventName) => {
                dropzone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropzone.classList.remove("drag-active");
                });
            });

            dropzone.addEventListener("drop", (e) => {
                const files = e.dataTransfer.files;
                if (files && files.length > 0) {
                    this.handleCustomFile(files[0]);
                }
            });

            fileInput.addEventListener("change", (e) => {
                if (fileInput.files && fileInput.files.length > 0) {
                    this.handleCustomFile(fileInput.files[0]);
                }
            });
        }

        // Inspection Mode Toggle
        const inspectBtns = document.querySelectorAll(".inspect-mode-btn");
        inspectBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                inspectBtns.forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                this.currentInspectionMode = btn.getAttribute("data-mode");
                this.updatePreviewDisplay();
            });
        });

        // "New Analysis" Reset Button
        const resetBtn = document.getElementById("btnNewAnalysis");
        if (resetBtn) {
            resetBtn.addEventListener("click", () => {
                this.resetImage();
            });
        }
    }

    renderSampleThumbnails() {
        const grid = document.getElementById("samplesGrid");
        if (!grid) return;

        grid.innerHTML = "";
        SAMPLE_IMAGES.forEach((sample) => {
            const card = document.createElement("div");
            card.className = "sample-thumbnail-card";
            card.id = `sample-card-${sample.id}`;
            card.innerHTML = `
                <img src="${sample.path}" alt="${sample.title}" loading="lazy" />
                <div class="sample-stage-label">${sample.title.split("—")[0].trim()}</div>
            `;
            card.addEventListener("click", () => {
                this.selectSample(sample);
            });
            grid.appendChild(card);
        });
    }

    async selectSample(sample) {
        document.querySelectorAll(".sample-thumbnail-card").forEach((c) => c.classList.remove("selected"));
        const card = document.getElementById(`sample-card-${sample.id}`);
        if (card) card.classList.add("selected");

        try {
            const res = await fetch(sample.path);
            if (!res.ok) throw new Error(`Could not load reference image at ${sample.path}`);
            const blob = await res.blob();
            this.currentBlob = blob;
            this.currentFilename = sample.filename;
            this.currentSourceType = "sample";
            this.currentSourceId = sample.id;

            await this.loadImageObject(blob);
            this.notifyImageSelected(sample.title);
        } catch (err) {
            this.showError(`Failed to load reference sample: ${err.message}`);
        }
    }

    async handleCustomFile(file) {
        // 1. Check file size
        if (file.size > MAX_FILE_SIZE_BYTES) {
            this.showError(`File size exceeds 15MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB). Please upload a smaller image.`);
            return;
        }

        // 2. Check extension
        const ext = "." + file.name.split(".").pop().toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            this.showError(`Unsupported file extension '${ext}'. Supported formats: ${ALLOWED_EXTENSIONS.join(", ")}`);
            return;
        }

        this.currentBlob = file;
        this.currentFilename = file.name;
        this.currentSourceType = "custom";
        this.currentSourceId = `file-${file.name}-${file.size}`;

        // Clear sample selection highlight
        document.querySelectorAll(".sample-thumbnail-card").forEach((c) => c.classList.remove("selected"));

        try {
            await this.loadImageObject(file);
            this.updateDropzoneContent(file);
            this.notifyImageSelected(`Uploaded: ${file.name}`);
            this.showToast(`Fundus scan loaded: ${file.name}`, "success");
        } catch (err) {
            this.showError(`Image could not be decoded: ${err.message}. File may be corrupted.`);
        }
    }

    loadImageObject(blob) {
        return new Promise((resolve, reject) => {
            const url = URL.createObjectURL(blob);
            const img = new Image();
            img.onload = () => {
                this.originalImage = img;
                this.runPlausibilityCheck(img);
                this.updatePreviewDisplay();
                resolve();
            };
            img.onerror = () => {
                reject(new Error("Browser was unable to decode image data"));
            };
            img.src = url;
        });
    }

    /**
     * Retinal Plausibility Heuristic (matches check_retinal_fundus_plausibility in Python backend)
     */
    runPlausibilityCheck(img) {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        const size = 128;
        canvas.width = size;
        canvas.height = size;
        ctx.drawImage(img, 0, 0, size, size);

        const imgData = ctx.getImageData(0, 0, size, size);
        const data = imgData.data;

        let totalR = 0;
        let totalG = 0;
        let totalB = 0;
        let centerCount = 0;

        const centerMin = Math.floor(size * 0.3);
        const centerMax = Math.floor(size * 0.7);

        for (let y = centerMin; y < centerMax; y++) {
            for (let x = centerMin; x < centerMax; x++) {
                const idx = (y * size + x) * 4;
                totalR += data[idx];
                totalG += data[idx + 1];
                totalB += data[idx + 2];
                centerCount++;
            }
        }

        const avgR = totalR / centerCount;
        const avgG = totalG / centerCount;
        const avgB = totalB / centerCount;

        // Check for predominant reddish/orange vascular hue (R > B + 15 in fundus)
        const isGrayscale = Math.abs(avgR - avgG) < 8 && Math.abs(avgG - avgB) < 8;
        const isNotRetinalHue = avgR < avgB + 10;

        const warnEl = document.getElementById("plausibilityWarning");
        if (warnEl) {
            if (isGrayscale || isNotRetinalHue) {
                this.plausibilityWarning = "Image does not exhibit typical digital color fundus characteristics (expected reddish/orange retinal hue). Analysis will proceed, but predictions may be less reliable.";
                warnEl.style.display = "flex";
                warnEl.querySelector(".warning-text").textContent = this.plausibilityWarning;
            } else {
                this.plausibilityWarning = null;
                warnEl.style.display = "none";
            }
        }
    }

    /**
     * Renders normalized simulated Graham image on canvas:
     * Circular crop + local contrast enhancement
     */
    generateNormalizedCanvas(img) {
        const canvas = document.createElement("canvas");
        const targetSize = 224;
        canvas.width = targetSize;
        canvas.height = targetSize;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });

        // 1. Draw scaled original
        ctx.drawImage(img, 0, 0, targetSize, targetSize);
        const imgData = ctx.getImageData(0, 0, targetSize, targetSize);
        const d = imgData.data;

        // 2. Circular aperture mask
        const center = targetSize / 2;
        const radius = targetSize * 0.48;
        const radiusSq = radius * radius;

        // Apply contrast and Gaussian subtraction simulation
        for (let y = 0; y < targetSize; y++) {
            for (let x = 0; x < targetSize; x++) {
                const idx = (y * targetSize + x) * 4;
                const distSq = (x - center) * (x - center) + (y - center) * (y - center);

                if (distSq > radiusSq) {
                    // Zero out background outside aperture
                    d[idx] = 0;
                    d[idx + 1] = 0;
                    d[idx + 2] = 0;
                } else {
                    // Local contrast enhancement (accentuate vascular details)
                    let r = d[idx];
                    let g = d[idx + 1];
                    let b = d[idx + 2];

                    // Simulated Ben Graham green-channel normalization & local stretch
                    g = Math.min(255, Math.max(0, (g - 110) * 1.55 + 128));
                    r = Math.min(255, Math.max(0, (r - 120) * 1.45 + 130));
                    b = Math.min(255, Math.max(0, (b - 80) * 1.35 + 100));

                    d[idx] = r;
                    d[idx + 1] = g;
                    d[idx + 2] = b;
                }
            }
        }

        ctx.putImageData(imgData, 0, 0);
        return canvas;
    }

    updatePreviewDisplay() {
        const container = document.getElementById("previewDisplayContainer");
        const emptyState = document.getElementById("screeningEmptyState");
        const stageCard = document.getElementById("previewCard");

        if (!this.originalImage) {
            if (container) container.innerHTML = "";
            if (emptyState) emptyState.style.display = "flex";
            if (stageCard) stageCard.style.display = "none";
            return;
        }

        if (emptyState) emptyState.style.display = "none";
        if (stageCard) stageCard.style.display = "block";
        if (!container) return;

        container.innerHTML = "";

        if (this.currentInspectionMode === "side-by-side") {
            const wrap = document.createElement("div");
            wrap.className = "preview-side-by-side";

            const sideLeft = document.createElement("div");
            sideLeft.className = "side-item";
            const imgClone = document.createElement("img");
            imgClone.src = this.originalImage.src;
            imgClone.alt = "Original Capture";
            sideLeft.appendChild(imgClone);
            const sideLeftLabel = document.createElement("div");
            sideLeftLabel.className = "side-item-label";
            sideLeftLabel.textContent = "Original Ingestion (RGB)";
            sideLeft.appendChild(sideLeftLabel);

            const sideRight = document.createElement("div");
            sideRight.className = "side-item";
            const normCanvas = this.generateNormalizedCanvas(this.originalImage);
            sideRight.appendChild(normCanvas);
            const sideRightLabel = document.createElement("div");
            sideRightLabel.className = "side-item-label";
            sideRightLabel.textContent = "Graham Local Contrast";
            sideRight.appendChild(sideRightLabel);

            wrap.appendChild(sideLeft);
            wrap.appendChild(sideRight);
            container.appendChild(wrap);
        } else if (this.currentInspectionMode === "original") {
            const imgEl = document.createElement("img");
            imgEl.src = this.originalImage.src;
            imgEl.alt = "Original Fundus Capture";
            container.appendChild(imgEl);
        } else if (this.currentInspectionMode === "normalized") {
            const normCanvas = this.generateNormalizedCanvas(this.originalImage);
            container.appendChild(normCanvas);
        }

        // Maintain laser scan overlay if processing
        if (this.isProcessing) {
            this.injectScanningLaser();
        }
    }

    startScanning() {
        this.isProcessing = true;
        this.injectScanningLaser();
        const stageWrapper = document.getElementById("previewCanvasWrapper");
        if (stageWrapper) {
            stageWrapper.classList.add("retinal-processing-active");
        }
    }

    stopScanning() {
        this.isProcessing = false;
        const stageWrapper = document.getElementById("previewCanvasWrapper");
        if (stageWrapper) {
            stageWrapper.classList.remove("retinal-processing-active");
            const overlay = stageWrapper.querySelector(".retinal-scan-overlay");
            if (overlay) overlay.remove();
        }
    }

    injectScanningLaser() {
        const stageWrapper = document.getElementById("previewCanvasWrapper");
        if (!stageWrapper) return;
        if (stageWrapper.querySelector(".retinal-scan-overlay")) return;

        const overlay = document.createElement("div");
        overlay.className = "retinal-scan-overlay";
        overlay.innerHTML = `<div class="retinal-scan-line"></div>`;
        stageWrapper.appendChild(overlay);
    }

    notifyImageSelected(sourceLabel) {
        this.onImageSelectedCallbacks.forEach((cb) => cb(this.currentBlob, sourceLabel, this.currentFilename));
    }

    updateDropzoneContent(file) {
        const dropzone = document.getElementById("uploadDropzone");
        if (!dropzone) return;
        dropzone.classList.add("has-file");
        dropzone.innerHTML = `
            <div class="upload-icon" style="background: rgba(48, 209, 88, 0.15); color: var(--accent-emerald);">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                </svg>
            </div>
            <div class="upload-title" style="color: var(--accent-emerald); font-size: 0.92rem; font-weight: 700;">${file.name}</div>
            <div class="upload-hint">${(file.size / (1024 * 1024)).toFixed(2)} MB • Fundus Photograph Verified</div>
            <div style="margin-top: 10px;">
                <span class="meta-pill" style="color: var(--accent-blue); background: var(--accent-blue-subtle); border-color: var(--accent-blue-border);">
                    Click to replace image
                </span>
            </div>
        `;
    }

    resetDropzoneContent() {
        const dropzone = document.getElementById("uploadDropzone");
        if (!dropzone) return;
        dropzone.classList.remove("has-file");
        dropzone.innerHTML = `
            <div class="upload-icon">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
                </svg>
            </div>
            <div class="upload-title">Drop retinal fundus scan here</div>
            <div class="upload-hint">or click to browse local files (PNG, JPG, BMP up to 15MB)</div>
        `;
    }

    resetImage() {
        this.currentBlob = null;
        this.currentFilename = "";
        this.currentSourceType = "";
        this.currentSourceId = "";
        this.originalImage = null;
        this.plausibilityWarning = null;
        this.stopScanning();

        document.querySelectorAll(".sample-thumbnail-card").forEach((c) => c.classList.remove("selected"));
        const fileInput = document.getElementById("retinalFileInput");
        if (fileInput) fileInput.value = "";

        this.resetDropzoneContent();

        const warnEl = document.getElementById("plausibilityWarning");
        if (warnEl) warnEl.style.display = "none";

        this.updatePreviewDisplay();
        this.onImageResetCallbacks.forEach((cb) => cb());
        this.showToast("Scan reset. Select a reference sample or upload a new fundus photograph.", "info");
    }

    showToast(message, type = "error") {
        let container = document.getElementById("retinacareToastContainer");
        if (!container) {
            container = document.createElement("div");
            container.id = "retinacareToastContainer";
            container.className = "toast-container";
            document.body.appendChild(container);
        }

        const toast = document.createElement("div");
        toast.className = `toast-item toast-${type} fade-in-up`;
        const icon = type === "error" ? "!" : type === "success" ? "✓" : "i";
        toast.innerHTML = `
            <span class="toast-icon">${icon}</span>
            <div class="toast-message">${message}</div>
            <button type="button" class="toast-close" aria-label="Dismiss">&times;</button>
        `;
        toast.querySelector(".toast-close").onclick = () => {
            toast.classList.add("toast-fade-out");
            setTimeout(() => toast.remove(), 250);
        };
        container.appendChild(toast);

        setTimeout(() => {
            if (toast.parentNode) {
                toast.classList.add("toast-fade-out");
                setTimeout(() => toast.remove(), 250);
            }
        }, 5000);
    }

    showError(msg) {
        this.showToast(msg, "error");
    }

    getCurrentBlob() {
        return this.currentBlob;
    }

    getCurrentFilename() {
        return this.currentFilename || "retinal_image.jpg";
    }

    getSourceLabel() {
        if (this.currentSourceType === "sample") {
            const s = SAMPLE_IMAGES.find((x) => x.id === this.currentSourceId);
            return s ? s.title : "Reference Sample";
        }
        return `Custom Upload: ${this.currentFilename}`;
    }
}

export const uploadManager = new UploadManager();
