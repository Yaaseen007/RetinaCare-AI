/**
 * RetinaCare AI - Magnetic Liquid Glass Dock Navigation
 * Inspired by macOS Dock and @componentry/magnetic-dock.
 * Pure Vanilla JavaScript implementation with physics-based proximity magnification,
 * smooth spring damping, keyboard accessibility, and zero-overhead RAF lifecycle.
 */

import { navigationManager } from "./navigation.js";
import { themeManager } from "./theme.js";

class MagneticDock {
    constructor() {
        this.dockEl = null;
        this.items = [];
        this.rafId = null;
        this.isHovering = false;
        this.mouseX = -9999;
        this.mouseY = -9999;
        
        // Physics constants (tuned for subtle, natural fluid Apple feel)
        this.radius = 135; // Interaction influence radius in px
        this.maxScale = 1.45; // Maximum magnification factor at center of hover
        this.maxLift = 7; // Maximum upward vertical float lift in px
        this.maxMagneticPull = 5; // Max horizontal pull toward cursor in px
        this.springDamping = 0.22; // Lerp smoothing factor for spring return

        // State trackers for spring animation
        this.itemStates = [];

        // Accessibility & Device capabilities
        this.prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        this.supportsFinePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

        // Settings modal
        this.settingsModal = null;
    }

    init() {
        this.dockEl = document.getElementById("magneticDock");
        if (!this.dockEl) return;

        // Collect all interactive dock items
        this.items = Array.from(this.dockEl.querySelectorAll(".dock-item"));
        this.settingsModal = document.getElementById("settingsModal");

        // Initialize per-item spring state
        this.itemStates = this.items.map(() => ({
            currentScale: 1,
            targetScale: 1,
            currentTx: 0,
            targetTx: 0,
            currentTy: 0,
            targetTy: 0
        }));

        this.bindEvents();
        this.initSettingsModal();
        this.syncActiveItem(navigationManager.getActiveView());
    }

    bindEvents() {
        // Pointer / Mouse events on Dock container
        if (this.supportsFinePointer && !this.prefersReducedMotion) {
            this.dockEl.addEventListener("pointerenter", (e) => this.onPointerEnter(e));
            this.dockEl.addEventListener("pointermove", (e) => this.onPointerMove(e));
            this.dockEl.addEventListener("pointerleave", (e) => this.onPointerLeave(e));
        }

        // Listen for media query updates (e.g. user toggles reduced motion or changes display)
        window.matchMedia("(prefers-reduced-motion: reduce)").addEventListener("change", (e) => {
            this.prefersReducedMotion = e.matches;
            if (this.prefersReducedMotion) {
                this.resetAllTransforms();
            }
        });

        // Click routing on each dock item
        this.items.forEach((item, index) => {
            item.addEventListener("click", (e) => {
                e.preventDefault();
                const target = item.getAttribute("data-dock-target");
                this.handleItemClick(target, item);
            });

            // Keyboard navigation (Arrow keys, Home, End)
            item.addEventListener("keydown", (e) => this.handleKeyDown(e, index));
        });

        // Listen to SPA view change events dispatched by navigationManager
        window.addEventListener("viewchange", (e) => {
            if (e.detail && e.detail.view) {
                this.syncActiveItem(e.detail.view);
            }
        });

        // Listen to hash changes directly
        window.addEventListener("hashchange", () => {
            const hash = window.location.hash.replace("#", "").toLowerCase();
            this.syncActiveItem(hash);
        });

        // Recalculate if window resizes
        window.addEventListener("resize", () => {
            this.supportsFinePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
            if (!this.supportsFinePointer) {
                this.resetAllTransforms();
            }
        });
    }

    onPointerEnter(e) {
        if (this.prefersReducedMotion || !this.supportsFinePointer) return;
        this.isHovering = true;
        this.updateMousePos(e);
        this.startAnimationLoop();
    }

    onPointerMove(e) {
        if (!this.isHovering) {
            this.isHovering = true;
            this.startAnimationLoop();
        }
        this.updateMousePos(e);
    }

    onPointerLeave() {
        this.isHovering = false;
        this.mouseX = -9999;
        this.mouseY = -9999;

        // Set all target scales back to rest
        this.itemStates.forEach((state) => {
            state.targetScale = 1;
            state.targetTx = 0;
            state.targetTy = 0;
        });

        // Loop continues until spring settles, then automatically halts
        if (!this.rafId) {
            this.startAnimationLoop();
        }
    }

    updateMousePos(e) {
        const rect = this.dockEl.getBoundingClientRect();
        this.mouseX = e.clientX;
        this.mouseY = e.clientY;
    }

    startAnimationLoop() {
        if (this.rafId) return;

        const tick = () => {
            let hasActiveMovement = false;

            // Compute target physics for each item
            if (this.isHovering && this.mouseX !== -9999) {
                this.items.forEach((item, index) => {
                    const rect = item.getBoundingClientRect();
                    const itemCenterX = rect.left + rect.width / 2;
                    const dist = Math.abs(this.mouseX - itemCenterX);

                    if (dist < this.radius) {
                        // Cosine bell curve interpolation (signature macOS Dock wave)
                        const norm = dist / this.radius;
                        const factor = 0.5 * (1 + Math.cos(Math.PI * norm));
                        
                        this.itemStates[index].targetScale = 1 + (this.maxScale - 1) * factor;
                        this.itemStates[index].targetTy = -this.maxLift * factor;
                        
                        // Subtle magnetic horizontal attraction pull toward cursor
                        const pullDir = (this.mouseX - itemCenterX) / this.radius;
                        this.itemStates[index].targetTx = pullDir * this.maxMagneticPull * factor;
                    } else {
                        this.itemStates[index].targetScale = 1;
                        this.itemStates[index].targetTx = 0;
                        this.itemStates[index].targetTy = 0;
                    }
                });
            }

            // Spring / Lerp updates to current values and apply CSS transforms
            this.items.forEach((item, index) => {
                const state = this.itemStates[index];
                
                // Exponential spring lerp
                const dScale = state.targetScale - state.currentScale;
                const dTx = state.targetTx - state.currentTx;
                const dTy = state.targetTy - state.currentTy;

                state.currentScale += dScale * this.springDamping;
                state.currentTx += dTx * this.springDamping;
                state.currentTy += dTy * this.springDamping;

                // Check if item is still moving noticeably
                if (
                    Math.abs(dScale) > 0.002 ||
                    Math.abs(dTx) > 0.05 ||
                    Math.abs(dTy) > 0.05
                ) {
                    hasActiveMovement = true;
                } else if (!this.isHovering) {
                    state.currentScale = 1;
                    state.currentTx = 0;
                    state.currentTy = 0;
                }

                // Apply CSS transforms directly via custom properties or transform string
                const iconWrapper = item.querySelector(".dock-icon-wrapper");
                if (iconWrapper) {
                    iconWrapper.style.transform = `scale(${state.currentScale.toFixed(3)}) translate3d(${state.currentTx.toFixed(2)}px, ${state.currentTy.toFixed(2)}px, 0)`;
                }
            });

            // If still hovering or spring is settling, continue loop. Otherwise stop cleanly to conserve CPU!
            if (this.isHovering || hasActiveMovement) {
                this.rafId = requestAnimationFrame(tick);
            } else {
                this.rafId = null;
            }
        };

        this.rafId = requestAnimationFrame(tick);
    }

    resetAllTransforms() {
        if (this.rafId) {
            cancelAnimationFrame(this.rafId);
            this.rafId = null;
        }
        this.itemStates.forEach((state) => {
            state.currentScale = 1;
            state.targetScale = 1;
            state.currentTx = 0;
            state.targetTx = 0;
            state.currentTy = 0;
            state.targetTy = 0;
        });
        this.items.forEach((item) => {
            const iconWrapper = item.querySelector(".dock-icon-wrapper");
            if (iconWrapper) {
                iconWrapper.style.transform = "";
            }
        });
    }

    handleItemClick(target, itemEl) {
        if (!target) return;

        // Visual click feedback (haptic spring recoil effect)
        itemEl.classList.add("dock-clicked");
        setTimeout(() => itemEl.classList.remove("dock-clicked"), 260);

        if (target === "settings") {
            this.openSettingsModal();
            return;
        }

        if (target === "benchmark") {
            // Model Performance: Switch to performance view and scroll directly down to benchmarks
            navigationManager.navigateTo("performance");
            this.syncActiveItem("benchmark");
            setTimeout(() => {
                const benchmarkSection = document.getElementById("benchmarkSection") ||
                                         document.querySelector(".benchmark-table-wrapper") ||
                                         document.getElementById("trainingDynamicsCanvas");
                if (benchmarkSection) {
                    benchmarkSection.scrollIntoView({ behavior: "smooth", block: "center" });
                }
            }, 100);
            return;
        }

        // Standard SPA routing via navigationManager
        navigationManager.navigateTo(target);
        this.syncActiveItem(target);

        // Specific sub-scrolling behavior for target sections
        if (target === "prediction") {
            // AI Screening: Smooth scroll to the upload/screening container
            setTimeout(() => {
                const uploadSection = document.getElementById("sampleIngestSection") ||
                                      document.querySelector(".screening-grid");
                if (uploadSection) {
                    uploadSection.scrollIntoView({ behavior: "smooth", block: "start" });
                }
            }, 80);
        } else if (target === "performance") {
            // Dashboard: Scroll to top of performance dashboard cards
            setTimeout(() => {
                window.scrollTo({ top: 0, behavior: "smooth" });
            }, 50);
        } else if (target === "home") {
            window.scrollTo({ top: 0, behavior: "smooth" });
        } else if (target === "about" || target === "contact") {
            window.scrollTo({ top: 0, behavior: "smooth" });
        }
    }

    handleKeyDown(e, currentIndex) {
        let nextIndex = null;

        switch (e.key) {
            case "ArrowRight":
            case "ArrowDown":
                e.preventDefault();
                nextIndex = (currentIndex + 1) % this.items.length;
                break;
            case "ArrowLeft":
            case "ArrowUp":
                e.preventDefault();
                nextIndex = (currentIndex - 1 + this.items.length) % this.items.length;
                break;
            case "Home":
                e.preventDefault();
                nextIndex = 0;
                break;
            case "End":
                e.preventDefault();
                nextIndex = this.items.length - 1;
                break;
            case "Enter":
            case " ":
                e.preventDefault();
                this.items[currentIndex].click();
                return;
            default:
                return;
        }

        if (nextIndex !== null && this.items[nextIndex]) {
            this.items[nextIndex].focus();
        }
    }

    syncActiveItem(activeView) {
        if (!activeView) activeView = "home";

        this.items.forEach((item) => {
            const target = item.getAttribute("data-dock-target");
            
            // Highlight benchmark item specifically if target matches
            let isActive = false;
            if (activeView === "benchmark" && target === "benchmark") {
                isActive = true;
            } else if (activeView === "performance" && target === "performance") {
                isActive = true;
            } else if (target === activeView) {
                isActive = true;
            }

            if (isActive) {
                item.classList.add("active");
                item.setAttribute("aria-current", "page");
            } else {
                item.classList.remove("active");
                item.removeAttribute("aria-current");
            }
        });
    }

    /* =========================================================================
       Settings Modal Lifecycle & Controls
       ========================================================================= */
    initSettingsModal() {
        if (!this.settingsModal) return;

        const closeBtn = document.getElementById("settingsCloseBtn");
        if (closeBtn) {
            closeBtn.addEventListener("click", () => this.closeSettingsModal());
        }

        // Close on backdrop click
        this.settingsModal.addEventListener("click", (e) => {
            if (e.target === this.settingsModal) {
                this.closeSettingsModal();
            }
        });

        // Close on Escape key
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && this.settingsModal.classList.contains("settings-open")) {
                this.closeSettingsModal();
            }
        });

        // Wire Theme Selection buttons in Settings Modal
        const themeBtns = this.settingsModal.querySelectorAll("[data-setting-theme]");
        themeBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                const chosen = btn.getAttribute("data-setting-theme");
                if (chosen === "toggle") {
                    themeManager.toggle();
                } else if (chosen === "light" || chosen === "dark") {
                    themeManager.setTheme(chosen);
                }
                this.updateSettingsThemeUI();
            });
        });

        // Wire Magnification Intensity Toggle
        const intensityBtns = this.settingsModal.querySelectorAll("[data-dock-intensity]");
        intensityBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                intensityBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const val = btn.getAttribute("data-dock-intensity");
                if (val === "subtle") {
                    this.maxScale = 1.25;
                    this.radius = 110;
                } else if (val === "enhanced") {
                    this.maxScale = 1.60;
                    this.radius = 160;
                } else {
                    this.maxScale = 1.45;
                    this.radius = 135;
                }
            });
        });

        this.updateSettingsThemeUI();
    }

    openSettingsModal() {
        if (!this.settingsModal) return;
        this.updateSettingsThemeUI();
        this.settingsModal.style.display = "flex";
        // Force reflow
        void this.settingsModal.offsetWidth;
        this.settingsModal.classList.add("settings-open");
        document.body.style.overflow = "hidden";

        // Focus first focusable item in modal for accessibility
        const firstFocus = this.settingsModal.querySelector("button, [tabindex='0']");
        if (firstFocus) firstFocus.focus();
    }

    closeSettingsModal() {
        if (!this.settingsModal) return;
        this.settingsModal.classList.remove("settings-open");
        document.body.style.overflow = "";
        setTimeout(() => {
            if (!this.settingsModal.classList.contains("settings-open")) {
                this.settingsModal.style.display = "none";
            }
        }, 250);

        // Return focus to settings dock button
        const settingsBtn = document.getElementById("dockSettingsBtn");
        if (settingsBtn) settingsBtn.focus();
    }

    updateSettingsThemeUI() {
        if (!this.settingsModal) return;
        const currentTheme = themeManager.getTheme();
        const themeBtns = this.settingsModal.querySelectorAll("[data-setting-theme]");
        themeBtns.forEach((btn) => {
            const val = btn.getAttribute("data-setting-theme");
            if (val === currentTheme) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });
    }
}

export const magneticDock = new MagneticDock();

