/**
 * RetinaCare AI - Single Page Application Navigation & Routing
 * Controls active views (Home, Prediction, Performance, About) and mobile drawer.
 */

class NavigationManager {
    constructor() {
        this.activeView = "home";
        this.views = ["home", "prediction", "performance", "about", "contact"];
    }

    init() {
        // Wire navigation buttons
        document.querySelectorAll("[data-nav-target]").forEach((btn) => {
            btn.addEventListener("click", (e) => {
                e.preventDefault();
                const target = btn.getAttribute("data-nav-target");
                if (target) {
                    this.navigateTo(target);
                }
            });
        });

        // Listen for browser popstate / hash change
        window.addEventListener("hashchange", () => {
            this.handleHashChange();
        });

        // Mobile drawer toggle
        const mobileToggle = document.getElementById("mobileNavToggle");
        const navLinks = document.getElementById("navLinks");
        if (mobileToggle && navLinks) {
            mobileToggle.addEventListener("click", (e) => {
                e.stopPropagation();
                navLinks.classList.toggle("nav-open");
            });

            // Close mobile drawer when clicking anywhere outside
            document.addEventListener("click", (e) => {
                if (navLinks.classList.contains("nav-open") && !navLinks.contains(e.target) && !mobileToggle.contains(e.target)) {
                    navLinks.classList.remove("nav-open");
                }
            });

            // Automatically clean up on window resize back to desktop
            window.addEventListener("resize", () => {
                if (window.innerWidth > 768) {
                    navLinks.classList.remove("nav-open");
                }
            });
        }

        // Initial route
        this.handleHashChange();
    }

    handleHashChange() {
        const hash = window.location.hash.replace("#", "").toLowerCase();
        if (hash === "benchmark") {
            this.navigateTo("performance", false);
            window.dispatchEvent(new CustomEvent("viewchange", { detail: { view: "benchmark" } }));
            setTimeout(() => {
                const b = document.getElementById("benchmarkSection") || document.querySelector(".benchmark-table-wrapper");
                if (b) b.scrollIntoView({ behavior: "smooth", block: "center" });
            }, 100);
        } else if (this.views.includes(hash)) {
            this.navigateTo(hash, false);
        } else {
            this.navigateTo("home", false);
        }
    }

    navigateTo(viewName, updateHash = true) {
        if (!this.views.includes(viewName)) return;

        this.activeView = viewName;

        // Update view sections
        this.views.forEach((v) => {
            const el = document.getElementById(`view-${v}`);
            if (el) {
                if (v === viewName) {
                    el.classList.add("active-view", "page-enter");
                } else {
                    el.classList.remove("active-view", "page-enter");
                }
            }
        });

        // Update nav bar buttons
        document.querySelectorAll(".nav-item button").forEach((btn) => {
            const target = btn.getAttribute("data-nav-target");
            if (target === viewName) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });

        // Close mobile drawer if open
        const navLinks = document.getElementById("navLinks");
        if (navLinks) {
            navLinks.classList.remove("nav-open");
        }

        if (updateHash) {
            window.location.hash = `#${viewName}`;
        }

        // Smooth scroll to top of content
        window.scrollTo({ top: 0, behavior: "smooth" });

        // Dispatch view change event for charts or components
        window.dispatchEvent(new CustomEvent("viewchange", { detail: { view: viewName } }));
    }

    getActiveView() {
        return this.activeView;
    }
}

export const navigationManager = new NavigationManager();
