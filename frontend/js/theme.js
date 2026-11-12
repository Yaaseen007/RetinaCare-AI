/**
 * RetinaCare AI - Theme Manager
 * Apple-style Dark / Light Mode with system preference detection and persistence.
 */

const THEME_STORAGE_KEY = "retinacare-ui-theme";

class ThemeManager {
    constructor() {
        this.currentTheme = "dark";
        this.init();
    }

    init() {
        if (this.initialized) {
            this.updateToggleIcon();
            return;
        }
        this.initialized = true;

        // Read saved preference or system preference
        const saved = localStorage.getItem(THEME_STORAGE_KEY);
        if (saved && (saved === "dark" || saved === "light")) {
            this.setTheme(saved);
        } else {
            const systemPrefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
            this.setTheme(systemPrefersLight ? "light" : "dark");
        }

        // Listen for OS system theme changes
        window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", (e) => {
            if (!localStorage.getItem(THEME_STORAGE_KEY)) {
                this.setTheme(e.matches ? "light" : "dark");
            }
        });
    }

    setTheme(theme) {
        this.currentTheme = theme;
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem(THEME_STORAGE_KEY, theme);
        this.updateToggleIcon();

        // Dispatch custom event for components like charts
        window.dispatchEvent(new CustomEvent("themechange", { detail: { theme } }));
    }

    toggle() {
        const next = this.currentTheme === "dark" ? "light" : "dark";
        this.setTheme(next);
        return next;
    }

    getTheme() {
        return this.currentTheme;
    }

    updateToggleIcon() {
        const btn = document.getElementById("themeToggleBtn");
        if (!btn) return;

        if (this.currentTheme === "dark") {
            // Sun icon for switching to light
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58c-.39-.39-1.03-.39-1.41 0s-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41L5.99 4.58zm12.37 12.37c-.39-.39-1.03-.39-1.41 0s-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41l-1.06-1.06zm1.06-10.96c.39-.39.39-1.03 0-1.41s-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06zM7.05 18.36c.39-.39.39-1.03 0-1.41s-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06z"/>
                </svg>
            `;
            btn.setAttribute("title", "Switch to Light Mode");
            btn.setAttribute("aria-label", "Switch to Light Mode");
        } else {
            // Moon icon for switching to dark
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12.3 2a10 10 0 0 0-.19 20 10.04 10.04 0 0 0 9.89-8.49 1 1 0 0 0-1.19-1.16 8 8 0 0 1-9.75-9.75 1 1 0 0 0-1.16-1.19c-.53.11-1.07.31-1.6.59z"/>
                </svg>
            `;
            btn.setAttribute("title", "Switch to Dark Mode");
            btn.setAttribute("aria-label", "Switch to Dark Mode");
        }
    }
}

export const themeManager = new ThemeManager();
