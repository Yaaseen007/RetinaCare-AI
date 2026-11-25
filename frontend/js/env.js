/**
 * RetinaCare AI - Production Environment Variables
 * Overridden at Vercel build time if RETINACARE_API_URL is set in Project Settings.
 * For local development on localhost/127.0.0.1, config.js defaults to http://127.0.0.1:8000 automatically.
 */
window.__RETINACARE_API_URL__ = window.__RETINACARE_API_URL__ || "";
