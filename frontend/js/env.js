/**
 * RetinaCare AI - Production Environment Variables
 * This file sets the active backend API URL when deployed to production.
 * For local development, it defaults to http://127.0.0.1:8000 automatically.
 * On Vercel, this URL points to your deployed cloud backend.
 */
window.__RETINACARE_API_URL__ = window.__RETINACARE_API_URL__ || "https://yaaseen007-retinacare-ai.hf.space";
