/**
 * RetinaCare AI - Centralized FastAPI HTTP Client SDK
 * Handles network timeouts, FormData multipart serialization, response parsing, and errors.
 */

import { API_BASE_URL, getApiBaseUrl } from "./config.js";

/**
 * Returns the active base URL dynamically to ensure runtime updates (e.g. from Settings)
 * take effect immediately without requiring a browser reload.
 */
function getBase() {
    return typeof getApiBaseUrl === "function" ? getApiBaseUrl() : API_BASE_URL;
}

export class APIError extends Error {
    constructor(message, status = 0, detail = null, isNetworkError = false) {
        super(message);
        this.name = "APIError";
        this.status = status;
        this.detail = detail || message;
        this.isNetworkError = isNetworkError;
    }
}

/**
 * Executes a fetch request with timeout protection.
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = 45000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        return response;
    } catch (err) {
        clearTimeout(timeoutId);
        if (err.name === "AbortError") {
            throw new APIError(`Request timed out after ${Math.round(timeoutMs / 1000)}s`, 408, null, true);
        }
        throw new APIError(`Network connection failed: ${err.message}`, 0, null, true);
    }
}

/**
 * Pings GET /api/health to check service availability.
 */
export async function checkHealth(timeoutMs = 4000) {
    const url = `${getBase()}/api/health`;
    try {
        const res = await fetchWithTimeout(url, { method: "GET" }, timeoutMs);
        if (!res.ok) {
            throw new APIError(`Health check failed with status ${res.status}`, res.status);
        }
        return await res.json();
    } catch (err) {
        if (err instanceof APIError) throw err;
        throw new APIError(`Cannot connect to RetinaCare API at ${getBase()}`, 0, err.message, true);
    }
}

/**
 * Fetches list of all neural models from GET /api/models.
 */
export async function fetchModels(timeoutMs = 6000) {
    const url = `${getBase()}/api/models`;
    try {
        const res = await fetchWithTimeout(url, { method: "GET" }, timeoutMs);
        if (!res.ok) {
            const errBody = await res.text().catch(() => "");
            throw new APIError(`Failed to fetch models: ${res.statusText}`, res.status, errBody);
        }
        const data = await res.json();
        return data.models || [];
    } catch (err) {
        if (err instanceof APIError) throw err;
        throw new APIError(`Model list fetch failed: ${err.message}`, 0, null, true);
    }
}

/**
 * Sends fundus image to POST /api/compare for concurrent multi-model analysis.
 *
 * @param {Blob|File} imageBlob - Decoded retinal fundus photograph blob
 * @param {string} filename - Source filename
 * @param {number} timeoutMs - Max execution duration (default 90s)
 */
export async function compareModels(imageBlob, filename = "retinal_image.jpg", timeoutMs = 90000) {
    const url = `${getBase()}/api/compare`;
    const formData = new FormData();
    formData.append("file", imageBlob, filename);

    try {
        const res = await fetchWithTimeout(url, {
            method: "POST",
            body: formData
        }, timeoutMs);

        if (!res.ok) {
            let errorDetail = `HTTP ${res.status} ${res.statusText}`;
            try {
                const jsonErr = await res.json();
                if (jsonErr && jsonErr.detail) {
                    errorDetail = jsonErr.detail;
                }
            } catch {
                const textErr = await res.text();
                if (textErr) errorDetail = textErr;
            }
            throw new APIError(errorDetail, res.status, errorDetail);
        }

        return await res.json();
    } catch (err) {
        if (err instanceof APIError) throw err;
        throw new APIError(`Comparison pipeline execution error: ${err.message}`, 0, null, true);
    }
}

/**
 * Sends fundus image to dedicated website prediction route POST /api/predict/web (VGG16).
 *
 * @param {Blob|File} imageBlob - Decoded retinal fundus photograph blob
 * @param {string} filename - Source filename
 * @param {number} timeoutMs - Max execution duration (default 60s)
 */
export async function predictWeb(imageBlob, filename = "retinal_image.jpg", timeoutMs = 60000) {
    const url = `${getBase()}/api/predict/web`;
    const formData = new FormData();
    formData.append("file", imageBlob, filename);
    formData.append("confidence_threshold", "50.0");
    formData.append("include_preprocessed_image", "true");

    try {
        const res = await fetchWithTimeout(url, {
            method: "POST",
            body: formData
        }, timeoutMs);

        if (!res.ok) {
            let errorDetail = `HTTP ${res.status} ${res.statusText}`;
            try {
                const jsonErr = await res.json();
                if (jsonErr && jsonErr.detail) errorDetail = jsonErr.detail;
            } catch {
                const textErr = await res.text();
                if (textErr) errorDetail = textErr;
            }
            throw new APIError(errorDetail, res.status, errorDetail);
        }

        return await res.json();
    } catch (err) {
        if (err instanceof APIError) throw err;
        throw new APIError(`Website VGG16 screening request failed: ${err.message}`, 0, null, true);
    }
}

/**
 * Sends fundus image to dedicated mobile prediction route POST /api/predict/mobile (MobileNetV2).
 *
 * @param {Blob|File} imageBlob - Decoded retinal fundus photograph blob
 * @param {string} filename - Source filename
 * @param {number} timeoutMs - Max execution duration (default 60s)
 */
export async function predictMobile(imageBlob, filename = "retinal_image.jpg", timeoutMs = 60000) {
    const url = `${getBase()}/api/predict/mobile`;
    const formData = new FormData();
    formData.append("file", imageBlob, filename);
    formData.append("confidence_threshold", "50.0");
    formData.append("include_preprocessed_image", "true");

    try {
        const res = await fetchWithTimeout(url, {
            method: "POST",
            body: formData
        }, timeoutMs);

        if (!res.ok) {
            let errorDetail = `HTTP ${res.status} ${res.statusText}`;
            try {
                const jsonErr = await res.json();
                if (jsonErr && jsonErr.detail) errorDetail = jsonErr.detail;
            } catch {
                const textErr = await res.text();
                if (textErr) errorDetail = textErr;
            }
            throw new APIError(errorDetail, res.status, errorDetail);
        }

        return await res.json();
    } catch (err) {
        if (err instanceof APIError) throw err;
        throw new APIError(`Mobile prediction request failed: ${err.message}`, 0, null, true);
    }
}

/**
 * Sends fundus image to POST /api/predict for a single specific neural model.
 */
export async function predictSingle(imageBlob, modelName = "VGG16", confidenceThreshold = 50.0, filename = "retinal_image.jpg", timeoutMs = 60000) {
    const url = `${getBase()}/api/predict`;
    const formData = new FormData();
    formData.append("file", imageBlob, filename);
    formData.append("model_name", modelName);
    formData.append("client_type", modelName === "MobileNetV2" ? "mobile" : "web");
    formData.append("confidence_threshold", String(confidenceThreshold));
    formData.append("include_preprocessed_image", "true");

    try {
        const res = await fetchWithTimeout(url, {
            method: "POST",
            body: formData
        }, timeoutMs);

        if (!res.ok) {
            let errorDetail = `HTTP ${res.status} ${res.statusText}`;
            try {
                const jsonErr = await res.json();
                if (jsonErr && jsonErr.detail) errorDetail = jsonErr.detail;
            } catch {
                const textErr = await res.text();
                if (textErr) errorDetail = textErr;
            }
            throw new APIError(errorDetail, res.status, errorDetail);
        }

        return await res.json();
    } catch (err) {
        if (err instanceof APIError) throw err;
        throw new APIError(`Inference request failed: ${err.message}`, 0, null, true);
    }
}

/**
 * Fetches offline telemetry metrics from GET /api/model-metrics, falling back to static JSON if offline.
 */
export async function fetchModelMetrics(timeoutMs = 6000) {
    const url = `${getBase()}/api/model-metrics`;
    try {
        const res = await fetchWithTimeout(url, { method: "GET" }, timeoutMs);
        if (res.ok) {
            return await res.json();
        }
    } catch {
        // Silently proceed to static fallback
    }

    // Static fallback to assets/data/*.json
    try {
        const [compRes, metricsRes] = await Promise.all([
            fetch("assets/data/model_comparison.json"),
            fetch("assets/data/metrics.json")
        ]);

        const compData = compRes.ok ? await compRes.json() : {};
        const primaryData = metricsRes.ok ? await metricsRes.json() : null;

        return {
            model_comparison: compData,
            primary_model_metrics: primaryData
        };
    } catch (err) {
        throw new APIError(`Metrics unavailable: ${err.message}`, 0, null, true);
    }
}
