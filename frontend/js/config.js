/**
 * RetinaCare AI - Application Configuration & Clinical Domain Constants
 * Standardized for VGG16 and MobileNetV2 architectures and ICDR 5-stage grading.
 */

// Centralized API Base URL with Multi-Tier Fallback:
// 1. User manual override via Settings modal (stored in localStorage)
// 2. Global window environment variable (window.__RETINACARE_API_URL__ from js/env.js)
// 3. Current origin if served by FastAPI on port 8000
// 4. Localhost fallback (http://127.0.0.1:8000) for local frontend servers
// 5. Cloud backend fallback for production (Vercel)
export function getApiBaseUrl() {
    try {
        const stored = localStorage.getItem("retinacare_api_url");
        if (stored && stored.trim()) {
            return stored.trim().replace(/\/+$/, "");
        }
    } catch (e) {}

    if (window.__RETINACARE_API_URL__ && window.__RETINACARE_API_URL__.trim()) {
        return window.__RETINACARE_API_URL__.trim().replace(/\/+$/, "");
    }

    if (window.location.port === "8000") {
        return window.location.origin;
    }

    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
        return "http://127.0.0.1:8000";
    }

    return "https://yaaseen007-retinacare-ai.hf.space";
}

export let API_BASE_URL = getApiBaseUrl();

export function setApiBaseUrl(newUrl) {
    if (!newUrl) return getApiBaseUrl();
    const clean = newUrl.trim().replace(/\/+$/, "");
    try {
        localStorage.setItem("retinacare_api_url", clean);
    } catch (e) {}
    window.__RETINACARE_API_URL__ = clean;
    API_BASE_URL = clean;
    return API_BASE_URL;
}

// The strictly retained deep learning architectures
// Platform Model Allocation: VGG16 exclusively for Website, MobileNetV2 exclusively for Mobile
export const WEB_MODEL = "VGG16";
export const MOBILE_MODEL = "MobileNetV2";
export const RETAINED_MODELS = ["VGG16", "MobileNetV2"];
export const RETIRED_MODELS = ["Custom CNN", "ResNet50", "EfficientNetB0"];

// Standardized 5-Stage ICDR Clinical Class Mapping
export const NUM_CLASSES = 5;

export const CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Non-Proliferative Retinopathy (NPDR)",
    "Moderate Non-Proliferative Retinopathy (NPDR)",
    "Severe Non-Proliferative Retinopathy (NPDR)",
    "Proliferative Diabetic Retinopathy (PDR)"
];

export const CLASS_SHORT_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR (PDR)"
];

// Liquid Glass Healthcare Severity Palette
export const SEVERITY_COLORS = [
    "#42D6A4", // Stage 0 - Success Emerald
    "#78DCE8", // Stage 1 - Soft Cyan
    "#F4BD64", // Stage 2 - Amber Warning
    "#FF9248", // Stage 3 - Vibrant Tangerine
    "#FF727C"  // Stage 4 - Precision Crimson Error
];

// Clinical Knowledge Base per Disease Stage
export const CLINICAL_STAGES = {
    0: {
        stageId: 0,
        name: "No Diabetic Retinopathy",
        short: "No DR",
        color: "#42D6A4",
        description: "No microaneurysms, hemorrhages, or microvascular abnormalities visible. Retinal anatomy and vascular caliber are clinically intact.",
        recommendation: "Routine annual diabetic fundus screening recommended. Maintain optimal glycemic and blood pressure control."
    },
    1: {
        stageId: 1,
        name: "Mild Non-Proliferative Retinopathy (NPDR)",
        short: "Mild NPDR",
        color: "#78DCE8",
        description: "Microaneurysms only. Represents the earliest clinically observable stage of microvascular injury without overt exudation.",
        recommendation: "Glycemic optimization and follow-up dilated screening in 6 to 12 months with an eye care specialist."
    },
    2: {
        stageId: 2,
        name: "Moderate Non-Proliferative Retinopathy (NPDR)",
        short: "Moderate NPDR",
        color: "#F4BD64",
        description: "Multiple microaneurysms, dot-and-blot hemorrhages, venous beading, or cotton wool spots, but less than severe criteria.",
        recommendation: "Referral to ophthalmology within 3 to 6 months. Closer metabolic monitoring and comprehensive dilated fundus exam."
    },
    3: {
        stageId: 3,
        name: "Severe Non-Proliferative Retinopathy (NPDR)",
        short: "Severe NPDR",
        color: "#FF9248",
        description: "Any 4-2-1 rule criterion met: >20 intraretinal hemorrhages in each of 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant.",
        recommendation: "Urgent ophthalmology referral within 2 to 4 weeks. High risk of rapid progression to proliferative retinopathy."
    },
    4: {
        stageId: 4,
        name: "Proliferative Diabetic Retinopathy (PDR)",
        short: "Proliferative DR (PDR)",
        color: "#FF727C",
        description: "Neovascularization of the optic disc or elsewhere, or vitreous/preretinal hemorrhage. High imminent risk of profound vision loss.",
        recommendation: "Immediate ophthalmology referral within 24 to 48 hours for panretinal photocoagulation (PRP) or anti-VEGF therapy."
    }
};

// Available Reference Clinical Samples
export const SAMPLE_IMAGES = [
    {
        id: "sample_0",
        stage: 0,
        title: "Stage 0 — No DR",
        subtitle: "Healthy retina, intact vascular branches",
        filename: "sample_0_no_dr.png",
        path: "assets/sample_images/sample_0_no_dr.png"
    },
    {
        id: "sample_1",
        stage: 1,
        title: "Stage 1 — Mild",
        subtitle: "Isolated microaneurysms only",
        filename: "sample_1_mild.png",
        path: "assets/sample_images/sample_1_mild.png"
    },
    {
        id: "sample_2",
        stage: 2,
        title: "Stage 2 — Moderate",
        subtitle: "Multiple dot/blot hemorrhages & exudates",
        filename: "sample_2_moderate.png",
        path: "assets/sample_images/sample_2_moderate.png"
    },
    {
        id: "sample_3",
        stage: 3,
        title: "Stage 3 — Severe",
        subtitle: "4-quadrant hemorrhages & venous beading",
        filename: "sample_3_severe.png",
        path: "assets/sample_images/sample_3_severe.png"
    },
    {
        id: "sample_4",
        stage: 4,
        title: "Stage 4 — Proliferative",
        subtitle: "Neovascularization & vitreous traction risk",
        filename: "sample_4_proliferative.png",
        path: "assets/sample_images/sample_4_proliferative.png"
    }
];

export const CONFIDENCE_THRESHOLD_PCT = 50.0;

export const MEDICAL_DISCLAIMER = 
    "RetinaCare AI is an experimental clinical decision-support and screening research prototype. " +
    "Predictions are generated solely by deep learning models trained on the APTOS 2019 dataset and do NOT " +
    "constitute definitive medical diagnoses or clinical advice. All retinal findings must be evaluated by a qualified ophthalmologist.";
