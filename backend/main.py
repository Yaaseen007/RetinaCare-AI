"""
==============================================================================
RetinaCare AI - FastAPI Clinical Screening & Multi-Model Inference Backend
Department of Computer Science & Engineering - GM University
==============================================================================
"""

from __future__ import annotations
import base64
import io
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Suppress verbose TensorFlow C++ and oneDNN runtime output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Project root resolution & external libraries
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_PYLIB = Path("E:/pylib")
if _PYLIB.exists() and str(_PYLIB) not in sys.path:
    sys.path.insert(0, str(_PYLIB))

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image

# Import existing core domain modules
from src.model_manager import (
    CANONICAL_MODEL_NAMES,
    MODEL_REGISTRY,
    resolve_model_key,
    get_model_status,
    get_all_models_status,
    load_model_instance,
    MissingModelError,
    ModelCompatibilityError
)
from src.class_mapping import (
    CLASS_NAMES,
    NUM_CLASSES,
    DATASET_NAME,
    MEDICAL_DISCLAIMER,
    ClassMappingVerificationError
)
from src.validation import (
    inspect_and_decode_image,
    check_retinal_fundus_plausibility,
    ImageValidationError,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES
)
from src.prediction import (
    run_model_inference,
    compare_all_available_models,
    compute_model_agreement,
    validate_and_rank_model_predictions,
    PredictionExecutionError,
    CONFIDENCE_THRESHOLD_PCT
)
from utils.config import (
    MODEL_COMPARISON_JSON_PATH,
    METRICS_JSON_PATH,
    DEFAULT_MODEL_NAME,
    WEB_MODEL_NAME,
    MOBILE_MODEL_NAME
)
from backend.schemas import (
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
    PredictionResponse,
    ModelComparisonItem,
    ModelAgreementInfo,
    ModelComparisonResponse,
    MetricsResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetinaCare.Backend")

app = FastAPI(
    title="RetinaCare AI — Retinal Health API",
    description="High-performance clinical screening and multi-model deep learning inference API for Diabetic Retinopathy.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pil_to_base64_data_url(img: Union[Image.Image, np.ndarray], format: str = "JPEG") -> str:
    """Encodes PIL Image or NumPy array into a base64 data URL for web transport."""
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img.astype(np.uint8))
    else:
        pil_img = img

    buf = io.BytesIO()
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format=format, quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"



# ----------------------------------------------------------------------------
# 1. GET /api/health
# ----------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, summary="Service Health & Model Readiness")
def get_health() -> HealthResponse:
    """
    Returns API health status, server timestamp, available checkpoint count,
    and platform model assignments (VGG16 for Web, MobileNetV2 for Mobile).
    """
    statuses = get_all_models_status()
    available_count = sum(1 for s in statuses if s["is_available"])
    
    return HealthResponse(
        status="ok",
        app_name="RetinaCare AI Backend",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        available_models_count=available_count,
        total_models_count=len(CANONICAL_MODEL_NAMES),
        web_model=WEB_MODEL_NAME,
        mobile_model=MOBILE_MODEL_NAME
    )


# ----------------------------------------------------------------------------
# 2. GET /api/models
# ----------------------------------------------------------------------------
@app.get("/api/models", response_model=ModelsListResponse, summary="List All Supported Deep Learning Architectures")
def list_models() -> ModelsListResponse:
    """
    Returns registered neural architectures, input dimensions, model weights status,
    platform allocations, and checkpoint presence without any fabrication.
    """
    statuses = get_all_models_status()
    model_infos: list[ModelInfo] = []

    for s in statuses:
        model_infos.append(
            ModelInfo(
                canonical_name=s["canonical_name"],
                display_name=s["display_name"],
                architecture_type=s["architecture_type"],
                backbone=s["backbone"],
                training_type=s["training_type"],
                input_resolution_str=s["input_resolution_str"],
                classes=s["classes"],
                dataset=s["dataset"],
                platform=s.get("platform"),
                target=s.get("target"),
                is_available=s["is_available"],
                status_label=s["status_label"],
                checkpoint_path=s["checkpoint_path"],
                file_size_mb=s["file_size_mb"],
                training_command=s["training_command"]
            )
        )

    available_count = sum(1 for m in model_infos if m.is_available)
    return ModelsListResponse(
        models=model_infos,
        count=len(model_infos),
        available_count=available_count
    )


async def _execute_prediction_pipeline(
    file: UploadFile,
    target_model_name: str,
    confidence_threshold: float,
    include_preprocessed_image: bool
) -> PredictionResponse:
    """
    Standardized inference pipeline:
    1. Validates file buffer, file size, and extension
    2. Resolves canonical model identifier and verifies checkpoint presence
    3. Executes model-specific inference in worker threadpool
    4. Formats ICDR 5-stage clinical classification and probabilities
    """
    # Read file buffer
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file buffer: {str(exc)}"
        )

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file buffer is empty (0 bytes)."
        )

    # Validate file size
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
        )

    # Validate file extension if filename provided
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    # Validate model selection
    canon_name = resolve_model_key(target_model_name)
    if canon_name is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model '{target_model_name}'. Supported models: {CANONICAL_MODEL_NAMES}"
        )

    # Check model availability upfront
    st_info = get_model_status(canon_name)
    if not st_info["is_available"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{st_info['display_name']}' is not available on disk. Checkpoint missing. Train with: {st_info['training_command']}"
        )

    # Execute inference in worker threadpool to prevent blocking the async event loop
    try:
        pred_dict = await run_in_threadpool(
            run_model_inference,
            model_name=canon_name,
            image_input=contents,
            confidence_threshold=confidence_threshold
        )
    except ImageValidationError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err)
        )
    except MissingModelError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(err)
        )
    except ModelCompatibilityError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model loading or compatibility error: {str(err)}"
        )
    except ClassMappingVerificationError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model class mapping verification failed: {str(err)}"
        )
    except Exception as err:
        logger.exception("Inference anomaly occurred")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(err)}"
        )

    prep_b64 = None
    if include_preprocessed_image and "preprocessed_display" in pred_dict:
        try:
            prep_b64 = _pil_to_base64_data_url(pred_dict["preprocessed_display"])
        except Exception:
            prep_b64 = None

    return PredictionResponse(
        model_name=pred_dict["model_name"],
        model_id=pred_dict["model_id"],
        architecture_type=pred_dict["architecture_type"],
        backbone=pred_dict["backbone"],
        training_type=pred_dict["training_type"],
        dataset_name=pred_dict["dataset_name"],
        class_id=pred_dict["class_id"],
        class_name=pred_dict["class_name"],
        short_name=pred_dict["short_name"],
        confidence_pct=round(pred_dict["confidence_pct"], 2),
        latency_ms=round(pred_dict["latency_ms"], 1),
        is_low_confidence=pred_dict["is_low_confidence"],
        confidence_status=pred_dict["confidence_status"],
        confidence_threshold=pred_dict["confidence_threshold"],
        probabilities={k: round(v, 2) for k, v in pred_dict["probabilities"].items()},
        raw_probabilities=pred_dict["raw_probabilities"],
        description=pred_dict["description"],
        recommendation=pred_dict["recommendation"],
        severity_color=pred_dict["severity_color"],
        disclaimer=pred_dict["disclaimer"],
        retinal_warning=pred_dict["retinal_warning"],
        timestamp=pred_dict["timestamp"],
        preprocessed_image_base64=prep_b64
    )


# ----------------------------------------------------------------------------
# 3. POST /api/predict/web (Dedicated Website Route -> VGG16)
# ----------------------------------------------------------------------------
@app.post("/api/predict/web", response_model=PredictionResponse, summary="Execute Dedicated Website Retinal Screening (VGG16)")
async def predict_retinopathy_web(
    file: UploadFile = File(..., description="Digital color retinal fundus image (PNG, JPG, BMP)"),
    confidence_threshold: float = Form(CONFIDENCE_THRESHOLD_PCT, description="Confidence threshold percentage"),
    include_preprocessed_image: bool = Form(False, description="Whether to include base64 preprocessed image")
) -> PredictionResponse:
    """
    Dedicated website screening route. Exclusively uses the VGG16 deep transfer architecture.
    Ensures that all website prediction requests use VGG16 with full multi-scale receptive fields.
    """
    return await _execute_prediction_pipeline(
        file=file,
        target_model_name=WEB_MODEL_NAME,
        confidence_threshold=confidence_threshold,
        include_preprocessed_image=include_preprocessed_image
    )


# ----------------------------------------------------------------------------
# 4. POST /api/predict/mobile (Dedicated Mobile Route -> MobileNetV2)
# ----------------------------------------------------------------------------
@app.post("/api/predict/mobile", response_model=PredictionResponse, summary="Execute Dedicated Mobile App Retinal Screening (MobileNetV2)")
async def predict_retinopathy_mobile(
    file: UploadFile = File(..., description="Digital color retinal fundus image (PNG, JPG, BMP)"),
    confidence_threshold: float = Form(CONFIDENCE_THRESHOLD_PCT, description="Confidence threshold percentage"),
    include_preprocessed_image: bool = Form(False, description="Whether to include base64 preprocessed image")
) -> PredictionResponse:
    """
    Dedicated mobile app screening route. Exclusively uses MobileNetV2 for lightweight,
    low-latency on-device inference in the Flutter Android mobile application.
    """
    return await _execute_prediction_pipeline(
        file=file,
        target_model_name=MOBILE_MODEL_NAME,
        confidence_threshold=confidence_threshold,
        include_preprocessed_image=include_preprocessed_image
    )


# ----------------------------------------------------------------------------
# 5. POST /api/predict (Standard Route with Platform & Model Routing)
# ----------------------------------------------------------------------------
@app.post("/api/predict", response_model=PredictionResponse, summary="Execute Retinal Screening with Architectural Routing")
async def predict_retinopathy(
    file: UploadFile = File(..., description="Digital color retinal fundus image (PNG, JPG, BMP)"),
    model_name: Optional[str] = Form(None, description="Requested model (e.g. VGG16, MobileNetV2)"),
    client_type: Optional[str] = Form(None, description="Client platform: 'web' (routes to VGG16) or 'mobile' (routes to MobileNetV2)"),
    confidence_threshold: float = Form(CONFIDENCE_THRESHOLD_PCT, description="Confidence threshold percentage"),
    include_preprocessed_image: bool = Form(False, description="Whether to include base64 preprocessed image")
) -> PredictionResponse:
    """
    Routes retinal prediction requests based on platform constraints:
    - Website requests (client_type='web' or default) route strictly to VGG16.
    - Mobile requests (client_type='mobile' or model_name='MobileNetV2') route to MobileNetV2.
    - Enforces architectural separation: MobileNetV2 cannot be used for website predictions.
    """
    selected_client = (client_type or "").strip().lower()
    clean_model = None
    if model_name:
        clean_model = resolve_model_key(model_name)
        if clean_model is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model '{model_name}'. Supported models: {CANONICAL_MODEL_NAMES}"
            )

    # Enforce website exclusivity: website must never use MobileNetV2
    if selected_client == "web" and clean_model == "MobileNetV2":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MobileNetV2 is reserved exclusively for the Flutter mobile application. Website screening uses VGG16."
        )

    if selected_client == "mobile" or clean_model == "MobileNetV2":
        chosen_model = MOBILE_MODEL_NAME
    elif selected_client == "web" or clean_model == "VGG16":
        chosen_model = WEB_MODEL_NAME
    elif clean_model is not None:
        chosen_model = clean_model
    else:
        chosen_model = DEFAULT_MODEL_NAME

    return await _execute_prediction_pipeline(
        file=file,
        target_model_name=chosen_model,
        confidence_threshold=confidence_threshold,
        include_preprocessed_image=include_preprocessed_image
    )


# ----------------------------------------------------------------------------
# 4. POST /api/compare
# ----------------------------------------------------------------------------
@app.post("/api/compare", response_model=ModelComparisonResponse, summary="Compare Retinal Predictions Across All Available Models")
async def compare_models(
    file: UploadFile = File(..., description="Digital color retinal fundus image (PNG, JPG, BMP)")
) -> ModelComparisonResponse:
    """
    Evaluates the input retinal image across all available trained deep learning models
    simultaneously, reporting consensus agreement and per-model diagnostics.
    """
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file buffer: {str(exc)}"
        )

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file buffer is empty (0 bytes)."
        )

    # Validate file size
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
        )


    # Validate file extension if filename provided
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    try:
        raw_results = await run_in_threadpool(compare_all_available_models, contents)
        agreement_dict = await run_in_threadpool(compute_model_agreement, raw_results)
    except ImageValidationError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err)
        )
    except Exception as exc:
        logger.exception("Comparison pipeline error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison pipeline execution failed: {str(exc)}"
        )

    ranking_dict = await run_in_threadpool(validate_and_rank_model_predictions, raw_results)

    def _to_comparison_item(r: Dict[str, Any]) -> ModelComparisonItem:
        probs = None
        if r.get("probabilities") is not None:
            probs = {k: round(v, 2) for k, v in r["probabilities"].items()}
        raw_probs = None
        if r.get("raw_probabilities") is not None:
            raw_probs = [round(float(v), 4) for v in r["raw_probabilities"]]

        return ModelComparisonItem(
            model_name=r["model_name"],
            canonical_name=r.get("canonical_name"),
            model_id=r.get("model_id"),
            architecture_type=r.get("architecture_type"),
            backbone=r.get("backbone"),
            training_type=r.get("training_type"),
            dataset_name=r.get("dataset_name", "APTOS 2019"),
            status=r["status"],
            prediction=r["prediction"],
            class_name=r.get("class_name") or r.get("predicted_class_name") or (r["prediction"] if r["status"] == "Ready" else None),
            predicted_class_name=r.get("predicted_class_name") or r.get("class_name") or (r["prediction"] if r["status"] == "Ready" else None),
            short_name=r.get("short_name"),
            class_id=r.get("class_id"),
            predicted_class_index=r.get("predicted_class_index", r.get("class_id")),
            confidence_pct=round(r["confidence_pct"], 2) if r.get("confidence_pct") is not None else None,
            confidence=round(r["confidence"], 4) if r.get("confidence") is not None else None,
            is_highest_confidence=bool(r.get("is_highest_confidence", False)),
            is_low_confidence=r.get("is_low_confidence"),
            confidence_status=r.get("confidence_status"),
            confidence_threshold=r.get("confidence_threshold", CONFIDENCE_THRESHOLD_PCT),
            probabilities=probs,
            raw_probabilities=raw_probs,
            latency_ms=round(r["latency_ms"], 1) if r.get("latency_ms") is not None else None,
            description=r.get("description"),
            recommendation=r.get("recommendation"),
            severity_color=r.get("severity_color"),
            disclaimer=r.get("disclaimer"),
            retinal_warning=r.get("retinal_warning"),
            timestamp=r.get("timestamp"),
            error_msg=r.get("error_msg")
        )

    best_item = _to_comparison_item(ranking_dict["primary_result"]) if ranking_dict["primary_result"] else None
    alt_items = [_to_comparison_item(r) for r in ranking_dict["alternative_results"]]
    failed_items = [_to_comparison_item(r) for r in ranking_dict["unranked_or_failed"]]
    all_sorted_items = ([best_item] if best_item else []) + alt_items + failed_items

    agreement_info = ModelAgreementInfo(
        available_count=agreement_dict["available_count"],
        total_count=agreement_dict["total_count"],
        agreement_class_name=agreement_dict["agreement_class_name"],
        agreeing_count=agreement_dict["agreeing_count"],
        agreement_pct=round(agreement_dict["agreement_pct"], 2),
        agreement_label=agreement_dict["agreement_label"]
    )

    return ModelComparisonResponse(
        results=all_sorted_items,
        best_model=best_item,
        alternative_models=alt_items,
        unranked_or_failed=failed_items,
        agreement=agreement_info,
        total_evaluated=ranking_dict["total_evaluated"],
        successful_count=ranking_dict["successful_count"]
    )


# ----------------------------------------------------------------------------
# 6. GET /api/model-metrics
# ----------------------------------------------------------------------------
@app.get("/api/model-metrics", response_model=MetricsResponse, summary="Verified Offline Benchmarking & Performance Telemetry")
def get_model_metrics() -> MetricsResponse:
    """
    Returns authentic empirical benchmark metrics loaded from results/model_comparison.json,
    results/vgg16_metrics.json (Website model), and results/metrics.json (Mobile model).
    """
    comparison_data: Dict[str, Any] = {}
    if MODEL_COMPARISON_JSON_PATH.is_file():
        try:
            with open(MODEL_COMPARISON_JSON_PATH, "r", encoding="utf-8") as f:
                comparison_data = json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to read model_comparison.json: {exc}")

    vgg16_metrics: Optional[Dict[str, Any]] = None
    vgg_path = PROJECT_ROOT / "results" / "vgg16_metrics.json"
    if vgg_path.is_file():
        try:
            with open(vgg_path, "r", encoding="utf-8") as f:
                vgg16_metrics = json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to read vgg16_metrics.json: {exc}")

    mobilenet_metrics: Optional[Dict[str, Any]] = None
    if METRICS_JSON_PATH.is_file():
        try:
            with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                mobilenet_metrics = json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to read metrics.json: {exc}")

    # Primary model metrics for website is VGG16
    primary_metrics = vgg16_metrics or mobilenet_metrics

    return MetricsResponse(
        model_comparison=comparison_data,
        primary_model_metrics=primary_metrics,
        web_model_metrics=vgg16_metrics,
        mobile_model_metrics=mobilenet_metrics
    )


FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.is_dir():
    css_dir = FRONTEND_DIR / "css"
    js_dir = FRONTEND_DIR / "js"
    assets_dir = FRONTEND_DIR / "assets"
    if css_dir.is_dir():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.is_dir():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


# Root endpoint: serves web frontend when requested by browser, or JSON health telemetry
@app.get("/", include_in_schema=False)
def root_redirect(request: Request):
    accept = request.headers.get("accept", "")
    frontend_index = FRONTEND_DIR / "index.html"
    if "text/html" in accept and frontend_index.is_file():
        return FileResponse(frontend_index)
    return {
        "service": "RetinaCare AI Clinical Backend",
        "status": "online",
        "docs": "/docs",
        "health": "/api/health",
        "frontend": "/"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)

