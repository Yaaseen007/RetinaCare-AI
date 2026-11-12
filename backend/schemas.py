"""
==============================================================================
RetinaCare AI - FastAPI Data Schemas
Pydantic Request & Response Models for Clinical Telemetry & Multi-Model Inference
==============================================================================
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status indicator ('ok' or 'healthy')")
    app_name: str = Field("RetinaCare AI Backend", description="Name of the backend service")
    version: str = Field("1.0.0", description="API version")
    timestamp: str = Field(..., description="Server timestamp (ISO format)")
    available_models_count: int = Field(..., description="Number of ready model checkpoints")
    total_models_count: int = Field(..., description="Total supported architectures")
    web_model: str = Field("VGG16", description="Assigned deep learning model for website predictions")
    mobile_model: str = Field("MobileNetV2", description="Assigned deep learning model for mobile application predictions")


class ModelInfo(BaseModel):
    canonical_name: str = Field(..., description="Canonical model identifier")
    display_name: str = Field(..., description="Human-readable model name")
    architecture_type: str = Field(..., description="Architectural family")
    backbone: str = Field(..., description="Neural network backbone")
    training_type: str = Field(..., description="Training methodology")
    input_resolution_str: str = Field("224 × 224", description="Expected input resolution")
    classes: int = Field(5, description="Number of output disease stages")
    dataset: str = Field("APTOS 2019", description="Dataset provenance")
    platform: Optional[str] = Field(None, description="Assigned platform: 'Website', 'Mobile Application', or 'Academic Benchmark'")
    target: Optional[str] = Field(None, description="Target routing slug ('web', 'mobile', or 'benchmark')")
    is_available: bool = Field(..., description="Whether .keras checkpoint exists and is ready")
    status_label: str = Field(..., description="'READY' or 'NOT TRAINED'")
    checkpoint_path: Optional[str] = Field(None, description="Local checkpoint path if available")
    file_size_mb: float = Field(0.0, description="Checkpoint size in megabytes")
    training_command: str = Field(..., description="Command to train this model")


class ModelsListResponse(BaseModel):
    models: List[ModelInfo] = Field(..., description="List of all registered models")
    count: int = Field(..., description="Total number of registered models")
    available_count: int = Field(..., description="Number of trained and ready models")


class PredictionResponse(BaseModel):
    model_name: str = Field(..., description="Name of deep learning model used")
    model_id: str = Field(..., description="Model identifier slug")
    architecture_type: str = Field(..., description="Architecture category")
    backbone: str = Field(..., description="Underlying neural backbone")
    training_type: str = Field(..., description="Training regime")
    dataset_name: str = Field("APTOS 2019", description="Training dataset name")
    class_id: int = Field(..., description="Predicted ICDR stage (0-4)")
    class_name: str = Field(..., description="Predicted disease severity label")
    short_name: str = Field(..., description="Concise stage representation")
    confidence_pct: float = Field(..., description="Calibrated top-1 confidence score (0-100%)")
    latency_ms: float = Field(..., description="Forward pass latency in milliseconds")
    is_low_confidence: bool = Field(..., description="True if confidence is below threshold")
    confidence_status: str = Field(..., description="Clinical interpretation of confidence")
    confidence_threshold: float = Field(50.0, description="Applied confidence cutoff threshold")
    probabilities: Dict[str, float] = Field(..., description="Softmax probabilities for each of the 5 classes")
    raw_probabilities: List[float] = Field(..., description="Raw probability array [0, 1]")
    description: str = Field(..., description="Clinical biomarker explanation")
    recommendation: str = Field(..., description="Standard clinical follow-up advice")
    severity_color: str = Field(..., description="Hex color representation for UI")
    disclaimer: str = Field(..., description="Medical research disclaimer")
    retinal_warning: Optional[str] = Field(None, description="Quality/plausibility warning if detected")
    timestamp: str = Field(..., description="Execution timestamp")
    preprocessed_image_base64: Optional[str] = Field(None, description="Base64 data URL of preprocessed retinal image")


class ModelComparisonItem(BaseModel):
    model_name: str = Field(..., description="Model display name")
    canonical_name: Optional[str] = Field(None, description="Canonical model identifier")
    model_id: Optional[str] = Field(None, description="Model identifier slug")
    architecture_type: Optional[str] = Field(None, description="Architectural family")
    backbone: Optional[str] = Field(None, description="Neural network backbone")
    training_type: Optional[str] = Field(None, description="Training methodology")
    dataset_name: Optional[str] = Field("APTOS 2019", description="Dataset provenance")
    status: str = Field(..., description="'Ready', 'Not Available', or 'Error'")
    prediction: str = Field(..., description="Predicted stage name or status")
    class_name: Optional[str] = Field(None, description="Predicted disease severity label")
    predicted_class_name: Optional[str] = Field(None, description="Predicted disease severity label")
    short_name: Optional[str] = Field(None, description="Short stage string")
    class_id: Optional[int] = Field(None, description="Predicted class index (0-4)")
    predicted_class_index: Optional[int] = Field(None, description="Predicted class index (0-4)")
    confidence_pct: Optional[float] = Field(None, description="Top-1 confidence score (0-100%)")
    confidence: Optional[float] = Field(None, description="Top-1 confidence score (0.0-1.0)")
    is_highest_confidence: bool = Field(False, description="True if this is the highest confidence prediction")
    is_low_confidence: Optional[bool] = Field(None, description="True if confidence is below threshold")
    confidence_status: Optional[str] = Field(None, description="Clinical interpretation of confidence")
    confidence_threshold: float = Field(50.0, description="Applied confidence cutoff threshold")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities (percentage)")
    raw_probabilities: Optional[List[float]] = Field(None, description="Raw normalized probabilities (0-1)")
    latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")
    description: Optional[str] = Field(None, description="Clinical biomarker explanation")
    recommendation: Optional[str] = Field(None, description="Standard clinical follow-up advice")
    severity_color: Optional[str] = Field(None, description="Hex color representation for UI")
    disclaimer: Optional[str] = Field(None, description="Medical research disclaimer")
    retinal_warning: Optional[str] = Field(None, description="Retinal quality warning")
    timestamp: Optional[str] = Field(None, description="Execution timestamp")
    error_msg: Optional[str] = Field(None, description="Error explanation if unavailable or failed")


class ModelAgreementInfo(BaseModel):
    available_count: int = Field(..., description="Number of models evaluated")
    total_count: int = Field(..., description="Total architectures compared")
    agreement_class_name: Optional[str] = Field(None, description="Majority consensus class name")
    agreeing_count: int = Field(..., description="Number of models agreeing with majority")
    agreement_pct: float = Field(..., description="Consensus percentage")
    agreement_label: str = Field(..., description="Human-readable consensus statement")


class ModelComparisonResponse(BaseModel):
    results: List[ModelComparisonItem] = Field(..., description="Per-model predictions sorted by descending confidence")
    best_model: Optional[ModelComparisonItem] = Field(None, description="Primary result with highest softmax confidence")
    alternative_models: List[ModelComparisonItem] = Field(default_factory=list, description="Alternative model predictions")
    unranked_or_failed: List[ModelComparisonItem] = Field(default_factory=list, description="Unavailable or failed models")
    agreement: ModelAgreementInfo = Field(..., description="Cross-model consensus analysis")
    total_evaluated: int = Field(0, description="Total models evaluated")
    successful_count: int = Field(0, description="Number of successfully evaluated models")


class MetricsResponse(BaseModel):
    model_comparison: Dict[str, Any] = Field(..., description="Comparative metrics across backbones")
    primary_model_metrics: Optional[Dict[str, Any]] = Field(None, description="In-depth test set metrics for primary website model (VGG16)")
    web_model_metrics: Optional[Dict[str, Any]] = Field(None, description="Authentic evaluation metrics for VGG16 (Website)")
    mobile_model_metrics: Optional[Dict[str, Any]] = Field(None, description="Authentic evaluation metrics for MobileNetV2 (Flutter Mobile App)")
