"""
==============================================================================
RetinaCare AI - Multi-Model Deep Learning Manager
Department of Computer Science & Engineering - GM University

Centralizes registration, status discovery, safe cached loading, and
preprocessing specifications across the 5 Deep Learning architectures:
1. Custom CNN
2. VGG16
3. ResNet50
4. MobileNetV2
5. EfficientNetB0
==============================================================================
"""

from __future__ import annotations
import gc
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np

# Suppress verbose TensorFlow runtime logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Project root resolution & external libraries
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_PYLIB = Path("E:/pylib")
if _PYLIB.exists() and str(_PYLIB) not in sys.path:
    sys.path.insert(0, str(_PYLIB))

from src.class_mapping import (
    DATASET_NAME,
    NUM_CLASSES,
    INPUT_RESOLUTION_STR,
    DEFAULT_INPUT_RESOLUTION
)

logger = logging.getLogger("RetinaCare.ModelManager")


class ModelManagerError(RuntimeError):
    """Base exception for model manager failures."""
    pass


class MissingModelError(ModelManagerError):
    """Raised when an inference is requested on an unavailable/untrained model."""
    def __init__(self, model_key: str, display_name: str):
        super().__init__(
            f"Selected model '{display_name}' is currently unavailable. "
            f"This model hasn't been made available for prediction yet. "
            f"Please choose another trained model or train this model first."
        )
        self.model_key = model_key
        self.display_name = display_name


class ModelCompatibilityError(ModelManagerError):
    """Raised when a model checkpoint cannot be loaded due to architectural/weight incompatibility."""
    def __init__(self, model_key: str, display_name: str, technical_error: str):
        super().__init__(
            f"Unable to load '{display_name}'. The model configuration or weights may be incompatible."
        )
        self.model_key = model_key
        self.display_name = display_name
        self.technical_error = technical_error


# Canonical 5-Model Registry
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Custom CNN": {
        "id": "custom_cnn",
        "display_name": "Custom CNN",
        "aliases": ["CustomCNN", "Custom CNN", "cnn"],
        "architecture_type": "Convolutional Neural Network (From Scratch)",
        "backbone": "4-Stage Deep ConvNet Baseline",
        "training_type": "Trained from Scratch (Academic Baseline)",
        "input_size": (224, 224),
        "input_resolution_str": "224 × 224",
        "classes": NUM_CLASSES,
        "dataset": DATASET_NAME,
        "platform": "Academic Baseline",
        "target": "baseline",
        "checkpoint_candidates": [
            PROJECT_ROOT / "models" / "custom_cnn.keras",
            PROJECT_ROOT / "models" / "cnn" / "model.keras",
            PROJECT_ROOT / "models" / "custom_cnn.h5"
        ],
        "training_command": "python src/train.py --model CustomCNN --epochs 10 --no-fine-tune",
        "description": "A 4-stage convolutional neural network trained directly from random weight initialization as an empirical baseline."
    },
    "VGG16": {
        "id": "vgg16",
        "display_name": "VGG16",
        "aliases": ["VGG16", "vgg16", "VGG-16"],
        "architecture_type": "Deep Transfer Learning",
        "backbone": "VGG16 (Visual Geometry Group 16-Layer)",
        "training_type": "Transfer Learning (ImageNet Initialized)",
        "input_size": (224, 224),
        "input_resolution_str": "224 × 224",
        "classes": NUM_CLASSES,
        "dataset": DATASET_NAME,
        "platform": "Website",
        "target": "web",
        "checkpoint_candidates": [
            PROJECT_ROOT / "models" / "vgg16.keras",
            PROJECT_ROOT / "models" / "vgg16" / "model.keras",
            PROJECT_ROOT / "models" / "vgg16.h5"
        ],
        "training_command": "python src/train.py --model VGG16 --epochs 15 --fine-tune-epochs 10",
        "description": "Classic 16-layer architecture using uniform 3×3 convolutions for foundational multi-scale retinal feature representation. Dedicated to website clinical screening."
    },
    "ResNet50": {
        "id": "resnet50",
        "display_name": "ResNet50",
        "aliases": ["ResNet50", "resnet50", "ResNet-50"],
        "architecture_type": "Residual Transfer Learning",
        "backbone": "ResNet50 (50-Layer Deep Residual Network)",
        "training_type": "Transfer Learning (ImageNet Initialized)",
        "input_size": (224, 224),
        "input_resolution_str": "224 × 224",
        "classes": NUM_CLASSES,
        "dataset": DATASET_NAME,
        "platform": "Academic Benchmark",
        "target": "benchmark",
        "checkpoint_candidates": [
            PROJECT_ROOT / "models" / "resnet50.keras",
            PROJECT_ROOT / "models" / "resnet50" / "model.keras",
            PROJECT_ROOT / "models" / "resnet50.h5"
        ],
        "training_command": "python src/train.py --model ResNet50 --epochs 15 --fine-tune-epochs 10",
        "description": "Deep 50-layer residual network utilizing identity shortcut connections to mitigate vanishing gradients in fine microvascular feature learning."
    },
    "MobileNetV2": {
        "id": "mobilenetv2",
        "display_name": "MobileNetV2",
        "aliases": ["MobileNetV2", "mobilenetv2", "MobileNet-V2"],
        "architecture_type": "Lightweight Mobile Transfer Learning",
        "backbone": "MobileNetV2 (Inverted Residuals & Linear Bottlenecks)",
        "training_type": "Transfer Learning (ImageNet Initialized + Fine-Tuned)",
        "input_size": (224, 224),
        "input_resolution_str": "224 × 224",
        "classes": NUM_CLASSES,
        "dataset": DATASET_NAME,
        "platform": "Mobile Application",
        "target": "mobile",
        "checkpoint_candidates": [
            PROJECT_ROOT / "models" / "mobilenetv2.keras",
            PROJECT_ROOT / "models" / "mobilenetv2" / "model.keras",
            PROJECT_ROOT / "models" / "best_model.keras",
            PROJECT_ROOT / "models" / "best_model.h5"
        ],
        "training_command": "python src/train.py --model MobileNetV2 --epochs 15 --fine-tune-epochs 10",
        "description": "High-efficiency mobile architecture with depthwise separable convolutions optimized for low latency and high accuracy. Reserved exclusively for Flutter Android application."
    },
    "EfficientNetB0": {
        "id": "efficientnetb0",
        "display_name": "EfficientNetB0",
        "aliases": ["EfficientNetB0", "efficientnetb0", "EfficientNet-B0"],
        "architecture_type": "Compound Scaling Transfer Learning",
        "backbone": "EfficientNetB0 (Compound Scaled ConvNet)",
        "training_type": "Transfer Learning (Compound Scaled)",
        "input_size": (224, 224),
        "input_resolution_str": "224 × 224",
        "classes": NUM_CLASSES,
        "dataset": DATASET_NAME,
        "platform": "Academic Benchmark",
        "target": "benchmark",
        "checkpoint_candidates": [
            PROJECT_ROOT / "models" / "efficientnetb0.keras",
            PROJECT_ROOT / "models" / "efficientnetb0" / "model.keras",
            PROJECT_ROOT / "models" / "efficientnetb0.h5"
        ],
        "training_command": "python src/train.py --model EfficientNetB0 --epochs 15 --fine-tune-epochs 10",
        "description": "State-of-the-art compound scaled architecture simultaneously optimizing network depth, width, and resolution for maximum parameter efficiency."
    }
}

# The strictly supported 5 model keys in canonical order
CANONICAL_MODEL_NAMES: List[str] = [
    "Custom CNN",
    "VGG16",
    "ResNet50",
    "MobileNetV2",
    "EfficientNetB0"
]


def resolve_model_key(model_identifier: str) -> Optional[str]:
    """Resolves any variant/alias of model name to its canonical key."""
    if not model_identifier:
        return None
    cleaned = model_identifier.strip().lower().replace("-", "").replace(" ", "").replace("_", "")
    for canon_name, meta in MODEL_REGISTRY.items():
        c_clean = canon_name.lower().replace("-", "").replace(" ", "").replace("_", "")
        if cleaned == c_clean:
            return canon_name
        for alias in meta["aliases"]:
            a_clean = alias.lower().replace("-", "").replace(" ", "").replace("_", "")
            if cleaned == a_clean:
                return canon_name
    return None


def get_model_metadata(model_key: str) -> Dict[str, Any]:
    """Retrieves full metadata dictionary for the specified model."""
    canonical = resolve_model_key(model_key)
    if canonical is None or canonical not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model identifier '{model_key}'. Choose from: {CANONICAL_MODEL_NAMES}")
    return MODEL_REGISTRY[canonical]


def locate_model_checkpoint(model_key: str) -> Optional[Path]:
    """
    Checks all candidate paths for a model to find a real checkpoint on disk.
    Returns Path if found, otherwise None.
    """
    meta = get_model_metadata(model_key)
    for candidate in meta["checkpoint_candidates"]:
        if candidate.is_file() and candidate.stat().st_size > 1024:
            return candidate
    return None


def get_model_status(model_key: str) -> Dict[str, Any]:
    """
    Returns authentic availability status for the model without faking results.
    """
    canonical = resolve_model_key(model_key)
    if canonical is None:
        return {
            "canonical_name": model_key,
            "is_available": False,
            "status_label": "UNKNOWN",
            "checkpoint_path": None,
            "file_size_mb": 0.0
        }

    meta = MODEL_REGISTRY[canonical]
    ckpt_path = locate_model_checkpoint(canonical)
    is_avail = ckpt_path is not None

    return {
        "canonical_name": canonical,
        "display_name": meta["display_name"],
        "architecture_type": meta["architecture_type"],
        "backbone": meta["backbone"],
        "training_type": meta["training_type"],
        "input_resolution_str": meta["input_resolution_str"],
        "classes": meta["classes"],
        "dataset": meta["dataset"],
        "platform": meta.get("platform", "Benchmark"),
        "target": meta.get("target", "all"),
        "is_available": is_avail,
        "status_label": "READY" if is_avail else "NOT TRAINED",
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        "file_size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2) if is_avail else 0.0,
        "training_command": meta["training_command"]
    }


def get_all_models_status() -> List[Dict[str, Any]]:
    """Returns availability status for all 5 models."""
    return [get_model_status(name) for name in CANONICAL_MODEL_NAMES]


# Internal memory cache for loaded models
_LOADED_MODELS_CACHE: Dict[str, Any] = {}
_ACTIVE_MODEL_KEY: Optional[str] = None


def load_model_instance(model_key: str) -> Any:
    """
    Loads and caches only the selected deep learning model.
    Releases previously active model resources to conserve system memory.
    
    Raises:
        MissingModelError: If model file does not exist on disk.
        ModelCompatibilityError: If Keras fails to load weights/architecture.
    """
    global _ACTIVE_MODEL_KEY, _LOADED_MODELS_CACHE

    canonical = resolve_model_key(model_key)
    if canonical is None:
        raise ValueError(f"Invalid model name '{model_key}'.")

    meta = get_model_metadata(canonical)
    ckpt = locate_model_checkpoint(canonical)

    if ckpt is None:
        raise MissingModelError(canonical, meta["display_name"])

    # If already cached and valid, return it directly
    if canonical in _LOADED_MODELS_CACHE and _LOADED_MODELS_CACHE[canonical] is not None:
        _ACTIVE_MODEL_KEY = canonical
        return _LOADED_MODELS_CACHE[canonical]

    # Memory management: retain primary active production models in cache (VGG16 for web, MobileNetV2 for mobile)
    # Evict only temporary/benchmark architectures if needed to conserve memory
    PROTECTED_PRODUCTION_MODELS = {"VGG16", "MobileNetV2"}
    to_remove = [k for k in _LOADED_MODELS_CACHE.keys() if k not in PROTECTED_PRODUCTION_MODELS and k != canonical]
    for k in to_remove:
        del _LOADED_MODELS_CACHE[k]
    if to_remove:
        gc.collect()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ModelCompatibilityError(canonical, meta["display_name"], f"TensorFlow import failed: {exc}")

    try:
        logger.info(f"Loading model '{canonical}' from: {ckpt}")
        loaded_model = tf.keras.models.load_model(str(ckpt), compile=False)
        _LOADED_MODELS_CACHE[canonical] = loaded_model
        _ACTIVE_MODEL_KEY = canonical
        return loaded_model
    except Exception as exc:
        technical_err = str(exc)
        logger.error(f"Failed to load checkpoint {ckpt} for {canonical}: {technical_err}")
        raise ModelCompatibilityError(canonical, meta["display_name"], technical_err) from exc
