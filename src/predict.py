"""
==============================================================================
Diabetic Retinopathy Detection - Inference & Prediction Engine (Compatibility Layer)
Department of Computer Science & Engineering - GM University

Maintains backward-compatibility with existing code while delegating to the
multi-model architecture in src/prediction.py and src/model_manager.py.
==============================================================================
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image

from src.class_mapping import (
    CLASS_NAMES,
    CLASS_SHORT_NAMES,
    CLASS_DESCRIPTIONS,
    CLASS_RECOMMENDATIONS,
    CLASS_COLORS,
    NUM_CLASSES,
    MEDICAL_DISCLAIMER
)
from src.validation import (
    inspect_and_decode_image,
    ImageValidationError,
    CorruptedImageError
)
from src.preprocessing import (
    preprocess_retinal_image,
    get_display_pair,
    InvalidImageError
)
from src.model_manager import (
    get_model_status,
    load_model_instance,
    MissingModelError,
    ModelCompatibilityError
)
from src.prediction import (
    run_model_inference,
    compare_all_available_models,
    compute_model_agreement,
    CONFIDENCE_THRESHOLD_PCT
)


class ModelNotLoadedError(RuntimeError):
    """Raised when an inference attempt is made but no trained model is available."""
    pass


class PredictionError(RuntimeError):
    """Raised when an error occurs during model evaluation."""
    pass


class RetinaPredictor:
    """
    Backwards-compatible inference controller.
    Defaults to 'MobileNetV2' (the primary trained backbone) or accepts any of the 5 models.
    """

    def __init__(self, model_name: str = "MobileNetV2", model_path: Optional[Union[str, Path]] = None):
        self.model_name = model_name
        self.model_path = Path(model_path) if model_path else None

    def is_model_loaded(self) -> bool:
        """Returns True if the selected model checkpoint exists and is loadable."""
        status = get_model_status(self.model_name)
        return status["is_available"]

    def _load_model_safe(self) -> None:
        """Pre-warms the model cache."""
        try:
            load_model_instance(self.model_name)
        except Exception:
            pass

    def get_model_status(self) -> Tuple[bool, str]:
        """Provides a user-friendly status report."""
        status = get_model_status(self.model_name)
        if status["is_available"]:
            return True, f"Trained {status['display_name']} loaded successfully."
        return (
            False,
            f"No trained model found for '{status['display_name']}'. "
            f"Please train the model via: {status['training_command']}"
        )

    def predict(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """Runs inference via the unified multi-model engine."""
        try:
            return run_model_inference(self.model_name, image_input)
        except MissingModelError as e:
            raise ModelNotLoadedError(str(e)) from e
        except ImageValidationError as e:
            raise InvalidImageError(str(e)) from e
        except Exception as e:
            raise PredictionError(str(e)) from e
