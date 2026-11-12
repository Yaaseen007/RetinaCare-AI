"""
==============================================================================
RetinaCare AI - Multi-Model Inference & Diagnostic Engine
Department of Computer Science & Engineering - GM University

Executes authentic, model-specific inference grounded in verified dataset
class mappings, manages low-confidence calibrations, and computes multi-model
comparison and agreement analytics.
==============================================================================
"""

from __future__ import annotations
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

# Internal modules
from src.class_mapping import (
    CLASS_NAMES,
    CLASS_SHORT_NAMES,
    CLASS_DESCRIPTIONS,
    CLASS_RECOMMENDATIONS,
    CLASS_COLORS,
    NUM_CLASSES,
    DATASET_NAME,
    MEDICAL_DISCLAIMER,
    verify_class_mapping,
    ClassMappingVerificationError
)
from src.validation import (
    inspect_and_decode_image,
    check_retinal_fundus_plausibility,
    ImageValidationError
)
from src.preprocessing import (
    preprocess_retinal_image,
    get_display_pair
)
from src.model_manager import (
    resolve_model_key,
    get_model_metadata,
    get_model_status,
    load_model_instance,
    CANONICAL_MODEL_NAMES,
    MissingModelError,
    ModelCompatibilityError
)

# Configurable clinical confidence threshold
CONFIDENCE_THRESHOLD_PCT = 50.0


class PredictionExecutionError(RuntimeError):
    """Raised when an error occurs during model evaluation."""
    pass


def run_model_inference(
    model_name: str,
    image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image],
    confidence_threshold: float = CONFIDENCE_THRESHOLD_PCT
) -> Dict[str, Any]:
    """
    Executes model-specific inference strictly through the selected AI model.
    
    Steps:
    1. Resolve canonical model identifier & verify metadata
    2. Validate and decode retinal image (raising clean exceptions on corrupted/invalid files)
    3. Load selected model weights from disk (cached via model_manager)
    4. Apply verified preprocessing pipeline (Crop -> Ben Graham -> 224x224 -> Rescale [0, 1])
    5. Execute forward pass through selected architecture
    6. Verify output dimensionality against ICDR 5-class standard
    7. Calibrate softmax probabilities & determine highest-confidence prediction
    8. Check confidence threshold (flagging low-confidence state if < threshold)
    9. Associate clinical biomarkers and recommendations
    """
    # 1. Resolve model
    canon_name = resolve_model_key(model_name)
    if canon_name is None:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {CANONICAL_MODEL_NAMES}")

    meta = get_model_metadata(canon_name)

    # 2. Validate input and decode to RGB
    rgb_arr = inspect_and_decode_image(image_input)
    is_plausible, retinal_warning = check_retinal_fundus_plausibility(rgb_arr)

    # 3. Load model (raises MissingModelError if checkpoint does not exist)
    model = load_model_instance(canon_name)

    try:
        # 4. Generate display pair and input tensor
        orig_disp, prep_disp = get_display_pair(rgb_arr, target_size=meta["input_size"])

        # Model tensor (1, 224, 224, 3) normalized to [0.0, 1.0]
        preprocessed_float = preprocess_retinal_image(
            image_input=rgb_arr,
            target_size=meta["input_size"],
            crop_circle=True,
            enhance=True,
            normalize=True
        )
        input_tensor = np.expand_dims(preprocessed_float, axis=0)

        # 5. Forward Pass
        t_infer_start = time.perf_counter()
        raw_output = model.predict(input_tensor, verbose=0)
        latency_ms = round((time.perf_counter() - t_infer_start) * 1000, 1)
        probabilities = np.asarray(raw_output[0], dtype=np.float64)

        # 6. Verify dimensionality against dataset class mapping
        verify_class_mapping(model_output_shape=raw_output.shape)

        # Ensure probabilities sum to 1.0 (apply softmax if logits returned)
        if np.min(probabilities) < 0.0 or not np.isclose(np.sum(probabilities), 1.0, atol=1e-2):
            exp_p = np.exp(probabilities - np.max(probabilities))
            probabilities = exp_p / np.sum(exp_p)

        if np.isnan(probabilities).any() or np.isinf(probabilities).any():
            raise PredictionExecutionError(f"Model '{meta['display_name']}' produced invalid numerical probabilities (NaN/Inf).")

        # 7. Extract predicted class and confidence
        predicted_class_id = int(np.argmax(probabilities))
        confidence_pct = float(probabilities[predicted_class_id] * 100.0)

        # 8. Check confidence threshold
        is_low_confidence = bool(confidence_pct < confidence_threshold)
        confidence_status = (
            f"Model confidence is relatively low ({confidence_pct:.1f}% < {confidence_threshold:.0f}%). "
            "Consider professional clinical evaluation and repeat imaging."
            if is_low_confidence
            else "Deep learning prediction derived from retinal vascular lesion patterns."
        )

        # Build full probability dictionary
        prob_dict = {
            CLASS_NAMES[i]: float(probabilities[i] * 100.0)
            for i in range(NUM_CLASSES)
        }

        # Timestamp for session history
        now_str = datetime.now().strftime("%H:%M:%S")

        return {
            "model_name": meta["display_name"],
            "model_id": meta["id"],
            "architecture_type": meta["architecture_type"],
            "backbone": meta["backbone"],
            "training_type": meta["training_type"],
            "dataset_name": meta["dataset"],
            "class_id": predicted_class_id,
            "class_name": CLASS_NAMES[predicted_class_id],
            "short_name": CLASS_SHORT_NAMES[predicted_class_id],
            "confidence_pct": confidence_pct,
            "latency_ms": latency_ms,
            "is_low_confidence": is_low_confidence,
            "confidence_status": confidence_status,
            "confidence_threshold": confidence_threshold,
            "probabilities": prob_dict,
            "raw_probabilities": probabilities.tolist(),
            "description": CLASS_DESCRIPTIONS[predicted_class_id],
            "recommendation": CLASS_RECOMMENDATIONS[predicted_class_id],
            "severity_color": CLASS_COLORS[predicted_class_id],
            "disclaimer": MEDICAL_DISCLAIMER,
            "retinal_warning": retinal_warning,
            "original_display": orig_disp,
            "preprocessed_display": prep_disp,
            "timestamp": now_str
        }

    except (ClassMappingVerificationError, MissingModelError, ModelCompatibilityError, ImageValidationError):
        raise
    except Exception as exc:
        raise PredictionExecutionError(f"Prediction execution failed on model '{meta['display_name']}': {str(exc)}") from exc


def compare_all_available_models(
    image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image]
) -> List[Dict[str, Any]]:
    """
    Executes prediction across every AVAILABLE trained model.
    Unavailable models are explicitly reported as 'Not Available' with no fabricated values.
    Returns complete per-model inference structures for all supported architectures.
    """
    # Validate and decode input image first (raises ImageValidationError if invalid/corrupt)
    decoded_arr = inspect_and_decode_image(image_input)

    results = []

    for name in CANONICAL_MODEL_NAMES:
        status_info = get_model_status(name)
        if not status_info["is_available"]:
            results.append({
                "model_name": status_info["display_name"],
                "canonical_name": name,
                "model_id": status_info.get("id", name.lower().replace("_", "")),
                "architecture_type": status_info.get("architecture_type", "Convolutional Network"),
                "backbone": status_info.get("backbone", name),
                "training_type": status_info.get("training_type", "Deep Learning"),
                "dataset_name": DATASET_NAME,
                "status": "Not Available",
                "prediction": "Not Available",
                "predicted_class_name": None,
                "class_name": None,
                "short_name": None,
                "class_id": None,
                "predicted_class_index": None,
                "confidence_pct": None,
                "confidence": None,
                "is_highest_confidence": False,
                "is_low_confidence": None,
                "confidence_status": None,
                "confidence_threshold": CONFIDENCE_THRESHOLD_PCT,
                "probabilities": None,
                "raw_probabilities": None,
                "latency_ms": None,
                "description": None,
                "recommendation": None,
                "severity_color": None,
                "disclaimer": MEDICAL_DISCLAIMER,
                "retinal_warning": None,
                "timestamp": None,
                "error_msg": f"Model checkpoint not found. Train with: {status_info['training_command']}"
            })
            continue

        # Model is available -> run actual inference with pre-decoded image
        try:
            pred = run_model_inference(model_name=name, image_input=decoded_arr)
            results.append({
                "model_name": pred["model_name"],
                "canonical_name": name,
                "model_id": pred["model_id"],
                "architecture_type": pred["architecture_type"],
                "backbone": pred["backbone"],
                "training_type": pred["training_type"],
                "dataset_name": pred["dataset_name"],
                "status": "Ready",
                "prediction": pred["class_name"],
                "predicted_class_name": pred["class_name"],
                "class_name": pred["class_name"],
                "short_name": pred["short_name"],
                "class_id": pred["class_id"],
                "predicted_class_index": pred["class_id"],
                "confidence_pct": pred["confidence_pct"],
                "confidence": round(pred["confidence_pct"] / 100.0, 4),
                "is_highest_confidence": False,
                "is_low_confidence": pred["is_low_confidence"],
                "confidence_status": pred["confidence_status"],
                "confidence_threshold": pred["confidence_threshold"],
                "probabilities": pred["probabilities"],
                "raw_probabilities": pred["raw_probabilities"],
                "latency_ms": pred["latency_ms"],
                "description": pred["description"],
                "recommendation": pred["recommendation"],
                "severity_color": pred["severity_color"],
                "disclaimer": pred["disclaimer"],
                "retinal_warning": pred["retinal_warning"],
                "timestamp": pred["timestamp"],
                "error_msg": None
            })
        except Exception as exc:
            results.append({
                "model_name": status_info["display_name"],
                "canonical_name": name,
                "model_id": status_info.get("id", name.lower().replace("_", "")),
                "architecture_type": status_info.get("architecture_type", "Convolutional Network"),
                "backbone": status_info.get("backbone", name),
                "training_type": status_info.get("training_type", "Deep Learning"),
                "dataset_name": DATASET_NAME,
                "status": "Error",
                "prediction": "Inference Failed",
                "predicted_class_name": None,
                "class_name": None,
                "short_name": None,
                "class_id": None,
                "predicted_class_index": None,
                "confidence_pct": None,
                "confidence": None,
                "is_highest_confidence": False,
                "is_low_confidence": None,
                "confidence_status": None,
                "confidence_threshold": CONFIDENCE_THRESHOLD_PCT,
                "probabilities": None,
                "raw_probabilities": None,
                "latency_ms": None,
                "description": None,
                "recommendation": None,
                "severity_color": None,
                "disclaimer": MEDICAL_DISCLAIMER,
                "retinal_warning": None,
                "timestamp": None,
                "error_msg": str(exc)
            })

    return results


def validate_probability_vector(
    raw_probabilities: Optional[List[float]],
    probabilities_dict: Optional[Dict[str, float]],
    class_id: Optional[int]
) -> Tuple[bool, Optional[str], Optional[List[float]], Optional[Dict[str, float]]]:
    """
    Strict mathematical verification of a 5-class probability distribution:
    - Finite (no NaN, null, or Inf)
    - Non-negative (p >= 0.0)
    - Exactly 5 class entries
    - Normalized (sums to 1.0 within tolerance, or normalized from percentage)
    - Argmax index strictly matches the reported predicted class_id
    """
    if class_id is None or not (0 <= class_id < NUM_CLASSES):
        return False, f"Invalid or missing class_id: {class_id}", None, None

    probs: Optional[List[float]] = None

    if raw_probabilities is not None and isinstance(raw_probabilities, (list, tuple, np.ndarray)):
        probs = [float(v) for v in raw_probabilities]
    elif probabilities_dict is not None and isinstance(probabilities_dict, dict):
        if len(probabilities_dict) == NUM_CLASSES:
            # Reconstruct in canonical class order
            probs = [float(probabilities_dict.get(c, -1.0)) for c in CLASS_NAMES]

    if probs is None or len(probs) != NUM_CLASSES:
        return False, f"Probability distribution length must be {NUM_CLASSES}, got {len(probs) if probs else 0}", None, None

    # Check for NaN, Inf, negative
    for i, v in enumerate(probs):
        if v is None or not np.isfinite(v):
            return False, f"Probability at index {i} is not finite ({v})", None, None
        if v < 0.0:
            return False, f"Probability at index {i} is negative ({v})", None, None

    prob_sum = float(sum(probs))
    if prob_sum <= 0.0:
        return False, f"Probability sum is zero or negative ({prob_sum})", None, None

    # Normalize to [0.0, 1.0] sum
    if abs(prob_sum - 1.0) > 1e-4:
        if abs(prob_sum - 100.0) < 5.0:
            probs = [v / 100.0 for v in probs]
        else:
            probs = [v / prob_sum for v in probs]

    # Verify argmax matches class_id
    argmax_idx = int(np.argmax(probs))
    if argmax_idx != class_id:
        return False, f"Probability argmax ({argmax_idx}) does not match predicted class_id ({class_id})", None, None

    # Build validated dict
    validated_dict = {
        CLASS_NAMES[i]: round(probs[i] * 100.0, 2)
        for i in range(NUM_CLASSES)
    }

    return True, None, probs, validated_dict


def validate_and_rank_model_predictions(
    comparison_results: List[Dict[str, Any]],
    confidence_threshold: float = CONFIDENCE_THRESHOLD_PCT
) -> Dict[str, Any]:
    """
    Validates model outputs, sorts by descending predicted-class confidence,
    and partitions into primary result, alternative results, and unranked/failed models.
    
    Guarantees:
    - Primary result is selected strictly by highest softmax confidence among valid models.
    - Softmax confidence is NOT conflated with dataset-level validation accuracy.
    - Invalid, NaN, or non-normalizable probabilities are excluded from ranking.
    - Alternatives are sorted in descending order of confidence.
    """
    valid_predictions: List[Dict[str, Any]] = []
    unranked_or_failed: List[Dict[str, Any]] = []

    for item in comparison_results:
        # Check basic status
        if item.get("status") != "Ready":
            unranked_or_failed.append(dict(item))
            continue

        cid = item.get("class_id")
        conf = item.get("confidence_pct")
        if conf is None or not np.isfinite(conf) or conf < 0.0 or conf > 100.0:
            failed_item = dict(item)
            failed_item["status"] = "Error"
            failed_item["error_msg"] = f"Invalid confidence value: {conf}"
            unranked_or_failed.append(failed_item)
            continue

        # Strict probability distribution verification
        is_valid, err_msg, norm_probs, norm_dict = validate_probability_vector(
            raw_probabilities=item.get("raw_probabilities"),
            probabilities_dict=item.get("probabilities"),
            class_id=cid
        )

        if not is_valid:
            failed_item = dict(item)
            failed_item["status"] = "Error"
            failed_item["error_msg"] = f"Probability validation failed: {err_msg}"
            unranked_or_failed.append(failed_item)
            continue

        # Valid model prediction
        valid_item = dict(item)
        cname = (
            item.get("class_name")
            or item.get("predicted_class_name")
            or item.get("prediction")
            or (CLASS_NAMES[cid] if cid is not None and 0 <= cid < len(CLASS_NAMES) else "Unknown")
        )
        sname = (
            item.get("short_name")
            or (CLASS_SHORT_NAMES[cid] if cid is not None and 0 <= cid < len(CLASS_SHORT_NAMES) else f"Stage {cid}")
        )
        valid_item["class_name"] = cname
        valid_item["predicted_class_name"] = cname
        valid_item["prediction"] = cname
        valid_item["short_name"] = sname
        valid_item["probabilities"] = norm_dict
        valid_item["raw_probabilities"] = norm_probs
        valid_item["confidence"] = round(conf / 100.0, 4)
        valid_item["is_low_confidence"] = bool(conf < confidence_threshold)
        valid_item["confidence_threshold"] = confidence_threshold
        if not valid_item.get("description") and cid is not None and cid in CLASS_DESCRIPTIONS:
            valid_item["description"] = CLASS_DESCRIPTIONS[cid]
        if not valid_item.get("recommendation") and cid is not None and cid in CLASS_RECOMMENDATIONS:
            valid_item["recommendation"] = CLASS_RECOMMENDATIONS[cid]
        if not valid_item.get("severity_color") and cid is not None and cid in CLASS_COLORS:
            valid_item["severity_color"] = CLASS_COLORS[cid]
        if not valid_item.get("architecture_type"):
            valid_item["architecture_type"] = "Deep Convolutional Network"
        if not valid_item.get("training_type"):
            valid_item["training_type"] = "Transfer Learning"
        if not valid_item.get("dataset_name"):
            valid_item["dataset_name"] = DATASET_NAME
        valid_predictions.append(valid_item)

    # Sort valid predictions strictly by descending predicted-class confidence
    valid_predictions.sort(key=lambda x: x["confidence_pct"], reverse=True)

    primary_result: Optional[Dict[str, Any]] = None
    alternative_results: List[Dict[str, Any]] = []

    if valid_predictions:
        # Top-1 highest confidence prediction
        primary_result = dict(valid_predictions[0])
        primary_result["is_highest_confidence"] = True
        primary_result["confidence_label"] = "Highest model confidence"
        valid_predictions[0]["is_highest_confidence"] = True

        # Remaining models sorted in descending order
        for alt in valid_predictions[1:]:
            alt_copy = dict(alt)
            alt_copy["is_highest_confidence"] = False
            alt_copy["confidence_label"] = f"Alternative ({alt_copy['model_name']})"
            # Delta from primary confidence
            alt_copy["confidence_delta"] = round(alt_copy["confidence_pct"] - primary_result["confidence_pct"], 2)
            alternative_results.append(alt_copy)

    # Combined all results with primary marked first
    combined_all = valid_predictions + unranked_or_failed

    return {
        "primary_result": primary_result,
        "alternative_results": alternative_results,
        "all_ranked_results": valid_predictions,
        "unranked_or_failed": unranked_or_failed,
        "total_evaluated": len(comparison_results),
        "successful_count": len(valid_predictions),
        "has_valid_prediction": primary_result is not None
    }


def compute_model_agreement(comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates consensus / prediction agreement across available evaluated models.
    
    Returns:
        {
            "available_count": int,
            "total_count": int,
            "agreement_class_name": Optional[str],
            "agreeing_count": int,
            "agreement_pct": float,
            "agreement_label": str
        }
    """
    valid_predictions = [r for r in comparison_results if r["status"] == "Ready" and r["class_id"] is not None]
    total_models = len(comparison_results)
    available_count = len(valid_predictions)

    if available_count == 0:
        return {
            "available_count": 0,
            "total_count": total_models,
            "agreement_class_name": None,
            "agreeing_count": 0,
            "agreement_pct": 0.0,
            "agreement_label": "No trained models available for comparison."
        }

    # Count votes per class
    votes: Dict[int, int] = {}
    for r in valid_predictions:
        cid = r["class_id"]
        votes[cid] = votes.get(cid, 0) + 1

    # Find majority
    winning_cid = max(votes, key=votes.get)
    agreeing_count = votes[winning_cid]
    agreement_class_name = CLASS_NAMES[winning_cid]
    agreement_pct = (agreeing_count / available_count) * 100.0

    agreement_label = (
        f"{agreeing_count} of {available_count} available model(s) predicted: {agreement_class_name}"
    )

    return {
        "available_count": available_count,
        "total_count": total_models,
        "agreement_class_name": agreement_class_name,
        "agreeing_count": agreeing_count,
        "agreement_pct": agreement_pct,
        "agreement_label": agreement_label
    }
