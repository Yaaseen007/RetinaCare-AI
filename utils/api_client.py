"""
==============================================================================
RetinaCare AI - FastAPI Client SDK
HTTP Communication Layer between Streamlit UI and FastAPI Backend Service
==============================================================================
"""

from __future__ import annotations
import io
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import requests
from PIL import Image
import numpy as np

logger = logging.getLogger("RetinaCare.APIClient")

# Base URL for FastAPI Backend
DEFAULT_API_URL = "http://127.0.0.1:8000"
API_BASE_URL = os.environ.get("RETINACARE_API_URL", DEFAULT_API_URL).rstrip("/")


class BackendConnectionError(RuntimeError):
    """Raised when the FastAPI backend is unreachable."""
    pass


class BackendAPIError(RuntimeError):
    """Raised when the FastAPI backend returns an error HTTP status."""
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"Backend API Error ({status_code}): {detail}")
        self.status_code = status_code
        self.detail = detail


def _prepare_image_payload(image_source: Any) -> Tuple[str, bytes, str]:
    """Converts any supported image input into (filename, bytes, content_type)."""
    if isinstance(image_source, (str, Path)):
        p = Path(image_source)
        if not p.is_file():
            raise FileNotFoundError(f"Image file does not exist: {p}")
        with open(p, "rb") as f:
            data = f.read()
        suffix = p.suffix.lower()
        content_type = "image/png" if suffix == ".png" else "image/jpeg"
        return p.name, data, content_type

    if isinstance(image_source, bytes):
        return "retinal_image.jpg", image_source, "image/jpeg"

    if isinstance(image_source, io.BytesIO):
        image_source.seek(0)
        return "retinal_image.jpg", image_source.read(), "image/jpeg"

    if hasattr(image_source, "read"):
        data = image_source.read()
        name = getattr(image_source, "name", "retinal_image.jpg")
        suffix = Path(name).suffix.lower()
        content_type = "image/png" if suffix == ".png" else ("image/bmp" if suffix == ".bmp" else "image/jpeg")
        return name, data, content_type

    if isinstance(image_source, Image.Image):
        buf = io.BytesIO()
        rgb = image_source.convert("RGB")
        rgb.save(buf, format="JPEG", quality=95)
        return "retinal_image.jpg", buf.getvalue(), "image/jpeg"

    if isinstance(image_source, np.ndarray):
        pil_img = Image.fromarray(image_source.astype(np.uint8)).convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        return "retinal_image.jpg", buf.getvalue(), "image/jpeg"

    raise ValueError(f"Unsupported image type: {type(image_source)}")


def check_backend_health(timeout: float = 3.0) -> Dict[str, Any]:
    """
    Pings GET /api/health to verify backend readiness.
    Returns health dictionary or raises BackendConnectionError.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/api/health", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        raise BackendAPIError(resp.status_code, resp.text)
    except requests.exceptions.RequestException as exc:
        raise BackendConnectionError(f"Cannot connect to FastAPI at {API_BASE_URL}: {str(exc)}") from exc


def is_backend_alive(timeout: float = 2.0) -> bool:
    """Non-throwing check to see if backend is responsive."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def fetch_models_list(timeout: float = 5.0) -> List[Dict[str, Any]]:
    """
    Fetches list of registered models and their statuses from GET /api/models.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/api/models", timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("models", [])
        raise BackendAPIError(resp.status_code, resp.text)
    except requests.exceptions.RequestException as exc:
        raise BackendConnectionError(f"Failed to fetch models from {API_BASE_URL}: {str(exc)}") from exc


def predict_web_via_api(
    image_source: Any,
    confidence_threshold: float = 50.0,
    timeout: float = 45.0
) -> Dict[str, Any]:
    """
    Submits image to dedicated website prediction route POST /api/predict/web (VGG16).
    """
    filename, img_bytes, content_type = _prepare_image_payload(image_source)
    files = {"file": (filename, img_bytes, content_type)}
    data = {
        "confidence_threshold": str(confidence_threshold),
        "include_preprocessed_image": "true"
    }

    try:
        resp = requests.post(f"{API_BASE_URL}/api/predict/web", files=files, data=data, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise BackendConnectionError(f"Failed to reach FastAPI at {API_BASE_URL}: {str(exc)}") from exc

    if resp.status_code == 200:
        return resp.json()
    try:
        err_detail = resp.json().get("detail", resp.text)
    except Exception:
        err_detail = resp.text
    raise BackendAPIError(resp.status_code, err_detail)


def predict_mobile_via_api(
    image_source: Any,
    confidence_threshold: float = 50.0,
    timeout: float = 45.0
) -> Dict[str, Any]:
    """
    Submits image to dedicated mobile app prediction route POST /api/predict/mobile (MobileNetV2).
    """
    filename, img_bytes, content_type = _prepare_image_payload(image_source)
    files = {"file": (filename, img_bytes, content_type)}
    data = {
        "confidence_threshold": str(confidence_threshold),
        "include_preprocessed_image": "true"
    }

    try:
        resp = requests.post(f"{API_BASE_URL}/api/predict/mobile", files=files, data=data, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise BackendConnectionError(f"Failed to reach FastAPI at {API_BASE_URL}: {str(exc)}") from exc

    if resp.status_code == 200:
        return resp.json()
    try:
        err_detail = resp.json().get("detail", resp.text)
    except Exception:
        err_detail = resp.text
    raise BackendAPIError(resp.status_code, err_detail)


def predict_via_api(
    model_name: str = "VGG16",
    image_source: Any = None,
    client_type: Optional[str] = "web",
    confidence_threshold: float = 50.0,
    timeout: float = 45.0
) -> Dict[str, Any]:
    """
    Submits image to POST /api/predict on FastAPI backend.
    
    Returns:
        Structured prediction dictionary with all clinical biomarker fields.
    """
    filename, img_bytes, content_type = _prepare_image_payload(image_source)

    files = {
        "file": (filename, img_bytes, content_type)
    }
    data = {
        "model_name": model_name,
        "client_type": client_type or "web",
        "confidence_threshold": str(confidence_threshold),
        "include_preprocessed_image": "true"
    }

    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/predict",
            files=files,
            data=data,
            timeout=timeout
        )
    except requests.exceptions.RequestException as exc:
        raise BackendConnectionError(
            f"Failed to reach FastAPI backend at {API_BASE_URL}. Ensure uvicorn server is running on port 8000: {str(exc)}"
        ) from exc

    if resp.status_code == 200:
        return resp.json()
    else:
        try:
            err_detail = resp.json().get("detail", resp.text)
        except Exception:
            err_detail = resp.text
        raise BackendAPIError(resp.status_code, err_detail)


def compare_via_api(
    image_source: Any,
    timeout: float = 90.0
) -> Dict[str, Any]:
    """
    Submits image to POST /api/compare on FastAPI backend.
    
    Returns:
        {"results": List[Dict], "agreement": Dict}
    """
    filename, img_bytes, content_type = _prepare_image_payload(image_source)

    files = {
        "file": (filename, img_bytes, content_type)
    }

    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/compare",
            files=files,
            timeout=timeout
        )
    except requests.exceptions.RequestException as exc:
        raise BackendConnectionError(
            f"Failed to reach FastAPI backend at {API_BASE_URL}: {str(exc)}"
        ) from exc

    if resp.status_code == 200:
        return resp.json()
    else:
        try:
            err_detail = resp.json().get("detail", resp.text)
        except Exception:
            err_detail = resp.text
        raise BackendAPIError(resp.status_code, err_detail)


def fetch_metrics_via_api(timeout: float = 5.0) -> Dict[str, Any]:
    """
    Fetches performance metrics from GET /api/model-metrics.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/api/model-metrics", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        raise BackendAPIError(resp.status_code, resp.text)
    except requests.exceptions.RequestException as exc:
        raise BackendConnectionError(f"Failed to fetch metrics from {API_BASE_URL}: {str(exc)}") from exc
