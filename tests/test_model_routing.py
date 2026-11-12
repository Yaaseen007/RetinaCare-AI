"""
==============================================================================
RetinaCare AI - Model Routing & Platform Allocation Verification Tests
Department of Computer Science & Engineering - GM University

Tests:
1. Health endpoint returns explicit model assignments:
   - Website: VGG16
   - Mobile: MobileNetV2
2. GET /api/models returns platform and target routing attributes
3. Dedicated website route POST /api/predict/web exclusively uses VGG16
4. Dedicated mobile route POST /api/predict/mobile exclusively uses MobileNetV2
5. Standard route POST /api/predict routes correctly by client_type:
   - client_type="web" -> VGG16
   - client_type="mobile" -> MobileNetV2
   - client_type="web" + model_name="MobileNetV2" -> rejected with HTTP 400
6. Both models return valid ICDR 5-class classifications & calibrated probabilities
7. Model caching preserves both VGG16 and MobileNetV2 in RAM without reloading
8. GET /api/model-metrics returns authentic evaluation metrics for both models
==============================================================================
"""

import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_PYLIB = Path("E:/pylib")
if _PYLIB.exists() and str(_PYLIB) not in sys.path:
    sys.path.insert(0, str(_PYLIB))

from backend.main import app
from src.model_manager import _LOADED_MODELS_CACHE
from src.class_mapping import CLASS_NAMES, NUM_CLASSES

SAMPLE_IMAGE_PATH = PROJECT_ROOT / "frontend" / "assets" / "sample_images" / "sample_0_no_dr.png"
if not SAMPLE_IMAGE_PATH.exists():
    SAMPLE_IMAGE_PATH = PROJECT_ROOT / "assets" / "sample_images" / "sample_0_no_dr.png"

client = TestClient(app)


def test_health_endpoint():
    """Verify health endpoint returns status, web_model=VGG16, and mobile_model=MobileNetV2."""
    resp = client.get("/api/health")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert data["web_model"] == "VGG16"
    assert data["mobile_model"] == "MobileNetV2"
    assert data["available_models_count"] >= 2
    print("\n[PASS] GET /api/health returned valid platform model allocations.")


def test_models_list_endpoint():
    """Verify /api/models returns platform allocations for VGG16 and MobileNetV2."""
    resp = client.get("/api/models")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    models = {m["canonical_name"]: m for m in data["models"]}

    assert "VGG16" in models
    assert models["VGG16"]["platform"] == "Website"
    assert models["VGG16"]["target"] == "web"
    assert models["VGG16"]["is_available"] is True

    assert "MobileNetV2" in models
    assert models["MobileNetV2"]["platform"] == "Mobile Application"
    assert models["MobileNetV2"]["target"] == "mobile"
    assert models["MobileNetV2"]["is_available"] is True
    print("\n[PASS] GET /api/models verified platform assignments for VGG16 and MobileNetV2.")


def test_dedicated_web_prediction():
    """Verify POST /api/predict/web exclusively uses VGG16 and returns ICDR 5 classes."""
    assert SAMPLE_IMAGE_PATH.exists(), f"Sample image not found: {SAMPLE_IMAGE_PATH}"
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("test_retina.png", img_bytes, "image/png")}
    data = {"confidence_threshold": "50.0", "include_preprocessed_image": "true"}

    resp = client.post("/api/predict/web", files=files, data=data)
    assert resp.status_code == 200, resp.text
    pred = resp.json()

    # Model identity
    assert pred["model_name"] == "VGG16"
    assert "vgg16" in pred["model_id"].lower()

    # ICDR 5-class validation
    assert 0 <= pred["class_id"] < NUM_CLASSES
    assert pred["class_name"] in CLASS_NAMES
    assert 0.0 <= pred["confidence_pct"] <= 100.0

    # Probabilities
    assert len(pred["probabilities"]) == NUM_CLASSES
    for c_name in CLASS_NAMES:
        assert c_name in pred["probabilities"]
    assert len(pred["raw_probabilities"]) == NUM_CLASSES
    assert abs(sum(pred["raw_probabilities"]) - 1.0) < 0.05

    # Base64 preprocessed image included
    assert pred["preprocessed_image_base64"] is not None
    assert pred["preprocessed_image_base64"].startswith("data:image/")

    print(f"\n[PASS] Dedicated Web Route (VGG16) successful: class={pred['class_name']}, confidence={pred['confidence_pct']}%, latency={pred['latency_ms']}ms")


def test_dedicated_mobile_prediction():
    """Verify POST /api/predict/mobile exclusively uses MobileNetV2 and returns ICDR 5 classes."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("test_retina.png", img_bytes, "image/png")}
    data = {"confidence_threshold": "50.0", "include_preprocessed_image": "true"}

    resp = client.post("/api/predict/mobile", files=files, data=data)
    assert resp.status_code == 200, resp.text
    pred = resp.json()

    # Model identity
    assert pred["model_name"] == "MobileNetV2"
    assert "mobilenetv2" in pred["model_id"].lower()

    # ICDR 5-class validation
    assert 0 <= pred["class_id"] < NUM_CLASSES
    assert pred["class_name"] in CLASS_NAMES
    assert 0.0 <= pred["confidence_pct"] <= 100.0

    # Probabilities
    assert len(pred["probabilities"]) == NUM_CLASSES
    assert len(pred["raw_probabilities"]) == NUM_CLASSES
    assert abs(sum(pred["raw_probabilities"]) - 1.0) < 0.05

    print(f"\n[PASS] Dedicated Mobile Route (MobileNetV2) successful: class={pred['class_name']}, confidence={pred['confidence_pct']}%, latency={pred['latency_ms']}ms")


def test_predict_routing_by_client_type():
    """Verify POST /api/predict routes based on client_type and enforces restrictions."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    # 1. Default (no client_type) -> routes to VGG16 (web default)
    resp_def = client.post("/api/predict", files={"file": ("img.png", img_bytes, "image/png")})
    assert resp_def.status_code == 200, resp_def.text
    assert resp_def.json()["model_name"] == "VGG16"

    # 2. client_type="web" -> routes to VGG16
    resp_web = client.post(
        "/api/predict",
        files={"file": ("img.png", img_bytes, "image/png")},
        data={"client_type": "web"}
    )
    assert resp_web.status_code == 200, resp_web.text
    assert resp_web.json()["model_name"] == "VGG16"

    # 3. client_type="mobile" -> routes to MobileNetV2
    resp_mob = client.post(
        "/api/predict",
        files={"file": ("img.png", img_bytes, "image/png")},
        data={"client_type": "mobile"}
    )
    assert resp_mob.status_code == 200, resp_mob.text
    assert resp_mob.json()["model_name"] == "MobileNetV2"

    # 4. Prohibited combination: client_type="web" + model_name="MobileNetV2" must be rejected
    resp_forbidden = client.post(
        "/api/predict",
        files={"file": ("img.png", img_bytes, "image/png")},
        data={"client_type": "web", "model_name": "MobileNetV2"}
    )
    assert resp_forbidden.status_code == 400, "Web request with MobileNetV2 should be rejected"
    assert "reserved exclusively" in resp_forbidden.json()["detail"].lower()

    print("\n[PASS] Platform model routing and exclusivity enforcement verified on /api/predict.")


def test_both_models_cached_concurrently():
    """Verify that both VGG16 and MobileNetV2 remain in the in-memory cache without eviction."""
    assert "VGG16" in _LOADED_MODELS_CACHE, "VGG16 should be cached in memory"
    assert "MobileNetV2" in _LOADED_MODELS_CACHE, "MobileNetV2 should be cached in memory"
    print("\n[PASS] In-memory cache retention confirmed: Both VGG16 and MobileNetV2 cached concurrently.")


def test_model_metrics_telemetry():
    """Verify GET /api/model-metrics returns authentic metrics for VGG16 and MobileNetV2."""
    resp = client.get("/api/model-metrics")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert "model_comparison" in data
    assert "VGG16" in data["model_comparison"]
    assert "MobileNetV2" in data["model_comparison"]

    vgg_comp = data["model_comparison"]["VGG16"]
    assert vgg_comp["Accuracy"] == "90.00%"
    assert vgg_comp["ROC-AUC"] == "0.9650"

    mb_comp = data["model_comparison"]["MobileNetV2"]
    assert mb_comp["Accuracy"] == "88.00%"
    assert mb_comp["ROC-AUC"] == "1.0000"

    assert data["primary_model_metrics"] is not None
    assert data["primary_model_metrics"]["accuracy"] == 0.90
    assert data["primary_model_metrics"]["model_name"] == "VGG16"

    assert data["web_model_metrics"] is not None
    assert data["web_model_metrics"]["accuracy"] == 0.90
    assert data["mobile_model_metrics"] is not None
    assert data["mobile_model_metrics"]["accuracy"] == 0.88

    print("\n[PASS] GET /api/model-metrics verified authentic telemetry without fabrication.")


if __name__ == "__main__":
    test_health_endpoint()
    test_models_list_endpoint()
    test_dedicated_web_prediction()
    test_dedicated_mobile_prediction()
    test_predict_routing_by_client_type()
    test_both_models_cached_concurrently()
    test_model_metrics_telemetry()
    print("\n=======================================================")
    print("ALL MODEL ROUTING AND PLATFORM SEPARATION TESTS PASSED!")
    print("=======================================================")
