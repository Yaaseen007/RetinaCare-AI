"""
RetinaCare AI - Reconciliation Test Suite & Empirical Evidence Log
Tests:
1. GET /api/health
2. GET /api/models
3. POST /api/predict/web (VGG16 with real fundus image)
4. POST /api/predict/mobile (MobileNetV2 with real fundus image)
5. Cross-model rejection check
"""

import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def run_tests():
    print("=" * 70)
    print("RETINACARE AI - IN-PROCESS ASGI TEST EVIDENCE")
    print("=" * 70)

    # Test 1: Health
    t0 = time.time()
    r_health = client.get("/api/health")
    t_health = (time.time() - t0) * 1000
    print(f"\n[TEST 1] GET /api/health")
    print(f"  HTTP Status: {r_health.status_code}")
    print(f"  Latency: {t_health:.2f} ms")
    print(f"  Response JSON: {r_health.json()}")
    assert r_health.status_code == 200, f"Health check failed: {r_health.status_code}"

    # Test 2: Models List
    t0 = time.time()
    r_models = client.get("/api/models")
    t_models = (time.time() - t0) * 1000
    models_data = r_models.json()
    print(f"\n[TEST 2] GET /api/models")
    print(f"  HTTP Status: {r_models.status_code}")
    print(f"  Latency: {t_models:.2f} ms")
    print(f"  Model Count: {models_data.get('count')} (Available: {models_data.get('available_count')})")
    for m in models_data.get("models", []):
        print(f"    - {m['canonical_name']} ({m['display_name']}): Status={m['status_label']}, Size={m['file_size_mb']} MB, Target={m['target']}")
    assert r_models.status_code == 200, f"Models check failed: {r_models.status_code}"

    # Sample image
    sample_path = PROJECT_ROOT / "assets" / "sample_images" / "sample_0_no_dr.png"
    assert sample_path.exists(), f"Sample image not found at {sample_path}"

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    # Test 3: Web Prediction (VGG16)
    t0 = time.time()
    r_web = client.post(
        "/api/predict/web",
        files={"file": ("sample_0_no_dr.png", img_bytes, "image/png")},
        data={"confidence_threshold": "50.0"}
    )
    t_web = (time.time() - t0) * 1000
    web_json = r_web.json()
    print(f"\n[TEST 3] POST /api/predict/web (Website VGG16 Dedicated Endpoint)")
    print(f"  HTTP Status: {r_web.status_code}")
    print(f"  Total Client Latency: {t_web:.2f} ms")
    print(f"  Model Used: {web_json.get('model_name')}")
    print(f"  Architecture: {web_json.get('architecture_type')}")
    print(f"  Predicted Class ID: {web_json.get('class_id')}")
    print(f"  Predicted Class Name: {web_json.get('class_name')}")
    print(f"  Confidence: {web_json.get('confidence_pct')}%")
    print(f"  Internal Model Latency: {web_json.get('latency_ms')} ms")
    print(f"  Low Confidence Flag: {web_json.get('is_low_confidence')}")
    assert r_web.status_code == 200, f"Web predict failed: {r_web.status_code}"
    assert web_json.get("model_name") == "VGG16", f"Expected VGG16 but got {web_json.get('model_name')}"

    # Test 4: Mobile Prediction (MobileNetV2)
    t0 = time.time()
    r_mob = client.post(
        "/api/predict/mobile",
        files={"file": ("sample_0_no_dr.png", img_bytes, "image/png")},
        data={"confidence_threshold": "50.0"}
    )
    t_mob = (time.time() - t0) * 1000
    mob_json = r_mob.json()
    print(f"\n[TEST 4] POST /api/predict/mobile (Mobile MobileNetV2 Dedicated Endpoint)")
    print(f"  HTTP Status: {r_mob.status_code}")
    print(f"  Total Client Latency: {t_mob:.2f} ms")
    print(f"  Model Used: {mob_json.get('model_name')}")
    print(f"  Architecture: {mob_json.get('architecture_type')}")
    print(f"  Predicted Class ID: {mob_json.get('class_id')}")
    print(f"  Predicted Class Name: {mob_json.get('class_name')}")
    print(f"  Confidence: {mob_json.get('confidence_pct')}%")
    print(f"  Internal Model Latency: {mob_json.get('latency_ms')} ms")
    print(f"  Low Confidence Flag: {mob_json.get('is_low_confidence')}")
    assert r_mob.status_code == 200, f"Mobile predict failed: {r_mob.status_code}"
    assert mob_json.get("model_name") == "MobileNetV2", f"Expected MobileNetV2 but got {mob_json.get('model_name')}"

    # Test 5: Cross-platform restriction enforcement
    r_viol = client.post(
        "/api/predict",
        files={"file": ("sample_0_no_dr.png", img_bytes, "image/png")},
        data={"model_name": "MobileNetV2", "client_type": "web"}
    )
    print(f"\n[TEST 5] POST /api/predict with Cross-Platform Violation (MobileNetV2 on Web)")
    print(f"  HTTP Status: {r_viol.status_code} (Expected 400 Bad Request)")
    print(f"  Error Detail: {r_viol.json().get('detail')}")
    assert r_viol.status_code == 400, f"Expected 400, got {r_viol.status_code}"

    print("\n" + "=" * 70)
    print("ALL 5 RECONCILIATION TESTS PASSED WITH 100% SUCCESS")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
