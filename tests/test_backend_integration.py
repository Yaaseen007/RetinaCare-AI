"""
==============================================================================
RetinaCare AI - Backend & Streamlit Integration Verification Test Suite
Department of Computer Science & Engineering - GM University
==============================================================================
"""

import io
import sys
from pathlib import Path

# Add project root and pylib to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_PYLIB = Path("E:/pylib")
if _PYLIB.exists() and str(_PYLIB) not in sys.path:
    sys.path.insert(0, str(_PYLIB))

import unittest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from src.class_mapping import CLASS_NAMES, NUM_CLASSES
from src.model_manager import CANONICAL_MODEL_NAMES
import utils.api_client as api_client


class TestBackendIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_path = PROJECT_ROOT / "assets" / "sample_images" / "sample_0_no_dr.png"
        assert cls.sample_path.is_file(), f"Sample image missing at {cls.sample_path}"
        with open(cls.sample_path, "rb") as f:
            cls.sample_bytes = f.read()

    # ------------------------------------------------------------------------
    # 1. Health Endpoint Verification
    # ------------------------------------------------------------------------
    def test_01_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["app_name"], "RetinaCare AI Backend")
        self.assertEqual(data["version"], "1.0.0")
        self.assertEqual(data["available_models_count"], 5)
        self.assertEqual(data["total_models_count"], 5)
        self.assertIn("timestamp", data)
        print(" [PASS] test_01_health_endpoint")

    # ------------------------------------------------------------------------
    # 2. Models List Endpoint Verification
    # ------------------------------------------------------------------------
    def test_02_models_list_endpoint(self):
        resp = self.client.get("/api/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["count"], 5)
        self.assertEqual(data["available_count"], 5)
        self.assertEqual(len(data["models"]), 5)

        names = [m["canonical_name"] for m in data["models"]]
        for cname in CANONICAL_MODEL_NAMES:
            self.assertIn(cname, names)

        for m in data["models"]:
            self.assertTrue(m["is_available"])
            self.assertEqual(m["status_label"], "READY")
            self.assertEqual(m["classes"], 5)
            self.assertGreater(m["file_size_mb"], 0.0)
            self.assertIsNotNone(m["checkpoint_path"])
        print(" [PASS] test_02_models_list_endpoint")

    # ------------------------------------------------------------------------
    # 3. Model Metrics & Genuine Evaluation Telemetry
    # ------------------------------------------------------------------------
    def test_03_model_metrics_endpoint(self):
        resp = self.client.get("/api/model-metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("model_comparison", data)
        self.assertIn("primary_model_metrics", data)

        comp = data["model_comparison"]
        self.assertIsInstance(comp, dict)
        self.assertIn("MobileNetV2", comp)
        self.assertIn("Accuracy", comp["MobileNetV2"])
        self.assertEqual(comp["MobileNetV2"]["Status"], "Trained")

        primary = data["primary_model_metrics"]
        self.assertIsNotNone(primary)
        self.assertIn("accuracy", primary)
        self.assertIn("macro_precision", primary)
        self.assertIn("macro_recall", primary)
        self.assertIn("macro_f1", primary)
        self.assertIn("classification_report", primary)
        print(" [PASS] test_03_model_metrics_endpoint")

    # ------------------------------------------------------------------------
    # 4. Predict Endpoint: Valid Images Across Multiple Models
    # ------------------------------------------------------------------------
    def test_04_predict_all_models_and_probability_validity(self):
        for model_name in CANONICAL_MODEL_NAMES:
            resp = self.client.post(
                "/api/predict",
                files={"file": ("sample_0.png", self.sample_bytes, "image/png")},
                data={
                    "model_name": model_name,
                    "confidence_threshold": "50.0",
                    "include_preprocessed_image": "true"
                }
            )
            self.assertEqual(resp.status_code, 200, f"Failed on model {model_name}: {resp.text}")
            d = resp.json()

            # Class verification
            self.assertIn(d["class_id"], range(NUM_CLASSES))
            self.assertEqual(d["class_name"], CLASS_NAMES[d["class_id"]])
            self.assertIn(d["model_name"], [model_name, "MobileNetV2", "Custom CNN", "VGG16", "ResNet50", "EfficientNetB0"])

            # Probability mathematical validity
            probs = d["probabilities"]
            self.assertEqual(len(probs), NUM_CLASSES)
            prob_sum = sum(probs.values())
            self.assertAlmostEqual(prob_sum, 100.0, delta=1.5)

            # Argmax matching
            best_prob_class = max(probs, key=probs.get)
            self.assertEqual(d["class_name"], best_prob_class, "Predicted class doesn't match highest probability class!")
            self.assertAlmostEqual(d["confidence_pct"], probs[best_prob_class], delta=0.1)

            # Raw probabilities alignment
            raw_p = d["raw_probabilities"]
            self.assertEqual(len(raw_p), NUM_CLASSES)
            self.assertEqual(int(np.argmax(raw_p)), d["class_id"])

            # Latency and base64 preprocessed image
            self.assertGreater(d["latency_ms"], 0)
            self.assertIsNotNone(d["preprocessed_image_base64"])
            self.assertTrue(d["preprocessed_image_base64"].startswith("data:image/jpeg;base64,"))

            print(f" [PASS] test_04_predict: {model_name:<16} -> {d['class_name']} ({d['confidence_pct']}%)")

    # ------------------------------------------------------------------------
    # 5. Invalid File & Error Handling Verification
    # ------------------------------------------------------------------------
    def test_05_predict_empty_file_rejected(self):
        resp = self.client.post(
            "/api/predict",
            files={"file": ("empty.png", b"", "image/png")},
            data={"model_name": "MobileNetV2"}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("empty", resp.json()["detail"].lower())
        print(" [PASS] test_05_predict_empty_file_rejected")

    def test_06_predict_unsupported_extension_rejected(self):
        resp = self.client.post(
            "/api/predict",
            files={"file": ("document.pdf", b"dummy pdf content", "application/pdf")},
            data={"model_name": "MobileNetV2"}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported file format", resp.json()["detail"])
        print(" [PASS] test_06_predict_unsupported_extension_rejected")

    def test_07_predict_corrupted_image_rejected(self):
        resp = self.client.post(
            "/api/predict",
            files={"file": ("corrupt.png", b"invalid random non-image bytes 12345678", "image/png")},
            data={"model_name": "MobileNetV2"}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Image could not be opened", resp.json()["detail"])
        print(" [PASS] test_07_predict_corrupted_image_rejected")

    def test_08_predict_invalid_model_rejected(self):
        resp = self.client.post(
            "/api/predict",
            files={"file": ("sample.png", self.sample_bytes, "image/png")},
            data={"model_name": "NonExistentModel123"}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid model", resp.json()["detail"])
        print(" [PASS] test_08_predict_invalid_model_rejected")

    # ------------------------------------------------------------------------
    # 6. Model Comparison Endpoint Verification
    # ------------------------------------------------------------------------
    def test_09_compare_models_endpoint(self):
        resp = self.client.post(
            "/api/compare",
            files={"file": ("sample.png", self.sample_bytes, "image/png")}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertIn("results", data)
        self.assertIn("agreement", data)

        results = data["results"]
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertEqual(r["status"], "Ready")
            self.assertIn(r["prediction"], CLASS_NAMES)
            self.assertIsNotNone(r["confidence_pct"])
            self.assertIsNotNone(r["probabilities"])

        agreement = data["agreement"]
        self.assertEqual(agreement["available_count"], 5)
        self.assertEqual(agreement["total_count"], 5)
        self.assertIn(agreement["agreement_class_name"], CLASS_NAMES)
        self.assertGreater(agreement["agreeing_count"], 0)
        self.assertGreater(agreement["agreement_pct"], 0.0)
        self.assertIn("available model(s) predicted", agreement["agreement_label"])
        print(f" [PASS] test_09_compare_models: Consensus -> {agreement['agreement_label']}")

    def test_10_compare_models_invalid_file_rejected(self):
        resp = self.client.post(
            "/api/compare",
            files={"file": ("bad.txt", b"plain text data", "text/plain")}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported file format", resp.json()["detail"])
        print(" [PASS] test_10_compare_models_invalid_file_rejected")

    def test_11_compare_models_empty_file_rejected(self):
        resp = self.client.post(
            "/api/compare",
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("empty", resp.json()["detail"].lower())
        print(" [PASS] test_11_compare_models_empty_file_rejected")

    # ------------------------------------------------------------------------
    # 7. Low Confidence Threshold Flagging
    # ------------------------------------------------------------------------
    def test_12_low_confidence_threshold_flagging(self):
        resp = self.client.post(
            "/api/predict",
            files={"file": ("sample_0.png", self.sample_bytes, "image/png")},
            data={"model_name": "MobileNetV2", "confidence_threshold": "99.9"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["is_low_confidence"])
        self.assertIn("low", data["confidence_status"].lower())
        print(" [PASS] test_12_low_confidence_threshold_flagging")


if __name__ == "__main__":
    unittest.main()
