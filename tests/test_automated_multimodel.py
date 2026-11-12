"""
==============================================================================
RetinaCare AI - Automated Multi-Model & Alternative Results Test Suite
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
from src.prediction import (
    validate_probability_vector,
    validate_and_rank_model_predictions,
    compute_model_agreement,
    CONFIDENCE_THRESHOLD_PCT
)


class TestAutomatedMultiModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_path = PROJECT_ROOT / "assets" / "sample_images" / "sample_0_no_dr.png"
        assert cls.sample_path.is_file(), f"Sample image missing at {cls.sample_path}"
        with open(cls.sample_path, "rb") as f:
            cls.sample_bytes = f.read()

    # ------------------------------------------------------------------------
    # 1. Highest-Confidence Model Selected as Primary
    # ------------------------------------------------------------------------
    def test_01_highest_confidence_selected_as_primary(self):
        mock_results = [
            {
                "model_name": "Model B",
                "status": "Ready",
                "class_id": 0,
                "confidence_pct": 65.0,
                "raw_probabilities": [0.65, 0.15, 0.10, 0.05, 0.05],
                "probabilities": {"No DR": 65.0, "Mild": 15.0, "Moderate": 10.0, "Severe": 5.0, "Proliferative": 5.0}
            },
            {
                "model_name": "Model A (Champion)",
                "status": "Ready",
                "class_id": 0,
                "confidence_pct": 82.5,
                "raw_probabilities": [0.825, 0.075, 0.05, 0.03, 0.02],
                "probabilities": {"No DR": 82.5, "Mild": 7.5, "Moderate": 5.0, "Severe": 3.0, "Proliferative": 2.0}
            },
            {
                "model_name": "Model C",
                "status": "Ready",
                "class_id": 0,
                "confidence_pct": 55.0,
                "raw_probabilities": [0.55, 0.20, 0.15, 0.05, 0.05],
                "probabilities": {"No DR": 55.0, "Mild": 20.0, "Moderate": 15.0, "Severe": 5.0, "Proliferative": 5.0}
            }
        ]

        ranked = validate_and_rank_model_predictions(mock_results)
        self.assertTrue(ranked["has_valid_prediction"])
        self.assertEqual(ranked["primary_result"]["model_name"], "Model A (Champion)")
        self.assertEqual(ranked["primary_result"]["confidence_pct"], 82.5)
        self.assertTrue(ranked["primary_result"]["is_highest_confidence"])
        self.assertEqual(ranked["primary_result"]["confidence_label"], "Highest model confidence")

    # ------------------------------------------------------------------------
    # 2. Alternative Predictions Sorted in Descending Order of Confidence
    # ------------------------------------------------------------------------
    def test_02_alternatives_sorted_descending(self):
        mock_results = [
            {
                "model_name": "Low",
                "status": "Ready",
                "class_id": 1,
                "confidence_pct": 48.0,
                "raw_probabilities": [0.20, 0.48, 0.20, 0.06, 0.06]
            },
            {
                "model_name": "High",
                "status": "Ready",
                "class_id": 1,
                "confidence_pct": 78.0,
                "raw_probabilities": [0.05, 0.78, 0.10, 0.04, 0.03]
            },
            {
                "model_name": "Medium",
                "status": "Ready",
                "class_id": 1,
                "confidence_pct": 62.0,
                "raw_probabilities": [0.10, 0.62, 0.18, 0.05, 0.05]
            }
        ]

        ranked = validate_and_rank_model_predictions(mock_results)
        self.assertEqual(ranked["primary_result"]["model_name"], "High")
        alts = ranked["alternative_results"]
        self.assertEqual(len(alts), 2)
        self.assertEqual(alts[0]["model_name"], "Medium")
        self.assertEqual(alts[1]["model_name"], "Low")
        self.assertGreater(alts[0]["confidence_pct"], alts[1]["confidence_pct"])
        self.assertFalse(alts[0]["is_highest_confidence"])
        self.assertFalse(alts[1]["is_highest_confidence"])

    # ------------------------------------------------------------------------
    # 3. Model Failures Do Not Discard Valid Predictions
    # ------------------------------------------------------------------------
    def test_03_model_failures_do_not_discard_valid_predictions(self):
        mock_results = [
            {
                "model_name": "Working Model 1",
                "status": "Ready",
                "class_id": 0,
                "confidence_pct": 71.0,
                "raw_probabilities": [0.71, 0.15, 0.08, 0.04, 0.02]
            },
            {
                "model_name": "Missing Model",
                "status": "Not Available",
                "class_id": None,
                "confidence_pct": None,
                "raw_probabilities": None,
                "error_msg": "Checkpoint not found"
            },
            {
                "model_name": "Crashing Model",
                "status": "Error",
                "class_id": None,
                "confidence_pct": None,
                "raw_probabilities": None,
                "error_msg": "CUDA out of memory"
            },
            {
                "model_name": "Working Model 2",
                "status": "Ready",
                "class_id": 0,
                "confidence_pct": 59.0,
                "raw_probabilities": [0.59, 0.20, 0.11, 0.06, 0.04]
            }
        ]

        ranked = validate_and_rank_model_predictions(mock_results)
        self.assertTrue(ranked["has_valid_prediction"])
        self.assertEqual(ranked["primary_result"]["model_name"], "Working Model 1")
        self.assertEqual(len(ranked["alternative_results"]), 1)
        self.assertEqual(ranked["alternative_results"][0]["model_name"], "Working Model 2")
        self.assertEqual(len(ranked["unranked_or_failed"]), 2)
        self.assertEqual(ranked["successful_count"], 2)
        self.assertEqual(ranked["total_evaluated"], 4)

    # ------------------------------------------------------------------------
    # 4. Invalid, NaN, Infinite & Malformed Probabilities Rejected
    # ------------------------------------------------------------------------
    def test_04_invalid_nan_infinite_malformed_probabilities_rejected(self):
        # Case A: NaN in probabilities
        valid, msg, _, _ = validate_probability_vector([0.5, float("nan"), 0.2, 0.1, 0.1], None, 0)
        self.assertFalse(valid)
        self.assertIn("not finite", msg)

        # Case B: Inf in probabilities
        valid, msg, _, _ = validate_probability_vector([0.5, float("inf"), 0.2, 0.1, 0.1], None, 0)
        self.assertFalse(valid)
        self.assertIn("not finite", msg)

        # Case C: Negative probability
        valid, msg, _, _ = validate_probability_vector([0.7, -0.1, 0.2, 0.1, 0.1], None, 0)
        self.assertFalse(valid)
        self.assertIn("negative", msg)

        # Case D: Incorrect length (4 elements instead of 5)
        valid, msg, _, _ = validate_probability_vector([0.4, 0.3, 0.2, 0.1], None, 0)
        self.assertFalse(valid)
        self.assertIn("length must be 5", msg)

        # Case E: Argmax mismatch (class_id=0 but argmax is at index 2)
        valid, msg, _, _ = validate_probability_vector([0.1, 0.1, 0.6, 0.1, 0.1], None, 0)
        self.assertFalse(valid)
        self.assertIn("argmax (2) does not match", msg)

        # In ranking, bad models must end up in unranked_or_failed
        bad_results = [{
            "model_name": "Corrupt Model",
            "status": "Ready",
            "class_id": 0,
            "confidence_pct": 50.0,
            "raw_probabilities": [0.5, float("nan"), 0.2, 0.1, 0.1]
        }]
        ranked = validate_and_rank_model_predictions(bad_results)
        self.assertFalse(ranked["has_valid_prediction"])
        self.assertEqual(len(ranked["unranked_or_failed"]), 1)
        self.assertEqual(ranked["unranked_or_failed"][0]["status"], "Error")

    # ------------------------------------------------------------------------
    # 5. Class Index and Name Mapping Integrity
    # ------------------------------------------------------------------------
    def test_05_class_index_and_name_mapping_integrity(self):
        expected_classes = [
            "No Diabetic Retinopathy",
            "Mild",
            "Moderate",
            "Severe",
            "Proliferative Diabetic Retinopathy"
        ]
        self.assertEqual(CLASS_NAMES, expected_classes)
        self.assertEqual(NUM_CLASSES, 5)

        for i, name in enumerate(CLASS_NAMES):
            probs = [0.05] * 5
            probs[i] = 0.80
            valid, err, _, p_dict = validate_probability_vector(probs, None, i)
            self.assertTrue(valid, err)
            self.assertEqual(p_dict[name], 80.0)

    # ------------------------------------------------------------------------
    # 6. Low Confidence Alert Flagging
    # ------------------------------------------------------------------------
    def test_06_low_confidence_flagging(self):
        low_conf = [{
            "model_name": "Uncertain Model",
            "status": "Ready",
            "class_id": 2,
            "confidence_pct": 41.5,
            "raw_probabilities": [0.15, 0.20, 0.415, 0.135, 0.10]
        }]
        ranked_low = validate_and_rank_model_predictions(low_conf, confidence_threshold=50.0)
        self.assertTrue(ranked_low["primary_result"]["is_low_confidence"])

        high_conf = [{
            "model_name": "Certain Model",
            "status": "Ready",
            "class_id": 2,
            "confidence_pct": 85.0,
            "raw_probabilities": [0.03, 0.05, 0.85, 0.05, 0.02]
        }]
        ranked_high = validate_and_rank_model_predictions(high_conf, confidence_threshold=50.0)
        self.assertFalse(ranked_high["primary_result"]["is_low_confidence"])

    # ------------------------------------------------------------------------
    # 7. Backend /api/compare Endpoint Verification
    # ------------------------------------------------------------------------
    def test_07_backend_compare_endpoint_structure(self):
        resp = self.client.post(
            "/api/compare",
            files={"file": ("sample.png", self.sample_bytes, "image/png")}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()

        # Check best_model presence and highest confidence flag
        self.assertIn("best_model", data)
        best = data["best_model"]
        self.assertIsNotNone(best)
        self.assertTrue(best["is_highest_confidence"])
        self.assertIn(best["prediction"], CLASS_NAMES)
        self.assertGreater(best["confidence_pct"], 0.0)

        # Check alternative_models
        self.assertIn("alternative_models", data)
        alts = data["alternative_models"]
        self.assertEqual(len(alts), 4)

        # Verify sorted in descending order
        prev_conf = best["confidence_pct"]
        for alt in alts:
            self.assertLessEqual(alt["confidence_pct"], prev_conf)
            self.assertFalse(alt["is_highest_confidence"])
            prev_conf = alt["confidence_pct"]

        # Check agreement consensus
        agreement = data["agreement"]
        self.assertEqual(agreement["available_count"], 5)
        self.assertIn(agreement["agreement_class_name"], CLASS_NAMES)

    # ------------------------------------------------------------------------
    # 8. Empty & Corrupted Image Rejection
    # ------------------------------------------------------------------------
    def test_08_empty_or_corrupt_images_rejected(self):
        # Empty file
        resp_empty = self.client.post(
            "/api/compare",
            files={"file": ("empty.png", b"", "image/png")}
        )
        self.assertEqual(resp_empty.status_code, 400)
        self.assertIn("empty", resp_empty.json()["detail"].lower())

        # Corrupted bytes
        resp_corrupt = self.client.post(
            "/api/compare",
            files={"file": ("corrupt.png", b"NOT_A_VALID_IMAGE_HEADER_12345", "image/png")}
        )
        self.assertEqual(resp_corrupt.status_code, 400)

    # ------------------------------------------------------------------------
    # 9. Existing Endpoints Still Functional
    # ------------------------------------------------------------------------
    def test_09_existing_endpoints_preserved(self):
        # Health
        resp_health = self.client.get("/api/health")
        self.assertEqual(resp_health.status_code, 200)

        # Models
        resp_models = self.client.get("/api/models")
        self.assertEqual(resp_models.status_code, 200)
        self.assertEqual(resp_models.json()["available_count"], 5)

        # Single predict
        resp_pred = self.client.post(
            "/api/predict",
            files={"file": ("sample.png", self.sample_bytes, "image/png")},
            data={"model_name": "MobileNetV2"}
        )
        self.assertEqual(resp_pred.status_code, 200)
        self.assertEqual(resp_pred.json()["model_name"], "MobileNetV2")


if __name__ == "__main__":
    unittest.main()
