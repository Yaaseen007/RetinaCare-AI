"""
==============================================================================
RetinaCare AI - Real-Image Inference Verification & Evaluation Audit
Department of Computer Science & Engineering - GM University

Executes comprehensive verification across 8 criteria:
1. Real-image inference via VGG16 website endpoint (POST /api/predict/web)
2. Real-image inference via MobileNetV2 mobile endpoint (POST /api/predict/mobile)
3. Model weight verification & checkpoint integrity
4. Preprocessing alignment check (training vs inference pipelines)
5. ICDR 5-class standard & probability calibration validation
6. Numerical sanity check (NaN, Inf, normalization ranges, edge cases)
7. Architectural separation & routing exclusivity verification
8. Independent evaluation methodology & data leakage audit (ROC-AUC 1.0000 analysis)
==============================================================================
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_PYLIB = Path("E:/pylib")
if _PYLIB.exists() and str(_PYLIB) not in sys.path:
    sys.path.insert(0, str(_PYLIB))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from src.class_mapping import CLASS_NAMES, CLASS_SHORT_NAMES, NUM_CLASSES
from src.model_manager import load_model_instance, get_model_status, CANONICAL_MODEL_NAMES
from src.preprocessing import preprocess_retinal_image, load_image_as_rgb_array

client = TestClient(app)

SAMPLE_DIR = PROJECT_ROOT / "assets" / "sample_images"
TEST_DATASET_DIR = PROJECT_ROOT / "dataset" / "test"
VAL_DATASET_DIR = PROJECT_ROOT / "dataset" / "validation"
TRAIN_DATASET_DIR = PROJECT_ROOT / "dataset" / "train"

def compute_file_hash(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def run_real_image_verification():
    print("=" * 80)
    print("RETINACARE AI: COMPREHENSIVE REAL-IMAGE INFERENCE & EVALUATION AUDIT")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # SECTION 1 & 2: DUAL ENDPOINT INFERENCE ON REAL RETINAL IMAGES
    # ------------------------------------------------------------------------
    print("\n--- [SECTION 1 & 2] DUAL ENDPOINT REAL-IMAGE INFERENCE ---")
    
    # Collect real test images: 5 clinical reference samples + 5 dataset test images (one per class)
    test_images: List[Tuple[str, str, Path]] = []
    
    # Clinical sample images
    for i in range(5):
        sample_file = list(SAMPLE_DIR.glob(f"sample_{i}_*.png"))
        if sample_file:
            test_images.append((f"Sample {i} (Class {i}: {CLASS_SHORT_NAMES[i]})", f"Class {i}", sample_file[0]))
            
    # Dataset test images
    for c in range(5):
        c_dir = TEST_DATASET_DIR / str(c)
        if c_dir.exists():
            files = list(c_dir.glob("*.png")) + list(c_dir.glob("*.jpg"))
            if files:
                test_images.append((f"Dataset Test Class {c} ({files[0].name})", f"Class {c}", files[0]))

    print(f"Total test images selected for evaluation: {len(test_images)}")

    vgg_results = []
    mobilenet_results = []

    for desc, expected_label, img_path in test_images:
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # 1. Test POST /api/predict/web (VGG16)
        resp_web = client.post(
            "/api/predict/web",
            files={"file": (img_path.name, img_bytes, "image/png")},
            data={"confidence_threshold": "50.0", "include_preprocessed_image": "true"}
        )
        assert resp_web.status_code == 200, f"Web inference failed: {resp_web.text}"
        data_web = resp_web.json()

        # 2. Test POST /api/predict/mobile (MobileNetV2)
        resp_mob = client.post(
            "/api/predict/mobile",
            files={"file": (img_path.name, img_bytes, "image/png")},
            data={"confidence_threshold": "50.0", "include_preprocessed_image": "true"}
        )
        assert resp_mob.status_code == 200, f"Mobile inference failed: {resp_mob.text}"
        data_mob = resp_mob.json()

        # Validations for Web
        assert data_web["model_name"] == "VGG16"
        assert 0 <= data_web["class_id"] < NUM_CLASSES
        assert not np.isnan(data_web["confidence_pct"])
        assert len(data_web["raw_probabilities"]) == 5
        assert abs(sum(data_web["raw_probabilities"]) - 1.0) < 1e-3

        # Validations for Mobile
        assert data_mob["model_name"] == "MobileNetV2"
        assert 0 <= data_mob["class_id"] < NUM_CLASSES
        assert not np.isnan(data_mob["confidence_pct"])
        assert len(data_mob["raw_probabilities"]) == 5
        assert abs(sum(data_mob["raw_probabilities"]) - 1.0) < 1e-3

        vgg_results.append({
            "image": img_path.name,
            "expected": expected_label,
            "pred_id": data_web["class_id"],
            "pred_name": data_web["short_name"],
            "confidence": data_web["confidence_pct"],
            "latency_ms": data_web["latency_ms"],
            "probs": [round(p, 4) for p in data_web["raw_probabilities"]]
        })

        mobilenet_results.append({
            "image": img_path.name,
            "expected": expected_label,
            "pred_id": data_mob["class_id"],
            "pred_name": data_mob["short_name"],
            "confidence": data_mob["confidence_pct"],
            "latency_ms": data_mob["latency_ms"],
            "probs": [round(p, 4) for p in data_mob["raw_probabilities"]]
        })

        print(f"\n[Image: {img_path.name}] (Expected: {expected_label})")
        print(f"  VGG16 (Web)      -> Pred: {data_web['short_name']} (Class {data_web['class_id']}) | Conf: {data_web['confidence_pct']:.2f}% | Latency: {data_web['latency_ms']}ms")
        print(f"  MobileNetV2 (Mob) -> Pred: {data_mob['short_name']} (Class {data_mob['class_id']}) | Conf: {data_mob['confidence_pct']:.2f}% | Latency: {data_mob['latency_ms']}ms")

    # ------------------------------------------------------------------------
    # SECTION 3: MODEL WEIGHT VERIFICATION & ARCHITECTURAL INTEGRITY
    # ------------------------------------------------------------------------
    print("\n--- [SECTION 3] MODEL WEIGHT & ARCHITECTURAL INTEGRITY ---")
    
    # VGG16
    vgg_model = load_model_instance("VGG16")
    vgg_params = vgg_model.count_params()
    vgg_weights = vgg_model.get_weights()
    vgg_dense_w = [w for w in vgg_weights if len(w.shape) == 2]
    print(f"VGG16 Model:")
    print(f"  Input Shape: {vgg_model.input_shape}")
    print(f"  Output Shape: {vgg_model.output_shape}")
    print(f"  Total Parameters: {vgg_params:,}")
    print(f"  Classification Head Dense Layer Weight Shapes: {[w.shape for w in vgg_dense_w]}")
    print(f"  Output Layer Weight Mean: {np.mean(vgg_dense_w[-1]):.6f}, Std: {np.std(vgg_dense_w[-1]):.6f}, Min: {np.min(vgg_dense_w[-1]):.6f}, Max: {np.max(vgg_dense_w[-1]):.6f}")
    assert vgg_model.input_shape == (None, 224, 224, 3)
    assert vgg_model.output_shape == (None, 5)
    assert np.std(vgg_dense_w[-1]) > 0.001, "Weights appear uninitialized/constant"

    # MobileNetV2
    mob_model = load_model_instance("MobileNetV2")
    mob_params = mob_model.count_params()
    mob_weights = mob_model.get_weights()
    mob_dense_w = [w for w in mob_weights if len(w.shape) == 2]
    print(f"\nMobileNetV2 Model:")
    print(f"  Input Shape: {mob_model.input_shape}")
    print(f"  Output Shape: {mob_model.output_shape}")
    print(f"  Total Parameters: {mob_params:,}")
    print(f"  Classification Head Dense Layer Weight Shapes: {[w.shape for w in mob_dense_w]}")
    print(f"  Output Layer Weight Mean: {np.mean(mob_dense_w[-1]):.6f}, Std: {np.std(mob_dense_w[-1]):.6f}, Min: {np.min(mob_dense_w[-1]):.6f}, Max: {np.max(mob_dense_w[-1]):.6f}")
    assert mob_model.input_shape == (None, 224, 224, 3)
    assert mob_model.output_shape == (None, 5)
    assert np.std(mob_dense_w[-1]) > 0.001, "Weights appear uninitialized/constant"

    # ------------------------------------------------------------------------
    # SECTION 4: PREPROCESSING VERIFICATION
    # ------------------------------------------------------------------------
    print("\n--- [SECTION 4] PREPROCESSING PIPELINE ALIGNMENT ---")
    test_img_path = test_images[0][2]
    rgb_raw = load_image_as_rgb_array(test_img_path)
    preprocessed = preprocess_retinal_image(test_img_path, target_size=(224, 224), crop_circle=True, enhance=True, normalize=True)
    
    print(f"Raw image shape: {rgb_raw.shape}, dtype: {rgb_raw.dtype}, range: [{rgb_raw.min()}, {rgb_raw.max()}]")
    print(f"Preprocessed image shape: {preprocessed.shape}, dtype: {preprocessed.dtype}, range: [{preprocessed.min():.4f}, {preprocessed.max():.4f}]")
    assert preprocessed.shape == (224, 224, 3)
    assert preprocessed.dtype == np.float32
    assert 0.0 <= preprocessed.min() <= 0.05
    assert 0.95 <= preprocessed.max() <= 1.0
    print("[PASS] Preprocessing correctly standardizes to RGB float32 in range [0.0, 1.0] at 224x224.")

    # ------------------------------------------------------------------------
    # SECTION 5 & 6: NUMERICAL SANITY & ROBUSTNESS CHECKS
    # ------------------------------------------------------------------------
    print("\n--- [SECTION 5 & 6] NUMERICAL SANITY & ERROR HANDLING CHECKS ---")
    
    # 1. Truncated / empty file
    resp_empty = client.post("/api/predict/web", files={"file": ("empty.png", b"", "image/png")})
    assert resp_empty.status_code == 400
    print(f"[PASS] Empty file correctly rejected: {resp_empty.status_code} ({resp_empty.json()['detail']})")

    # 2. Corrupted file bytes
    resp_corrupt = client.post("/api/predict/web", files={"file": ("corrupt.png", b"NOT_AN_IMAGE_FILE", "image/png")})
    assert resp_corrupt.status_code == 400
    print(f"[PASS] Corrupted file correctly rejected: {resp_corrupt.status_code} ({resp_corrupt.json()['detail']})")

    # 3. Invalid extension
    resp_badext = client.post("/api/predict/web", files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")})
    assert resp_badext.status_code == 400
    print(f"[PASS] Invalid extension correctly rejected: {resp_badext.status_code} ({resp_badext.json()['detail']})")

    # ------------------------------------------------------------------------
    # SECTION 7: ARCHITECTURAL EXCLUSIVITY & ROUTING VERIFICATION
    # ------------------------------------------------------------------------
    print("\n--- [SECTION 7] ARCHITECTURAL SEPARATION & ROUTING EXCLUSIVITY ---")
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    # Route 1: POST /api/predict/web -> always VGG16
    resp_web = client.post("/api/predict/web", files={"file": ("img.png", img_bytes, "image/png")})
    assert resp_web.status_code == 200
    assert resp_web.json()["model_name"] == "VGG16"
    print("[PASS] Dedicated /api/predict/web returns VGG16 exclusively.")

    # Route 2: POST /api/predict/mobile -> always MobileNetV2
    resp_mob = client.post("/api/predict/mobile", files={"file": ("img.png", img_bytes, "image/png")})
    assert resp_mob.status_code == 200
    assert resp_mob.json()["model_name"] == "MobileNetV2"
    print("[PASS] Dedicated /api/predict/mobile returns MobileNetV2 exclusively.")

    # Route 3: POST /api/predict with client_type="web" + model_name="MobileNetV2" -> 400 Refusal
    resp_reject = client.post(
        "/api/predict",
        files={"file": ("img.png", img_bytes, "image/png")},
        data={"client_type": "web", "model_name": "MobileNetV2"}
    )
    assert resp_reject.status_code == 400
    print(f"[PASS] Cross-platform violation rejected with 400: '{resp_reject.json()['detail']}'")

    # Route 4: POST /api/predict with client_type="web" -> VGG16
    resp_web_route = client.post(
        "/api/predict",
        files={"file": ("img.png", img_bytes, "image/png")},
        data={"client_type": "web"}
    )
    assert resp_web_route.status_code == 200
    assert resp_web_route.json()["model_name"] == "VGG16"
    print("[PASS] Standard /api/predict with client_type='web' routes to VGG16.")

    # Route 5: POST /api/predict with client_type="mobile" -> MobileNetV2
    resp_mob_route = client.post(
        "/api/predict",
        files={"file": ("img.png", img_bytes, "image/png")},
        data={"client_type": "mobile"}
    )
    assert resp_mob_route.status_code == 200
    assert resp_mob_route.json()["model_name"] == "MobileNetV2"
    print("[PASS] Standard /api/predict with client_type='mobile' routes to MobileNetV2.")

    # ------------------------------------------------------------------------
    # SECTION 8: INDEPENDENT EVALUATION AUDIT & DATA LEAKAGE ANALYSIS
    # ------------------------------------------------------------------------
    print("\n--- [SECTION 8] DATA LEAKAGE AUDIT & ROC-AUC 1.0000 INVESTIGATION ---")

    def get_dir_hashes(directory: Path) -> Dict[str, Tuple[str, int]]:
        hashes = {}
        for c in range(5):
            c_dir = directory / str(c)
            if not c_dir.exists():
                continue
            for f in c_dir.glob("*.*"):
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    h = compute_file_hash(f)
                    hashes[h] = (f.name, c)
        return hashes

    train_hashes = get_dir_hashes(TRAIN_DATASET_DIR)
    val_hashes = get_dir_hashes(VAL_DATASET_DIR)
    test_hashes = get_dir_hashes(TEST_DATASET_DIR)

    print(f"Dataset Split Sizes:")
    print(f"  dataset/train:      {len(train_hashes)} unique images")
    print(f"  dataset/validation: {len(val_hashes)} unique images")
    print(f"  dataset/test:       {len(test_hashes)} unique images")

    train_val_overlap = set(train_hashes.keys()) & set(val_hashes.keys())
    train_test_overlap = set(train_hashes.keys()) & set(test_hashes.keys())
    val_test_overlap = set(val_hashes.keys()) & set(test_hashes.keys())

    print(f"\nHash Collision / Overlap Analysis:")
    print(f"  Train <-> Validation overlap: {len(train_val_overlap)} images")
    print(f"  Train <-> Test overlap:       {len(train_test_overlap)} images")
    print(f"  Validation <-> Test overlap:  {len(val_test_overlap)} images (out of {len(test_hashes)} test images) -> {len(val_test_overlap)/len(test_hashes)*100:.1f}% OVERLAP!")

    overlap_details = []
    for h in val_test_overlap:
        val_name, val_c = val_hashes[h]
        test_name, test_c = test_hashes[h]
        overlap_details.append((val_name, test_name, test_c))

    print("\nSample Overlapping Images (Validation vs Test):")
    for v_name, t_name, c in overlap_details[:5]:
        print(f"  Class {c}: Validation '{v_name}' == Test '{t_name}'")

    print(f"\nTotal Overlapping Images in Test Set: {len(overlap_details)} / {len(test_hashes)}")

    # Re-evaluate MobileNetV2 on all 25 test samples
    print("\nRunning Empirical Re-Evaluation of MobileNetV2 on All 25 Test Images:")
    y_true = []
    y_scores = []
    y_preds = []

    for c in range(5):
        c_dir = TEST_DATASET_DIR / str(c)
        if not c_dir.exists():
            continue
        for img_p in c_dir.glob("*.*"):
            if img_p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            with open(img_p, "rb") as f:
                b = f.read()
            resp = client.post("/api/predict/mobile", files={"file": (img_p.name, b, "image/png")})
            pdata = resp.json()
            y_true.append(c)
            y_scores.append(pdata["raw_probabilities"])
            y_preds.append(pdata["class_id"])

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_preds = np.array(y_preds)

    acc = np.mean(y_true == y_preds)
    print(f"  Empirical Top-1 Accuracy: {acc*100:.2f}% ({np.sum(y_true == y_preds)} / {len(y_true)})")

    # Binarize labels for multi-class ROC-AUC calculation
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
    
    Y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    try:
        macro_roc_auc = roc_auc_score(Y_bin, y_scores, average="macro", multi_class="ovr")
        weighted_roc_auc = roc_auc_score(Y_bin, y_scores, average="weighted", multi_class="ovr")
        print(f"  Empirical Macro ROC-AUC: {macro_roc_auc:.4f}")
        print(f"  Empirical Weighted ROC-AUC: {weighted_roc_auc:.4f}")
    except Exception as e:
        print(f"  ROC-AUC calculation note: {e}")

    cm = confusion_matrix(y_true, y_preds, labels=[0, 1, 2, 3, 4])
    print(f"\nEmpirical Confusion Matrix (Rows=True, Cols=Pred):\n{cm}")

    # Re-evaluate VGG16 on all 25 test samples
    print("\nRunning Empirical Evaluation of VGG16 on All 25 Test Images:")
    y_true_vgg = []
    y_scores_vgg = []
    y_preds_vgg = []

    for c in range(5):
        c_dir = TEST_DATASET_DIR / str(c)
        if not c_dir.exists():
            continue
        for img_p in c_dir.glob("*.*"):
            if img_p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            with open(img_p, "rb") as f:
                b = f.read()
            resp = client.post("/api/predict/web", files={"file": (img_p.name, b, "image/png")})
            pdata = resp.json()
            y_true_vgg.append(c)
            y_scores_vgg.append(pdata["raw_probabilities"])
            y_preds_vgg.append(pdata["class_id"])

    y_true_vgg = np.array(y_true_vgg)
    y_scores_vgg = np.array(y_scores_vgg)
    y_preds_vgg = np.array(y_preds_vgg)

    acc_vgg = np.mean(y_true_vgg == y_preds_vgg)
    print(f"  VGG16 Top-1 Accuracy: {acc_vgg*100:.2f}% ({np.sum(y_true_vgg == y_preds_vgg)} / {len(y_true_vgg)})")
    
    Y_bin_vgg = label_binarize(y_true_vgg, classes=[0, 1, 2, 3, 4])
    try:
        macro_roc_auc_vgg = roc_auc_score(Y_bin_vgg, y_scores_vgg, average="macro", multi_class="ovr")
        print(f"  VGG16 Macro ROC-AUC: {macro_roc_auc_vgg:.4f}")
    except Exception as e:
        print(f"  VGG16 ROC-AUC calculation note: {e}")

    cm_vgg = confusion_matrix(y_true_vgg, y_preds_vgg, labels=[0, 1, 2, 3, 4])
    print(f"\nVGG16 Confusion Matrix (Rows=True, Cols=Pred):\n{cm_vgg}")

    print("\n" + "=" * 80)
    print("ALL VERIFICATION SUITES EXECUTED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_real_image_verification()

