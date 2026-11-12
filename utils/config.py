"""
==============================================================================
Diabetic Retinopathy Detection Using Convolutional Neural Networks
Academic PBL Project - Department of Computer Science & Engineering
GM University

Configuration Module: Centralizes all paths, hyperparameters, clinical labels,
and system constants used across preprocessing, training, evaluation, and UI.
==============================================================================
"""

import os
import sys
from pathlib import Path

# Add external library directory if present (bypasses Windows MAX_PATH limitations)
_PYLIB = Path("E:/pylib")
if _PYLIB.exists() and str(_PYLIB) not in sys.path:
    sys.path.insert(0, str(_PYLIB))

# ----------------------------------------------------------------------------
# 1. Project Directory Paths
# ----------------------------------------------------------------------------
# Determine base project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset subdirectories
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# Models and checkpoints
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
BACKUP_MODEL_PATH = MODELS_DIR / "best_model.h5"

# Evaluation artifacts and logs
RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PLOT_PATH = RESULTS_DIR / "training_history.png"
CLASS_DISTRIBUTION_PATH = RESULTS_DIR / "class_distribution.png"
EVALUATION_REPORT_PATH = RESULTS_DIR / "evaluation_report.png"
METRICS_JSON_PATH = RESULTS_DIR / "metrics.json"
MODEL_COMPARISON_JSON_PATH = RESULTS_DIR / "model_comparison.json"

# Static assets and sample imagery
ASSETS_DIR = PROJECT_ROOT / "assets"
SAMPLE_IMAGES_DIR = ASSETS_DIR / "sample_images"

# ----------------------------------------------------------------------------
# 2. Image Preprocessing Specifications
# ----------------------------------------------------------------------------
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Supported image file extensions for uploads and dataset reading
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ----------------------------------------------------------------------------
# 3. Clinical Disease Classification Standards (ICDR / APTOS 5-Class Scale)
# ----------------------------------------------------------------------------
NUM_CLASSES = 5

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative Diabetic Retinopathy"
]

CLASS_SHORT_NAMES = [
    "No DR (0)",
    "Mild (1)",
    "Moderate (2)",
    "Severe (3)",
    "Proliferative (4)"
]

# Detailed human-readable clinical biomarker explanations for academic & educational use
CLASS_DESCRIPTIONS = {
    0: (
        "No signs of diabetic retinopathy detected. Retinal vascular network "
        "is intact and normal; no microaneurysms, hemorrhages, or exudates observed."
    ),
    1: (
        "Mild Non-Proliferative Diabetic Retinopathy (NPDR). Hallmark sign: "
        "Presence of microaneurysms only (small circular outpouchings in tiny retinal capillaries)."
    ),
    2: (
        "Moderate NPDR. Multiple microaneurysms, dot-and-blot hemorrhages, "
        "cotton wool spots (nerve fiber layer infarcts), and/or hard lipid exudates present."
    ),
    3: (
        "Severe NPDR. High risk of rapid progression to proliferative stage. "
        "Characterized by the 4-2-1 rule: severe intraretinal hemorrhages in all 4 quadrants, "
        "venous beading in 2+ quadrants, or prominent IRMA (intraretinal microvascular abnormalities) in 1+ quadrant."
    ),
    4: (
        "Proliferative Diabetic Retinopathy (PDR). Advanced, sight-threatening stage. "
        "Characterized by neovascularization (abnormal fragile new blood vessels) and/or "
        "preretinal/vitreous hemorrhages. High risk of tractional retinal detachment."
    )
}

# Recommended medical follow-up guidelines (Educational reference)
CLASS_RECOMMENDATIONS = {
    0: "Schedule regular routine annual diabetic eye examination. Maintain optimal glycemic and blood pressure control.",
    1: "Repeat comprehensive dilated eye exam within 6 to 12 months. Monitor HbA1c and cholesterol levels.",
    2: "Referral to an ophthalmologist/retina specialist within 3 to 6 months for detailed evaluation and monitoring.",
    3: "Prompt referral to a retinal specialist within 2 to 4 weeks. Evaluation for panretinal photocoagulation or anti-VEGF therapy.",
    4: "Urgent/Immediate retinal specialist consultation within 24 to 72 hours. High risk of vision loss requiring surgical or laser intervention."
}

# Hex color scheme mapping severity to visual UI cues
CLASS_COLORS = {
    0: "#10b981",  # Emerald Green - Safe / Normal
    1: "#84cc16",  # Lime Green - Mild Attention
    2: "#f59e0b",  # Amber/Yellow - Moderate Alert
    3: "#f97316",  # Deep Orange - High Alert
    4: "#ef4444"   # Crimson Red - Critical / Proliferative
}

# ----------------------------------------------------------------------------
# 4. Model Architecture Allocations (Dual-Platform Architecture)
# ----------------------------------------------------------------------------
WEB_MODEL_NAME = "VGG16"           # Dedicated website screening model
MOBILE_MODEL_NAME = "MobileNetV2"  # Dedicated Flutter Android application model
DEFAULT_MODEL_NAME = "VGG16"       # Default screening architecture for web
CANDIDATE_MODELS = ["VGG16", "MobileNetV2", "EfficientNetB0", "ResNet50", "CustomCNN"]

BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

INITIAL_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.3
L2_WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# 5. Mandatory Medical Disclaimer
# ----------------------------------------------------------------------------
MEDICAL_DISCLAIMER = (
    "DISCLAIMER: This system is an AI-based tool developed solely for academic research, "
    "educational demonstration, and preliminary screening support. It is NOT certified as "
    "a medical diagnostic device. The predictions must never replace clinical judgment, "
    "slit-lamp examination, or formal diagnosis by a licensed ophthalmologist."
)

# ----------------------------------------------------------------------------
# 6. Utility Functions
# ----------------------------------------------------------------------------
def ensure_directories():
    """Ensure all core project directories exist on disk."""
    dirs = [
        DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
        MODELS_DIR, RESULTS_DIR, ASSETS_DIR, SAMPLE_IMAGES_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Ensure class subdirectories exist in dataset folders
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for i in range(NUM_CLASSES):
            (split_dir / str(i)).mkdir(parents=True, exist_ok=True)
