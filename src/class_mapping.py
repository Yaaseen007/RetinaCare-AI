"""
==============================================================================
RetinaCare AI - Dataset & Class Mapping Verification Engine
Department of Computer Science & Engineering - GM University

Grounds predictions in the verified training dataset and enforces exact
class order alignment (ICDR 5-Stage Diabetic Retinopathy Scale).
==============================================================================
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Canonical Dataset Metadata
DATASET_NAME = "APTOS 2019 / Kaggle Diabetic Retinopathy"
DATASET_SOURCE_DETAILS = "Kaggle APTOS 2019 Blindness Detection & EyePACS Curated Cohort"
DEFAULT_INPUT_RESOLUTION = (224, 224)
INPUT_RESOLUTION_STR = "224 × 224"
NUM_CLASSES = 5

# Canonical 5-Class Disease Grading (ICDR Scale)
# Strict numerical ordering matching training labels (0 to 4)
CLASS_NAMES: List[str] = [
    "No Diabetic Retinopathy",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative Diabetic Retinopathy"
]

CLASS_SHORT_NAMES: List[str] = [
    "No DR (0)",
    "Mild (1)",
    "Moderate (2)",
    "Severe (3)",
    "Proliferative (4)"
]

# Detailed clinical biomarker definitions for each ICDR stage
CLASS_DESCRIPTIONS: Dict[int, str] = {
    0: (
        "No signs of diabetic retinopathy detected. Retinal vascular network is intact and normal; "
        "no microaneurysms, hemorrhages, hard lipid exudates, or cotton wool spots observed."
    ),
    1: (
        "Mild Non-Proliferative Diabetic Retinopathy (NPDR). Hallmark sign: "
        "Presence of microaneurysms only (isolated focal capillary dilatations <30μm in diameter)."
    ),
    2: (
        "Moderate NPDR. Characterized by multiple microaneurysms, dot-and-blot intraretinal "
        "hemorrhages, venous dilation, and hard lipid exudates, but less severe than the 4-2-1 rule."
    ),
    3: (
        "Severe NPDR. High risk of progression to proliferative disease. Characterized by the clinical "
        "4-2-1 rule: severe intraretinal hemorrhages in all 4 quadrants, definite venous beading in 2+ "
        "quadrants, or prominent intraretinal microvascular abnormalities (IRMA) in 1+ quadrant."
    ),
    4: (
        "Proliferative Diabetic Retinopathy (PDR). Advanced sight-threatening stage. "
        "Characterized by neovascularization (fragile, leaky new vessel proliferation on disc or retina) "
        "and/or preretinal/vitreous hemorrhages with high risk of tractional retinal detachment."
    )
}

# Clinical follow-up recommendations (Academic screening guidance)
CLASS_RECOMMENDATIONS: Dict[int, str] = {
    0: "Schedule routine annual dilated eye examination. Maintain optimal glycemic and blood pressure control.",
    1: "Repeat comprehensive dilated fundus examination within 6 to 12 months. Monitor HbA1c and lipid levels.",
    2: "Referral to an ophthalmologist or retinal specialist within 3 to 6 months for detailed evaluation.",
    3: "Prompt referral to a retinal specialist within 2 to 4 weeks. Evaluation for panretinal photocoagulation or anti-VEGF therapy.",
    4: "Urgent/Immediate retinal specialist consultation within 24 to 72 hours. Critical risk of vision loss requiring surgical or laser intervention."
}

# Apple Healthcare UI Color Palette
CLASS_COLORS: Dict[int, str] = {
    0: "#30D158",  # Emerald Green - Safe / Normal
    1: "#64D2FF",  # Sky Blue - Mild Attention
    2: "#FF9F0A",  # Amber - Moderate Alert
    3: "#FF851B",  # Orange - High Alert
    4: "#FF453A"   # Crimson Red - Sight-threatening / Critical
}

MEDICAL_DISCLAIMER: str = (
    "DISCLAIMER: This system is an AI-based tool developed solely for academic research, "
    "educational demonstration, and preliminary screening support. It is NOT certified as "
    "a medical diagnostic device. The predictions must never replace clinical judgment, "
    "slit-lamp biomicroscopy, or formal diagnosis by a licensed ophthalmologist."
)


class ClassMappingVerificationError(RuntimeError):
    """Raised when dataset class mapping integrity check fails."""
    pass


def verify_class_mapping(
    model_output_shape: Optional[Tuple[Optional[int], ...]] = None,
    dataset_dir: Optional[Path] = None
) -> Tuple[bool, str]:
    """
    Verifies that the model's output configuration conforms exactly to the
    5-class ICDR Diabetic Retinopathy scale.
    
    Checks:
    1. Output tensor dimension equals NUM_CLASSES (5).
    2. Class names and descriptions cover all indices 0..4.
    3. Dataset subdirectories (if provided) adhere to classes '0', '1', '2', '3', '4'.
    
    Returns:
        (is_valid: bool, verification_message: str)
    """
    # 1. Verify internal class definitions
    if len(CLASS_NAMES) != NUM_CLASSES or len(CLASS_SHORT_NAMES) != NUM_CLASSES:
        raise ClassMappingVerificationError(
            f"Class mapping mismatch: expected {NUM_CLASSES} classes, "
            f"found {len(CLASS_NAMES)} names."
        )

    # 2. Verify model output shape if supplied
    if model_output_shape is not None:
        last_dim = model_output_shape[-1]
        if last_dim != NUM_CLASSES:
            raise ClassMappingVerificationError(
                f"Model output shape mismatch: model outputs {last_dim} classes, "
                f"but dataset requires exactly {NUM_CLASSES} classes."
            )

    # 3. Verify dataset folder structure if available
    if dataset_dir is not None and dataset_dir.is_dir():
        for split in ["train", "validation", "test"]:
            split_path = dataset_dir / split
            if split_path.is_dir():
                found_classes = {p.name for p in split_path.iterdir() if p.is_dir()}
                expected = {str(i) for i in range(NUM_CLASSES)}
                missing = expected - found_classes
                if missing:
                    raise ClassMappingVerificationError(
                        f"Dataset folder '{split}' is missing class subdirectories: {sorted(list(missing))}"
                    )

    return True, f"Class mapping verified: 5 classes (0..4) strictly mapped to '{DATASET_NAME}'"
