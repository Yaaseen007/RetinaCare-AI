"""
==============================================================================
Diabetic Retinopathy Detection - Synthetic Sample Fundus Generator
Department of Computer Science & Engineering - GM University

This utility creates representative synthetic retinal fundus images for:
1. Quick web application demonstration in assets/sample_images/ (No external downloads required)
2. Local pipeline testing and smoke testing of the training/inference code
3. Educational visualization of retinal anatomical landmarks:
   - Optic Disc (yellowish-orange oval)
   - Retinal Vascular Arcade (branching arteriolar/venular network)
   - Macula and Fovea Centralis (central dark avascular zone)
   - Pathological Biomarkers:
     * Class 0: Clean normal fundus
     * Class 1 (Mild): Microaneurysms (tiny punctate red spots)
     * Class 2 (Moderate): Blot hemorrhages + hard lipid exudates
     * Class 3 (Severe): Extensive intraretinal hemorrhages + cotton wool spots
     * Class 4 (Proliferative): Neovascularization network + vitreous hemorrhage
==============================================================================
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    ASSETS_DIR,
    SAMPLE_IMAGES_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    NUM_CLASSES,
    CLASS_SHORT_NAMES,
    ensure_directories
)


def create_synthetic_fundus(
    class_id: int = 0,
    size: int = 512,
    seed: int = 42
) -> np.ndarray:
    """
    Generates a realistic synthetic retinal fundus image representing the specified DR severity.
    """
    np.random.seed(seed + class_id * 100)

    # 1. Base retinal background (reddish-orange sphere with realistic illumination)
    img = np.zeros((size, size, 3), dtype=np.float32)
    center = (size // 2, size // 2)
    radius = int(size * 0.46)

    # Coordinate grids
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    fundus_mask = dist_from_center <= radius

    # Background gradient: slightly brighter near center, darker towards periphery (vignetting)
    vignette = 1.0 - 0.4 * (dist_from_center / radius) ** 1.5
    vignette = np.clip(vignette, 0.5, 1.0)

    # Retinal tissue color: R~200, G~80, B~40
    base_color = np.array([195.0, 75.0, 30.0], dtype=np.float32)
    
    # Choroidal texture variation (subtle low-frequency variations)
    noise = np.random.normal(0, 5.0, (size, size, 3)).astype(np.float32)
    fundus = (base_color + noise) * vignette[:, :, np.newaxis]
    fundus = np.clip(fundus, 0, 255)

    img[fundus_mask] = fundus[fundus_mask]

    # Convert to uint8 for drawing retinal structures
    canvas = img.astype(np.uint8)

    # 2. Optic Disc (bright yellowish-orange oval on nasal side)
    disc_center = (int(size * 0.28), int(size * 0.48))
    disc_axes = (int(size * 0.07), int(size * 0.08))
    cv2.ellipse(canvas, disc_center, disc_axes, 0, 0, 360, (245, 195, 120), -1)
    # Physiologic cup (brighter center)
    cv2.ellipse(canvas, disc_center, (int(disc_axes[0] * 0.5), int(disc_axes[1] * 0.5)), 0, 0, 360, (255, 235, 180), -1)

    # 3. Macula & Fovea (darker brownish-red zone on temporal side)
    macula_center = (int(size * 0.62), int(size * 0.50))
    cv2.circle(canvas, macula_center, int(size * 0.08), (145, 45, 20), -1)
    # Fovea centralis
    cv2.circle(canvas, macula_center, int(size * 0.02), (115, 30, 15), -1)

    # 4. Retinal Vascular Tree (Arterioles and Venules emerging from disc)
    vessel_color_main = (120, 20, 10)
    vessel_color_branch = (140, 30, 15)

    # Superior and inferior major arcades
    branches = [
        # Superior temporal arcade
        [(disc_center[0], disc_center[1]),
         (int(size * 0.38), int(size * 0.25)),
         (int(size * 0.58), int(size * 0.22)),
         (int(size * 0.75), int(size * 0.32))],
        # Inferior temporal arcade
        [(disc_center[0], disc_center[1]),
         (int(size * 0.38), int(size * 0.72)),
         (int(size * 0.58), int(size * 0.76)),
         (int(size * 0.76), int(size * 0.68))],
        # Nasal vessels
        [(disc_center[0], disc_center[1]),
         (int(size * 0.18), int(size * 0.40)),
         (int(size * 0.10), int(size * 0.35))],
        [(disc_center[0], disc_center[1]),
         (int(size * 0.18), int(size * 0.56)),
         (int(size * 0.10), int(size * 0.60))]
    ]

    for branch in branches:
        pts = np.array(branch, np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=vessel_color_main, thickness=3, lineType=cv2.LINE_AA)
        
        # Add smaller child branches
        for i in range(1, len(branch)):
            start_pt = branch[i]
            delta_x = np.random.randint(-20, 25)
            delta_y = np.random.randint(-25, 25)
            end_pt = (np.clip(start_pt[0] + delta_x, 0, size - 1), np.clip(start_pt[1] + delta_y, 0, size - 1))
            cv2.line(canvas, start_pt, end_pt, vessel_color_branch, thickness=2, lineType=cv2.LINE_AA)

    # 5. Pathological Lesions according to ICDR / APTOS Severity Level
    if class_id == 0:
        # No Diabetic Retinopathy: Intact, clean retina
        pass

    elif class_id == 1:
        # Mild: Only microaneurysms (tiny punctate red capillary outpouchings)
        for _ in range(np.random.randint(6, 12)):
            rx = np.random.randint(int(size * 0.35), int(size * 0.75))
            ry = np.random.randint(int(size * 0.30), int(size * 0.70))
            cv2.circle(canvas, (rx, ry), np.random.randint(1, 3), (95, 10, 5), -1)

    elif class_id == 2:
        # Moderate: Multiple microaneurysms, dot-and-blot hemorrhages, hard lipid exudates
        # Microaneurysms
        for _ in range(np.random.randint(15, 25)):
            rx = np.random.randint(int(size * 0.25), int(size * 0.80))
            ry = np.random.randint(int(size * 0.25), int(size * 0.75))
            cv2.circle(canvas, (rx, ry), np.random.randint(2, 3), (90, 10, 5), -1)
        # Dot-and-blot hemorrhages
        for _ in range(np.random.randint(6, 12)):
            rx = np.random.randint(int(size * 0.35), int(size * 0.75))
            ry = np.random.randint(int(size * 0.30), int(size * 0.70))
            cv2.circle(canvas, (rx, ry), np.random.randint(3, 6), (75, 5, 5), -1)
        # Hard Exudates (bright yellowish-white lipid deposits near macula)
        for _ in range(np.random.randint(8, 16)):
            ex = np.random.randint(int(size * 0.45), int(size * 0.70))
            ey = np.random.randint(int(size * 0.35), int(size * 0.65))
            cv2.circle(canvas, (ex, ey), np.random.randint(2, 4), (250, 240, 140), -1)

    elif class_id == 3:
        # Severe: Extensive intraretinal hemorrhages (>20 in 4 quadrants), venous beading, cotton wool spots
        # Widespread hemorrhages across all quadrants
        for _ in range(np.random.randint(35, 55)):
            rx = np.random.randint(int(size * 0.20), int(size * 0.85))
            ry = np.random.randint(int(size * 0.15), int(size * 0.85))
            cv2.circle(canvas, (rx, ry), np.random.randint(4, 9), (70, 5, 5), -1)
        # Cotton Wool Spots (fluffy whitish-gray nerve fiber infarcts)
        for _ in range(np.random.randint(4, 8)):
            cwx = np.random.randint(int(size * 0.30), int(size * 0.70))
            cwy = np.random.randint(int(size * 0.25), int(size * 0.75))
            cv2.ellipse(canvas, (cwx, cwy), (np.random.randint(7, 12), np.random.randint(5, 8)),
                        np.random.randint(0, 180), 0, 360, (220, 215, 210), -1)
        # Abundant exudates
        for _ in range(np.random.randint(15, 25)):
            ex = np.random.randint(int(size * 0.35), int(size * 0.75))
            ey = np.random.randint(int(size * 0.25), int(size * 0.75))
            cv2.circle(canvas, (ex, ey), np.random.randint(2, 5), (250, 245, 160), -1)

    elif class_id == 4:
        # Proliferative: Hallmark neovascularization (delicate abnormal vessel loops) + preretinal hemorrhage
        # Severe hemorrhages
        for _ in range(np.random.randint(40, 65)):
            rx = np.random.randint(int(size * 0.18), int(size * 0.85))
            ry = np.random.randint(int(size * 0.15), int(size * 0.85))
            cv2.circle(canvas, (rx, ry), np.random.randint(5, 11), (60, 0, 0), -1)
        # Preretinal/vitreous dense boat-shaped hemorrhage
        vh_center = (int(size * 0.50), int(size * 0.65))
        cv2.ellipse(canvas, vh_center, (int(size * 0.15), int(size * 0.08)), 15, 0, 180, (50, 0, 0), -1)
        # Neovascular fronds near optic disc
        for _ in range(12):
            nx = disc_center[0] + np.random.randint(-25, 45)
            ny = disc_center[1] + np.random.randint(-35, 35)
            cv2.line(canvas, disc_center, (nx, ny), (130, 15, 10), thickness=1, lineType=cv2.LINE_AA)

    # 6. Apply circular aperture mask to ensure authentic black fundus border
    canvas[~fundus_mask] = [0, 0, 0]

    # Subtle Gaussian smooth for natural ophthalmic lens dispersion
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5)
    return canvas


def generate_all_samples():
    """Generates 5 representative sample images in assets/sample_images/."""
    ensure_directories()
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    filenames = [
        "sample_0_no_dr.png",
        "sample_1_mild.png",
        "sample_2_moderate.png",
        "sample_3_severe.png",
        "sample_4_proliferative.png"
    ]

    print("[RetinaCare Generator] Generating reference sample images in assets/sample_images/...")
    for class_id in range(NUM_CLASSES):
        img = create_synthetic_fundus(class_id=class_id, size=512, seed=100 + class_id)
        # Convert RGB to BGR for cv2.imwrite
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        dest = SAMPLE_IMAGES_DIR / filenames[class_id]
        cv2.imwrite(str(dest), bgr)
        print(f"  Created: {dest.name} -> Class {class_id} ({CLASS_SHORT_NAMES[class_id]})")


def generate_mini_dataset(
    train_per_class: int = 12,
    val_per_class: int = 4,
    test_per_class: int = 4
):
    """
    Generates a lightweight, balanced synthetic dataset to allow immediate
    end-to-end pipeline verification without needing a multi-gigabyte download.
    """
    ensure_directories()
    print("\n[RetinaCare Generator] Creating mini demonstration dataset...")

    splits = [
        ("train", TRAIN_DIR, train_per_class),
        ("validation", VAL_DIR, val_per_class),
        ("test", TEST_DIR, test_per_class)
    ]

    for split_name, split_dir, count in splits:
        for class_id in range(NUM_CLASSES):
            class_folder = split_dir / str(class_id)
            class_folder.mkdir(parents=True, exist_ok=True)
            for idx in range(count):
                seed = (class_id * 1000) + (idx * 37) + (0 if split_name == "train" else 500)
                img = create_synthetic_fundus(class_id=class_id, size=256, seed=seed)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out_path = class_folder / f"{split_name}_{class_id}_{idx:03d}.png"
                cv2.imwrite(str(out_path), bgr)

        print(f"  Populated '{split_name}' with {count * NUM_CLASSES} images ({count} per class).")

    print("\nMini dataset created successfully inside 'dataset/' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic retinal fundus samples")
    parser.add_argument("--create-mini-dataset", action="store_true", help="Generate small train/val/test dataset for pipeline testing")
    args = parser.parse_args()

    generate_all_samples()
    if args.create_mini_dataset:
        generate_mini_dataset()
