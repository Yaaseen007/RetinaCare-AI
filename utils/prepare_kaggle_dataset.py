"""
==============================================================================
Diabetic Retinopathy Detection - Kaggle Dataset Preparation Utility
Department of Computer Science & Engineering - GM University

Dataset Source:
https://www.kaggle.com/datasets/shajinrp/diabetic-retinopathy

This script automates:
1. Locating the downloaded Kaggle dataset (.zip archive or extracted folder).
2. Parsing the 5 clinical stages:
     - Stage 0: No_DR
     - Stage 1: Mild
     - Stage 2: Moderate
     - Stage 3: Severe
     - Stage 4: Proliferate_DR / Proliferative
3. Performing a stratified train / validation / test split (70% / 15% / 15% default)
   strictly before model fitting, following ML best practices.
4. Organizing the images into:
     dataset/train/{0,1,2,3,4}
     dataset/validation/{0,1,2,3,4}
     dataset/test/{0,1,2,3,4}
5. Printing full class balance statistics and next-step training commands.
==============================================================================
"""

from __future__ import annotations

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_SHORT_NAMES,
    VALID_EXTENSIONS,
    ensure_directories
)

# Folder name mapping heuristics for Kaggle datasets (shajinrp/diabetic-retinopathy and derivatives)
CLASS_ALIASES: Dict[int, List[str]] = {
    0: ["0", "no_dr", "nodr", "no diabetic retinopathy", "normal", "healthy", "non_dr"],
    1: ["1", "mild", "mild_npdr", "mild npdr", "mild_dr"],
    2: ["2", "moderate", "moderate_npdr", "moderate npdr", "mod_dr"],
    3: ["3", "severe", "severe_npdr", "severe npdr", "sev_dr"],
    4: ["4", "proliferate_dr", "proliferate", "proliferative_dr", "proliferative", "pdr"]
}


def find_source_dataset(user_path: Optional[str] = None) -> Optional[Path]:
    """
    Locates the dataset source. If user_path is not specified, searches common locations:
    - User's Downloads directory (for archive.zip or diabetic-retinopathy.zip)
    - Project root / dataset directory
    """
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if p.exists():
            return p
        print(f"[RetinaCare Error] Specified source does not exist: {p}")
        return None

    # Auto-detection candidate directories
    search_dirs = [
        Path.home() / "Downloads",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT
    ]

    # Look for zip archives first
    for d in search_dirs:
        if not d.exists():
            continue
        for candidate in d.glob("*retinopathy*.zip"):
            if candidate.is_file():
                return candidate
        for candidate in d.glob("archive*.zip"):
            if candidate.is_file():
                return candidate

    # Look for extracted folders
    for d in search_dirs:
        if not d.exists():
            continue
        for candidate in d.iterdir():
            if candidate.is_dir() and any(k in candidate.name.lower() for k in ["retinopathy", "colored_images", "dr_dataset"]):
                return candidate

    return None


def extract_if_zip(source_path: Path, extract_target: Path) -> Path:
    """Extracts a zip file into a target directory if needed."""
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        extract_target.mkdir(parents=True, exist_ok=True)
        print(f"[RetinaCare Extractor] Extracting {source_path.name} to {extract_target}...")
        with zipfile.ZipFile(source_path, "r") as zf:
            zf.extractall(extract_target)
        print("[RetinaCare Extractor] Extraction complete.")
        return extract_target
    return source_path


def map_directory_to_classes(root_dir: Path) -> Dict[int, List[Path]]:
    """
    Scans the extracted directory recursively to find image files and match
    their parent folder names to clinical DR class indices (0 to 4).
    """
    class_images: Dict[int, List[Path]] = {i: [] for i in range(NUM_CLASSES)}

    # Collect all image files
    all_image_paths = []
    for ext in VALID_EXTENSIONS:
        all_image_paths.extend(root_dir.rglob(f"*{ext}"))
        all_image_paths.extend(root_dir.rglob(f"*{ext.upper()}"))
    all_image_paths = sorted(list(set(all_image_paths)))

    if not all_image_paths:
        return class_images

    # Match each image to a class based on its parent folder name
    for img_path in all_image_paths:
        parent_name = img_path.parent.name.lower().strip()
        matched_class = None

        for class_id, aliases in CLASS_ALIASES.items():
            if parent_name in aliases:
                matched_class = class_id
                break

        # Fallback: check grandparents if inside sub-subfolders
        if matched_class is None:
            grandparent = img_path.parent.parent.name.lower().strip()
            for class_id, aliases in CLASS_ALIASES.items():
                if grandparent in aliases:
                    matched_class = class_id
                    break

        if matched_class is not None:
            class_images[matched_class].append(img_path)

    return class_images


def split_and_organize_dataset(
    class_images: Dict[int, List[Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    clean_existing: bool = True
) -> Dict[str, Dict[int, int]]:
    """
    Performs stratified partitioning on the collected images and copies them
    into dataset/train/{0..4}, dataset/validation/{0..4}, and dataset/test/{0..4}.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    np.random.seed(seed)
    ensure_directories()

    split_counts: Dict[str, Dict[int, int]] = {
        "train": {i: 0 for i in range(NUM_CLASSES)},
        "validation": {i: 0 for i in range(NUM_CLASSES)},
        "test": {i: 0 for i in range(NUM_CLASSES)}
    }

    # Optionally clean prior split files to prevent mixing
    if clean_existing:
        for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            for class_id in range(NUM_CLASSES):
                folder = split_dir / str(class_id)
                if folder.exists():
                    shutil.rmtree(folder)
                folder.mkdir(parents=True, exist_ok=True)

    print("\n[RetinaCare Organizer] Partitioning images with stratified sampling...")

    for class_id in range(NUM_CLASSES):
        images = list(class_images[class_id])
        np.random.shuffle(images)
        n_total = len(images)

        if n_total == 0:
            continue

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Allocate remainder to test to prevent losing any sample
        n_test = n_total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        splits = [
            ("train", TRAIN_DIR / str(class_id), train_imgs),
            ("validation", VAL_DIR / str(class_id), val_imgs),
            ("test", TEST_DIR / str(class_id), test_imgs)
        ]

        for s_name, dest_dir, img_subset in splits:
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src_file in img_subset:
                # Retain unique identifier in filename
                dest_file = dest_dir / f"c{class_id}_{src_file.name}"
                if not dest_file.exists():
                    shutil.copy2(src_file, dest_file)
            split_counts[s_name][class_id] = len(img_subset)

    return split_counts


def print_summary_table(counts: Dict[str, Dict[int, int]]):
    """Prints a clear markdown summary table of the prepared splits."""
    print("\n" + "=" * 76)
    print("      RETINACARE AI - DATASET PARTITIONING SUMMARY (ICDR 5-CLASS)")
    print("=" * 76)
    header = f"{'Class ID':<10} | {'Clinical Stage':<22} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}"
    print(header)
    print("-" * 76)

    totals = {"train": 0, "validation": 0, "test": 0, "overall": 0}

    for c in range(NUM_CLASSES):
        tr = counts["train"][c]
        va = counts["validation"][c]
        te = counts["test"][c]
        tot = tr + va + te

        totals["train"] += tr
        totals["validation"] += va
        totals["test"] += te
        totals["overall"] += tot

        stage_name = CLASS_SHORT_NAMES[c]
        print(f"Class {c:<4} | {stage_name:<22} | {tr:<8} | {va:<8} | {te:<8} | {tot:<8}")

    print("-" * 76)
    print(f"{'TOTALS':<10} | {'All 5 Classes':<22} | {totals['train']:<8} | {totals['validation']:<8} | {totals['test']:<8} | {totals['overall']:<8}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Kaggle Diabetic Retinopathy dataset (shajinrp/diabetic-retinopathy)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to downloaded zip file or extracted directory containing the 5 class folders."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of images for training (default: 0.70)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of images for validation (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of images for testing (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified splitting (default: 42)"
    )
    args = parser.parse_args()

    source = find_source_dataset(args.source)
    if source is None:
        print("\n" + "!" * 76)
        print("[RetinaCare Notice] Could not automatically find the Kaggle dataset.")
        print("Please download 'diabetic-retinopathy.zip' from Kaggle:")
        print("  🔗 https://www.kaggle.com/datasets/shajinrp/diabetic-retinopathy")
        print("\nThen run this script specifying the download path:")
        print("  python utils/prepare_kaggle_dataset.py --source \"path/to/archive.zip\"")
        print("!" * 76 + "\n")
        sys.exit(1)

    print(f"[RetinaCare] Found source: {source}")

    # Temporary directory for zip extraction if needed
    temp_extract_dir = PROJECT_ROOT / "dataset" / "raw_extracted"
    extracted_root = extract_if_zip(source, temp_extract_dir)

    print(f"[RetinaCare] Scanning images in: {extracted_root}...")
    class_images = map_directory_to_classes(extracted_root)

    total_detected = sum(len(imgs) for imgs in class_images.values())
    print(f"[RetinaCare] Detected {total_detected} total images across classes.")

    if total_detected == 0:
        print("[RetinaCare Error] No images could be matched to the 5 DR classes.")
        print(f"Ensure subdirectories match: {list(CLASS_ALIASES.keys())} or {list(CLASS_ALIASES.values())}")
        sys.exit(1)

    for c, imgs in class_images.items():
        print(f"  Class {c} ({CLASS_SHORT_NAMES[c]}): {len(imgs)} images")

    # Perform stratified split and population
    split_counts = split_and_organize_dataset(
        class_images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    print_summary_table(split_counts)

    print("\nDataset setup complete!")
    print("You can now train the model using:")
    print("  python src/train.py --model MobileNetV2 --epochs 15 --fine-tune-epochs 10")
    print("Or evaluate with:")
    print("  python src/evaluate.py")
    print("And run the web UI with:")
    print("  streamlit run app.py\n")


if __name__ == "__main__":
    main()
