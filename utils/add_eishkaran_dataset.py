"""
==============================================================================
Diabetic Retinopathy - Ingest Eishkaran / AGAR300 Microaneurysm Dataset
Department of Computer Science & Engineering - GM University

Dataset Source:
https://www.kaggle.com/datasets/eishkaran/diabetes-using-retinopathy-prediction

Clinical Context:
The AGAR300 dataset uploaded by Eishkaran Singh contains 28 high-resolution 
fundus images specifically documenting early microaneurysms (MAs). In the 
international ICDR 5-stage scale used by this project, microaneurysms without
widespread hemorrhages represent Stage 1: Mild NPDR.

This script:
1. Locates the downloaded zip archive or extracted folder.
2. Extracts and indexes all fundus images.
3. Maps these images into Class 1 (Mild NPDR).
4. Distributes them (70% train, 15% val, 15% test) to augment existing data.
5. Displays updated class distribution and training instructions.
==============================================================================
"""

from __future__ import annotations

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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


def find_source_zip(custom_path: Optional[str] = None) -> Optional[Path]:
    """Locates the dataset zip or extracted folder."""
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.exists():
            return p
        print(f"[Error] Specified source path does not exist: {p}")
        return None

    # Search candidates
    search_dirs = [
        Path.home() / "Downloads",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT
    ]

    zip_patterns = [
        "*diabetes-using-retinopathy-prediction*.zip",
        "*retinopathy*.zip",
        "archive*.zip"
    ]

    for d in search_dirs:
        if not d.exists():
            continue
        for pat in zip_patterns:
            for candidate in d.glob(pat):
                if candidate.is_file():
                    return candidate

    return None


def extract_images(source: Path, temp_extract_dir: Path) -> List[Path]:
    """Extracts zip if needed and collects all valid fundus image paths."""
    if source.is_file() and source.suffix.lower() == ".zip":
        temp_extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RetinaCare] Extracting archive: {source.name}...")
        with zipfile.ZipFile(source, "r") as z:
            z.extractall(temp_extract_dir)
        scan_root = temp_extract_dir
    else:
        scan_root = source

    images = [
        p for p in scan_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
    return sorted(images)


def count_all_classes() -> None:
    """Prints current dataset distribution across all 5 classes."""
    print("\n" + "=" * 70)
    print(f"{'CURRENT DATASET DISTRIBUTION':^70}")
    print("=" * 70)
    print(f"{'Class':<8} | {'Clinical Stage':<22} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 70)

    totals = {"train": 0, "val": 0, "test": 0, "all": 0}

    for c in range(NUM_CLASSES):
        tr = len(list((TRAIN_DIR / str(c)).glob("*.*"))) if (TRAIN_DIR / str(c)).exists() else 0
        va = len(list((VAL_DIR / str(c)).glob("*.*"))) if (VAL_DIR / str(c)).exists() else 0
        te = len(list((TEST_DIR / str(c)).glob("*.*"))) if (TEST_DIR / str(c)).exists() else 0
        tot = tr + va + te

        totals["train"] += tr
        totals["val"] += va
        totals["test"] += te
        totals["all"] += tot

        print(f"Class {c:<2} | {CLASS_SHORT_NAMES[c]:<22} | {tr:<8} | {va:<8} | {te:<8} | {tot:<8}")

    print("-" * 70)
    print(f"{'TOTALS':<8} | {'All 5 Stages':<22} | {totals['train']:<8} | {totals['val']:<8} | {totals['test']:<8} | {totals['all']:<8}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Eishkaran / AGAR300 Microaneurysm dataset into Class 1 (Mild NPDR)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to downloaded zip archive or extracted folder."
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=1,
        choices=[0, 1, 2, 3, 4],
        help="Target clinical class (default: 1 for Mild NPDR microaneurysms)"
    )
    args = parser.parse_args()

    ensure_directories()

    source = find_source_zip(args.source)
    if source is None:
        print("\n" + "!" * 72)
        print("[RetinaCare Notice] Could not automatically locate the dataset zip.")
        print("1. Download the dataset from Kaggle:")
        print("   -> https://www.kaggle.com/datasets/eishkaran/diabetes-using-retinopathy-prediction")
        print("2. Run:")
        print('   python utils/add_eishkaran_dataset.py --source "path/to/archive.zip"')
        print("!" * 72 + "\n")
        sys.exit(1)

    print(f"[RetinaCare] Found source: {source}")
    temp_dir = PROJECT_ROOT / "dataset" / "temp_eishkaran_extract"
    images = extract_images(source, temp_dir)

    print(f"[RetinaCare] Discovered {len(images)} fundus images.")
    if len(images) == 0:
        print("[RetinaCare Error] No valid images found in source.")
        sys.exit(1)

    # 70% train, 15% val, 15% test split
    n = len(images)
    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))
    
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    target_class = str(args.class_id)
    (TRAIN_DIR / target_class).mkdir(parents=True, exist_ok=True)
    (VAL_DIR / target_class).mkdir(parents=True, exist_ok=True)
    (TEST_DIR / target_class).mkdir(parents=True, exist_ok=True)

    copied = {"train": 0, "val": 0, "test": 0}

    for img in train_imgs:
        dest = TRAIN_DIR / target_class / f"agar300_{img.name}"
        shutil.copy2(img, dest)
        copied["train"] += 1

    for img in val_imgs:
        dest = VAL_DIR / target_class / f"agar300_{img.name}"
        shutil.copy2(img, dest)
        copied["val"] += 1

    for img in test_imgs:
        dest = TEST_DIR / target_class / f"agar300_{img.name}"
        shutil.copy2(img, dest)
        copied["test"] += 1

    # Cleanup temp extraction
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"[RetinaCare] Successfully added {len(images)} images to Class {target_class} ({CLASS_NAMES[args.class_id]}):")
    print(f"  - Train: {copied['train']}")
    print(f"  - Validation: {copied['val']}")
    print(f"  - Test: {copied['test']}")

    count_all_classes()

    print("\nNext step: Train the model by running:")
    print("  python src/train.py --model MobileNetV2 --epochs 15 --fine-tune-epochs 10\n")


if __name__ == "__main__":
    main()
