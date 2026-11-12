# Retinal Fundus Dataset Setup Guide

This directory holds the retinal fundus image datasets used for training, validating, and testing deep learning models for Diabetic Retinopathy detection.

---

## 1. Supported Clinical Datasets

This project is architected to work primarily with:
* **APTOS 2019 Blindness Detection Dataset** (Kaggle)
* **EyePACS (Diabetic Retinopathy Detection)** (Kaggle)
* **Messidor / Messidor-2**
* **IDRiD (Indian Diabetic Retinopathy Image Dataset)**

> **PBL Compliance Notice:** Copyrighted and private medical datasets are **not** committed to this Git repository. Users must download the data from their official sources and place them into the standardized directory structure shown below.

---

## 2. Expected Directory Structure

The training pipeline (`src/train.py`) uses standard PyTorch / TensorFlow directory-based multi-class dataset parsing:

```text
dataset/
├── train/
│   ├── 0/       # No Diabetic Retinopathy
│   ├── 1/       # Mild NPDR
│   ├── 2/       # Moderate NPDR
│   ├── 3/       # Severe NPDR
│   └── 4/       # Proliferative Diabetic Retinopathy
│
├── validation/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
│
└── test/
    ├── 0/
    ├── 1/
    ├── 2/
    ├── 3/
    └── 4/
```

Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.

---

## 3. How to Download & Setup APTOS 2019

### Step 3.1: Download via Kaggle CLI or Web Browser
Visit the official competition page:  
🔗 [https://www.kaggle.com/c/aptos2019-blindness-detection/data](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

Or download using the Kaggle API:
```bash
kaggle competitions download -c aptos2019-blindness-detection
```

Unpack the archive to access `train_images/` and `train.csv`.

### Step 3.2: Splitting `train.csv` into 5 Class Subdirectories
The downloaded APTOS dataset comes with a `train.csv` containing columns `id_code` and `diagnosis` (0 to 4).

Run this simple Python snippet from the project root to automatically organize the images into `dataset/train/`, `dataset/validation/`, and `dataset/test/` splits (e.g. 70% train, 15% validation, 15% test):

```python
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths (adjust to your download location)
CSV_PATH = Path("path/to/train.csv")
IMAGES_DIR = Path("path/to/train_images")
DEST_DIR = Path("dataset")

df = pd.read_csv(CSV_PATH)

# Stratified split to preserve class ratios
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['diagnosis'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['diagnosis'], random_state=42)

splits = [('train', train_df), ('validation', val_df), ('test', test_df)]

for split_name, split_df in splits:
    for _, row in split_df.iterrows():
        img_name = f"{row['id_code']}.png"
        src_file = IMAGES_DIR / img_name
        dest_folder = DEST_DIR / split_name / str(row['diagnosis'])
        dest_folder.mkdir(parents=True, exist_ok=True)
        if src_file.exists():
            shutil.copy(src_file, dest_folder / img_name)

print("Dataset organized successfully!")
```

---

### Step 3.3: Using the Pre-Organized Kaggle Dataset (`shajinrp/diabetic-retinopathy`)
🔗 **Dataset URL:** [https://www.kaggle.com/datasets/shajinrp/diabetic-retinopathy](https://www.kaggle.com/datasets/shajinrp/diabetic-retinopathy)

This dataset contains pre-resized color fundus photographs structured across the 5 clinical stages (`No_DR`, `Mild`, `Moderate`, `Severe`, `Proliferate_DR`).

1. Download the archive (~217 MB) from Kaggle.
2. Run the automated preparation utility from the project root:
   ```bash
   # Automatically detects the zip in your Downloads folder:
   python utils/prepare_kaggle_dataset.py

   # Or specify the exact file path:
   python utils/prepare_kaggle_dataset.py --source "path/to/diabetic-retinopathy.zip"
   ```
This extracts the images, performs a stratified 70% train / 15% val / 15% test split, and places them directly into `dataset/train/{0..4}`, `dataset/validation/{0..4}`, and `dataset/test/{0..4}`.

---

## 4. Immediate Pipeline Smoke Testing (No Downloads Needed)

To test the entire training, evaluation, and Streamlit pipeline without downloading the full multi-gigabyte APTOS archive, generate synthetic demonstration fundus images by running:

```bash
python utils/generate_samples.py --create-mini-dataset
```

This populates `dataset/train/`, `dataset/validation/`, and `dataset/test/` with anatomically authentic synthetic fundus images across all 5 clinical stages.
