"""
==============================================================================
Diabetic Retinopathy Detection - Model Training Pipeline
Department of Computer Science & Engineering - GM University

This script executes the complete end-to-end training pipeline:
1. Dataset verification across train, validation, and test splits
2. Class distribution analysis and visualization (saving results/class_distribution.png)
3. Class weight calculation to mitigate severe medical dataset class imbalance
4. Medically-realistic data augmentation pipeline (rotation, zoom, shift, contrast)
5. Transfer learning model initialization (MobileNetV2, EfficientNetB0, etc.)
6. Phase 1: Feature Extraction (Frozen ImageNet backbone)
7. Phase 2: Fine-Tuning (Unfreezing top backbone layers with reduced learning rate)
8. Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (best_model.keras)
9. Training history curve plotting (results/training_history.png)
10. Automated test set evaluation and metrics generation
==============================================================================
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_PYLIB = Path("E:/pylib")
if _PYLIB.exists() and str(_PYLIB) not in sys.path:
    sys.path.insert(0, str(_PYLIB))

from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import callbacks

from utils.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODELS_DIR,
    BEST_MODEL_PATH,
    RESULTS_DIR,
    CLASS_DISTRIBUTION_PATH,
    TRAINING_HISTORY_PLOT_PATH,
    IMAGE_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_SHORT_NAMES,
    CLASS_COLORS,
    DEFAULT_MODEL_NAME,
    BATCH_SIZE,
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    INITIAL_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LEARNING_RATE,
    RANDOM_SEED,
    ensure_directories
)
from src.model import (
    build_transfer_learning_model,
    compile_model,
    unfreeze_top_layers,
    get_model_summary
)
from src.evaluate import evaluate_dataset, update_model_comparison_record


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Model for Diabetic Retinopathy Detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        choices=["MobileNetV2", "EfficientNetB0", "ResNet50", "VGG16", "CustomCNN"],
        help=f"Model architecture to train (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=INITIAL_EPOCHS,
        help=f"Number of initial training epochs with frozen base (default: {INITIAL_EPOCHS})"
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=FINE_TUNE_EPOCHS,
        help=f"Number of fine-tuning epochs with unfrozen layers (default: {FINE_TUNE_EPOCHS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Training batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=INITIAL_LEARNING_RATE,
        help=f"Initial learning rate (default: {INITIAL_LEARNING_RATE})"
    )
    parser.add_argument(
        "--no-fine-tune",
        action="store_true",
        help="Skip fine-tuning phase and perform only initial feature extraction"
    )
    return parser.parse_args()


def count_split_samples(split_dir: Path) -> Dict[int, int]:
    """Counts valid image samples per class in a directory split."""
    counts = {}
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for class_id in range(NUM_CLASSES):
        class_folder = split_dir / str(class_id)
        if class_folder.exists() and class_folder.is_dir():
            n = len([p for p in class_folder.iterdir() if p.suffix.lower() in valid_exts])
            counts[class_id] = n
        else:
            counts[class_id] = 0
    return counts


def plot_class_distribution(
    train_counts: Dict[int, int],
    val_counts: Dict[int, int],
    test_counts: Dict[int, int],
    output_path: Path = CLASS_DISTRIBUTION_PATH
) -> None:
    """
    Visualizes and saves the dataset class distribution across all splits.
    Helps identify class imbalance and motivates class weighting.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classes = [CLASS_SHORT_NAMES[i] for i in range(NUM_CLASSES)]
    train_vals = [train_counts.get(i, 0) for i in range(NUM_CLASSES)]
    val_vals = [val_counts.get(i, 0) for i in range(NUM_CLASSES)]
    test_vals = [test_counts.get(i, 0) for i in range(NUM_CLASSES)]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    rects1 = ax.bar(x - width, train_vals, width, label="Train", color="#0284c7")
    rects2 = ax.bar(x, val_vals, width, label="Validation", color="#38bdf8")
    rects3 = ax.bar(x + width, test_vals, width, label="Test", color="#94a3b8")

    ax.set_ylabel("Number of Fundus Images", fontsize=12, fontweight="semibold")
    ax.set_title("Diabetic Retinopathy Dataset - Class Distribution Across Splits", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=10)
    ax.legend(frameon=True, fontsize=10)

    # Value labels
    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def compute_dataset_class_weights(train_counts: Dict[int, int]) -> Dict[int, float]:
    """
    Computes inverse frequency class weights to balance gradient updates.
    
    Formula (Sklearn):
        weight[c] = total_samples / (n_classes * count[c])
        
    Medical Rationale:
    In clinical retinal datasets (e.g. APTOS 2019), Class 0 (No DR) typically comprises
    over 50-70% of the images, while Class 3 (Severe) and Class 4 (Proliferative) are
    infrequent. Without class weights, the model learns a biased trivial shortcut to
    predict Class 0, yielding dangerously low clinical sensitivity for severe disease.
    """
    labels = []
    for class_id, count in train_counts.items():
        labels.extend([class_id] * count)

    if not labels:
        return {i: 1.0 for i in range(NUM_CLASSES)}

    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=np.array(labels)
    )

    weight_dict = {i: 1.0 for i in range(NUM_CLASSES)}
    for cls, w in zip(unique_classes, weights):
        weight_dict[int(cls)] = float(w)

    return weight_dict


def build_data_augmentation_layer() -> tf.keras.Sequential:
    """
    Creates a medically realistic Keras augmentation sequential pipeline.
    
    Clinical Justification:
    - RandomFlip (Horizontal): Fundus cameras view both left and right eyes (OD vs OS).
      Horizontal mirroring preserves physiological validity.
    - RandomRotation (8% ~ +/-28 deg): Accommodates patient head tilt during examination.
    - RandomZoom (8%): Accounts for subtle differences in camera focal distance.
    - RandomTranslation (5%): Handles decentered positioning of optic disc and macula.
    - RandomContrast (10%): Simulates slight pupil dilation and lens transparency variations.
    
    Extreme transforms (e.g., severe elastic shear or heavy blur) are deliberately avoided
    to prevent destroying microaneurysms (<30 microns) or small hemorrhages.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
        tf.keras.layers.RandomRotation(0.08, seed=RANDOM_SEED),
        tf.keras.layers.RandomZoom(0.08, seed=RANDOM_SEED),
        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05, seed=RANDOM_SEED),
        tf.keras.layers.RandomContrast(0.10, seed=RANDOM_SEED)
    ], name="retinal_data_augmentation")


def load_dataset_splits(
    batch_size: int = BATCH_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset]]:
    """
    Loads training, validation, and test datasets using TensorFlow image_dataset_from_directory.
    Applies image resizing to (224, 224) and rescales pixels to [0, 1].
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(TRAIN_DIR),
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(VAL_DIR),
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=False
    )

    test_counts = count_split_samples(TEST_DIR)
    has_test_data = sum(test_counts.values()) > 0

    test_ds = None
    if has_test_data:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(TEST_DIR),
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=batch_size,
            image_size=IMAGE_SIZE,
            shuffle=False
        )

    # Rescaling and prefetching optimization
    rescale_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
    data_augmentation = build_data_augmentation_layer()

    # Train dataset: Augmentation + Rescaling + Prefetch
    train_ds = train_ds.map(
        lambda x, y: (rescale_layer(data_augmentation(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Validation dataset: Rescaling + Prefetch (No augmentation)
    val_ds = val_ds.map(
        lambda x, y: (rescale_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    if test_ds is not None:
        test_ds = test_ds.map(
            lambda x, y: (rescale_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def plot_training_history(
    history_phase1: tf.keras.callbacks.History,
    history_phase2: Optional[tf.keras.callbacks.History] = None,
    output_path: Path = TRAINING_HISTORY_PLOT_PATH
) -> None:
    """
    Plots training and validation curves across both training phases.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    acc = history_phase1.history.get("accuracy", [])
    val_acc = history_phase1.history.get("val_accuracy", [])
    loss = history_phase1.history.get("loss", [])
    val_loss = history_phase1.history.get("val_loss", [])

    initial_epochs = len(acc)

    if history_phase2 is not None:
        acc = acc + history_phase2.history.get("accuracy", [])
        val_acc = val_acc + history_phase2.history.get("val_accuracy", [])
        loss = loss + history_phase2.history.get("loss", [])
        val_loss = val_loss + history_phase2.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    # Accuracy subplot
    ax1.plot(epochs_range, acc, label="Training Accuracy", color="#0284c7", lw=2)
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="#10b981", lw=2)
    if history_phase2 is not None:
        ax1.axvline(x=initial_epochs, color="#ef4444", linestyle="--", label="Fine-Tuning Started")
    ax1.set_title("Model Accuracy over Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.legend(loc="lower right")
    ax1.set_ylim([0.0, 1.05])

    # Loss subplot
    ax2.plot(epochs_range, loss, label="Training Loss", color="#0284c7", lw=2)
    ax2.plot(epochs_range, val_loss, label="Validation Loss", color="#10b981", lw=2)
    if history_phase2 is not None:
        ax2.axvline(x=initial_epochs, color="#ef4444", linestyle="--", label="Fine-Tuning Started")
    ax2.set_title("Model Loss over Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss (Sparse Categorical Crossentropy)", fontsize=11)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    # Save numeric history
    history_json_path = RESULTS_DIR / "training_history.json"
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": [float(x) for x in acc],
            "val_accuracy": [float(x) for x in val_acc],
            "loss": [float(x) for x in loss],
            "val_loss": [float(x) for x in val_loss],
            "fine_tune_start_epoch": initial_epochs if history_phase2 is not None else None
        }, f, indent=4)


def main():
    """Main training execution entry point."""
    args = parse_arguments()
    ensure_directories()

    print("\n" + "=" * 70)
    print("  RETINACARE AI - DIABETIC RETINOPATHY CNN TRAINING PIPELINE")
    print("  B.Tech CSE Project Based Learning (PBL) | GM University")
    print("=" * 70)
    print(f"Target Architecture  : {args.model}")
    print(f"Initial Epochs       : {args.epochs}")
    print(f"Fine-Tuning Epochs   : {args.fine_tune_epochs if not args.no_fine_tune else 'Disabled'}")
    print(f"Batch Size           : {args.batch_size}")
    print(f"Initial LR           : {args.lr}")
    print("-" * 70)

    # 1. Dataset Verification
    train_counts = count_split_samples(TRAIN_DIR)
    val_counts = count_split_samples(VAL_DIR)
    test_counts = count_split_samples(TEST_DIR)

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_test = sum(test_counts.values())

    print("\n[Step 1/6] Verifying Dataset Splits:")
    print(f"  Training Samples    : {total_train}")
    print(f"  Validation Samples  : {total_val}")
    print(f"  Testing Samples     : {total_test}")

    if total_train == 0 or total_val == 0:
        print("\n" + "!" * 70)
        print("ERROR: Training and/or Validation datasets are empty!")
        print("Please place retinal fundus images inside 'dataset/train/{0,1,2,3,4}' and")
        print("'dataset/validation/{0,1,2,3,4}' following instructions in dataset/README.md.")
        print("\nTip: To quickly test the pipeline with synthetic sample data, run:")
        print("     python utils/generate_samples.py --create-mini-dataset")
        print("!" * 70 + "\n")
        sys.exit(1)

    # 2. Class Distribution Analysis
    print("\n[Step 2/6] Analyzing Class Distribution:")
    for class_id, name in enumerate(CLASS_NAMES):
        t_cnt = train_counts.get(class_id, 0)
        pct = (t_cnt / total_train) * 100 if total_train > 0 else 0
        print(f"  Class {class_id} [{name:<35}]: {t_cnt:>5} images ({pct:5.1f}%)")

    plot_class_distribution(train_counts, val_counts, test_counts)
    print(f"  Saved class distribution chart -> {CLASS_DISTRIBUTION_PATH}")

    # 3. Class Imbalance Mitigation
    class_weights = compute_dataset_class_weights(train_counts)
    print("\n[Step 3/6] Computed Balanced Class Weights:")
    for cid, w in class_weights.items():
        print(f"  Class {cid} ({CLASS_SHORT_NAMES[cid]}): weight = {w:.3f}")

    # 4. Data Loading
    print("\n[Step 4/6] Initializing Data Pipelines & Augmentation...")
    train_ds, val_ds, test_ds = load_dataset_splits(batch_size=args.batch_size)

    # 5. Model Construction
    print(f"\n[Step 5/6] Building Transfer Learning Model ({args.model})...")
    model = build_transfer_learning_model(
        model_name=args.model,
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        num_classes=NUM_CLASSES,
        trainable_base=False
    )
    model = compile_model(model, learning_rate=args.lr)

    # Callbacks configuration
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
        verbose=1
    )

    cb_list = [model_checkpoint, early_stopping, reduce_lr]

    # --- Phase 1: Feature Extraction ---
    start_time = time.time()
    print("\n>>> Starting Phase 1: Training Classification Head with Frozen Base...")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=cb_list
    )

    # --- Phase 2: Fine Tuning ---
    history_phase2 = None
    if not args.no_fine_tune and args.model.upper() != "CUSTOMCNN":
        print("\n>>> Starting Phase 2: Unfreezing Top Layers for Fine-Tuning...")
        model = unfreeze_top_layers(
            model=model,
            num_layers_to_unfreeze=30,
            fine_tune_lr=FINE_TUNE_LEARNING_RATE
        )
        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=len(history_phase1.history["loss"]),
            class_weight=class_weights,
            callbacks=cb_list
        )

    training_time = time.time() - start_time
    print(f"\n[Training Completed in {training_time / 60:.2f} minutes]")

    # Save training curves
    plot_training_history(history_phase1, history_phase2)
    print(f"Saved training history curves -> {TRAINING_HISTORY_PLOT_PATH}")

    # 6. Evaluation
    print("\n[Step 6/6] Evaluating on Test Dataset...")
    eval_target_ds = test_ds if test_ds is not None else val_ds
    eval_name = "Test Split" if test_ds is not None else "Validation Split (No test split found)"
    print(f"Evaluating using: {eval_name}")

    metrics = evaluate_dataset(
        model=model,
        test_data=eval_target_ds,
        model_name=args.model,
        output_dir=RESULTS_DIR
    )

    # Update comparison record with training time
    update_model_comparison_record(
        model_name=args.model,
        accuracy=metrics["accuracy"],
        precision=metrics["macro_precision"],
        recall=metrics["macro_recall"],
        f1=metrics["macro_f1"],
        roc_auc=metrics["macro_roc_auc"],
        training_time_sec=training_time
    )

    print("\n" + "=" * 70)
    print("  TRAINING & EVALUATION SUMMARY")
    print(f"  Architecture      : {args.model}")
    print(f"  Overall Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro F1-Score    : {metrics['macro_f1'] * 100:.2f}%")
    print(f"  Macro ROC-AUC     : {metrics['macro_roc_auc']:.4f}")
    print(f"  Best Model Saved  : {BEST_MODEL_PATH}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
