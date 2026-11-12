"""
==============================================================================
RetinaCare AI - Transfer Learning Training Pipeline
Train EfficientNetB0, ResNet50, VGG16 on E:/dataset
Department of Computer Science & Engineering - GM University

Dataset Format (E:/dataset):  flat folder + _classes.csv
  CSV header: filename, Mild, Moderate, No_DR, Proliferate_DR, Severe
  (one-hot encoded, argmax gives class)

Class Mapping:
  No_DR          -> 0  (No DR)
  Mild           -> 1
  Moderate       -> 2
  Severe         -> 3
  Proliferate_DR -> 4  (Proliferative DR)

Saves checkpoints to:
  models/efficientnetb0.keras
  models/resnet50.keras
  models/vgg16.keras
==============================================================================
"""

import os, sys, gc, csv, json, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

CURRENT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_PYLIB = Path("E:/pylib")
if _PYLIB.exists() and str(_PYLIB) not in sys.path:
    sys.path.insert(0, str(_PYLIB))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers, callbacks

# Paths
NEW_DATASET_ROOT = Path("E:/dataset")
TRAIN_DIR  = NEW_DATASET_ROOT / "train"
VAL_DIR    = NEW_DATASET_ROOT / "valid"
TEST_DIR   = NEW_DATASET_ROOT / "test"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
IMAGE_SIZE       = (224, 224)
INPUT_SHAPE      = (224, 224, 3)
BATCH_SIZE       = 32
NUM_CLASSES      = 5
RANDOM_SEED      = 42
INITIAL_EPOCHS   = 15
FINE_TUNE_EPOCHS = 10
INITIAL_LR       = 1e-4
FINE_TUNE_LR     = 1e-5
DROPOUT_RATE     = 0.30
L2_DECAY         = 1e-4
EARLY_STOP_PAT   = 6
REDUCE_LR_PAT    = 3

# CSV column -> label
CSV_COL_TO_LABEL = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferate_DR": 4,
}


def read_csv_labels(split_dir: Path) -> Dict[str, int]:
    csv_path = split_dir / "_classes.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"_classes.csv not found in {split_dir}")
    label_map = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            fname = row["filename"].strip()
            one_hot = {col: int(row[col].strip()) for col in CSV_COL_TO_LABEL}
            label_col = max(one_hot, key=one_hot.get)
            label_map[fname] = CSV_COL_TO_LABEL[label_col]
    return label_map


def build_dataset(split_dir: Path, batch_size: int = BATCH_SIZE,
                  augment: bool = False, shuffle: bool = False) -> tf.data.Dataset:
    label_map = read_csv_labels(split_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    filepaths, labels = [], []
    for img_path in sorted(split_dir.iterdir()):
        if img_path.suffix.lower() not in valid_exts:
            continue
        if img_path.name not in label_map:
            continue
        filepaths.append(str(img_path))
        labels.append(label_map[img_path.name])
    if not filepaths:
        raise RuntimeError(f"No labelled images found in {split_dir}")
    dist = {i: labels.count(i) for i in range(NUM_CLASSES)}
    print(f"  [{split_dir.name}] {len(filepaths)} images | dist: {dist}")

    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=RANDOM_SEED),
        layers.RandomRotation(0.08, seed=RANDOM_SEED),
        layers.RandomZoom(0.08, seed=RANDOM_SEED),
        layers.RandomTranslation(0.05, 0.05, seed=RANDOM_SEED),
        layers.RandomContrast(0.10, seed=RANDOM_SEED),
    ], name="aug")
    rescale = layers.Rescaling(1.0 / 255.0)

    def load(path, label):
        raw   = tf.io.read_file(path)
        img   = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img   = tf.image.resize(img, IMAGE_SIZE)
        img   = tf.cast(img, tf.float32)
        return img, label

    def with_aug(img, label):
        return rescale(aug(img, training=True)), label

    def no_aug(img, label):
        return rescale(img), label

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(len(filepaths), seed=RANDOM_SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(with_aug if augment else no_aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compute_class_weights(split_dir: Path) -> Dict[int, float]:
    label_map = read_csv_labels(split_dir)
    counts = {i: 0 for i in range(NUM_CLASSES)}
    for lbl in label_map.values():
        counts[lbl] += 1
    total = sum(counts.values())
    w = {cls: (total / (NUM_CLASSES * cnt) if cnt > 0 else 1.0)
         for cls, cnt in counts.items()}
    print(f"  Class weights: { {k: round(v,3) for k,v in w.items()} }")
    return w


def build_model(model_name: str) -> tf.keras.Model:
    name_up = model_name.upper()
    inputs  = layers.Input(shape=INPUT_SHAPE, name="input_retina")
    kw = dict(input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")
    if   name_up == "EFFICIENTNETB0": base = tf.keras.applications.EfficientNetB0(**kw)
    elif name_up == "RESNET50":       base = tf.keras.applications.ResNet50(**kw)
    elif name_up == "VGG16":          base = tf.keras.applications.VGG16(**kw)
    else: raise ValueError(f"Unsupported: {model_name}")
    base.trainable = False
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(L2_DECAY), name="d256")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop1")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(L2_DECAY), name="d128")(x)
    x = layers.Dropout(DROPOUT_RATE / 2, name="drop2")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="dr_out")(x)
    return models.Model(inputs=inputs, outputs=out, name=f"RetinaCare_{model_name}")


def compile_m(model, lr): 
    model.compile(optimizer=optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def unfreeze_top(model, n=30, lr=FINE_TUNE_LR):
    for lyr in model.layers:
        if isinstance(lyr, tf.keras.Model):
            lyr.trainable = True
            for sub in lyr.layers[:-n]: sub.trainable = False
            break
    return compile_m(model, lr)


def make_cbs(ckpt_path):
    return [
        callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_accuracy",
                                  save_best_only=True, mode="max", verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=EARLY_STOP_PAT,
                                restore_best_weights=True, mode="max", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=REDUCE_LR_PAT, min_lr=1e-6, verbose=1),
    ]


def plot_history(h1, h2, model_name, out_dir):
    acc  = h1.history.get("accuracy", [])
    val  = h1.history.get("val_accuracy", [])
    loss = h1.history.get("loss", [])
    vloss= h1.history.get("val_loss", [])
    n1   = len(acc)
    if h2:
        acc   += h2.history.get("accuracy", [])
        val   += h2.history.get("val_accuracy", [])
        loss  += h2.history.get("loss", [])
        vloss += h2.history.get("val_loss", [])
    ep = range(1, len(acc)+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5), dpi=150)
    ax1.plot(ep, acc, color="#0284c7", lw=2, label="Train Acc")
    ax1.plot(ep, val, color="#10b981", lw=2, label="Val Acc")
    if h2: ax1.axvline(n1, color="#ef4444", ls="--", label="Fine-Tune")
    ax1.set_title(f"{model_name} Accuracy", fontweight="bold"); ax1.legend(); ax1.set_ylim([0,1.05])
    ax2.plot(ep, loss,  color="#0284c7", lw=2, label="Train Loss")
    ax2.plot(ep, vloss, color="#10b981", lw=2, label="Val Loss")
    if h2: ax2.axvline(n1, color="#ef4444", ls="--", label="Fine-Tune")
    ax2.set_title(f"{model_name} Loss", fontweight="bold"); ax2.legend()
    plt.tight_layout()
    out_path = out_dir / f"training_history_{model_name.lower()}.png"
    plt.savefig(out_path); plt.close(fig)
    print(f"  Plot saved -> {out_path}")


def train_one_model(model_name: str, no_fine_tune: bool = False):
    safe = model_name.lower()
    ckpt = MODELS_DIR / f"{safe}.keras"
    print(f"\n{'='*70}\n  Model: {model_name}  |  Checkpoint: {ckpt}\n{'='*70}\n")

    print("Loading datasets...")
    train_ds = build_dataset(TRAIN_DIR, BATCH_SIZE, augment=True, shuffle=True)
    val_ds   = build_dataset(VAL_DIR,   BATCH_SIZE, augment=False, shuffle=False)
    has_test = (TEST_DIR / "_classes.csv").exists()
    test_ds  = build_dataset(TEST_DIR, BATCH_SIZE) if has_test else None
    cw = compute_class_weights(TRAIN_DIR)

    # Phase 1
    print(f"\nPhase 1: Feature Extraction ({INITIAL_EPOCHS} epochs)")
    model = compile_m(build_model(model_name), INITIAL_LR)
    model.summary(line_length=88)
    t0 = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=INITIAL_EPOCHS, class_weight=cw,
                   callbacks=make_cbs(ckpt), verbose=1)
    print(f"  Phase 1 done in {(time.time()-t0)/60:.1f} min")

    # Phase 2
    h2 = None
    if not no_fine_tune:
        print(f"\nPhase 2: Fine-Tuning ({FINE_TUNE_EPOCHS} epochs)")
        model = unfreeze_top(model, n=30, lr=FINE_TUNE_LR)
        t1 = time.time()
        h2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=FINE_TUNE_EPOCHS, class_weight=cw,
                       callbacks=make_cbs(ckpt), verbose=1)
        print(f"  Phase 2 done in {(time.time()-t1)/60:.1f} min")

    model.save(str(ckpt))
    print(f"  Saved -> {ckpt}  ({round(ckpt.stat().st_size/1024**2,2)} MB)")
    plot_history(h1, h2, model_name, RESULTS_DIR)

    metrics = {}
    if test_ds is not None:
        res = model.evaluate(test_ds, verbose=1)
        metrics = {"test_loss": round(res[0],4), "test_accuracy": round(res[1],4)}
        print(f"  Test accuracy: {metrics['test_accuracy']*100:.2f}%")

    # Update comparison JSON
    jpath = RESULTS_DIR / "model_comparison.json"
    data = {}
    if jpath.exists():
        try: data = json.loads(jpath.read_text(encoding="utf-8"))
        except: data = {}
    data[model_name] = {
        "test_accuracy":  metrics.get("test_accuracy", 0),
        "test_loss":      metrics.get("test_loss", 0),
        "dataset":        "E:\\dataset (5-class flat CSV)",
        "epochs_trained": INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    }
    jpath.write_text(json.dumps(data, indent=2), encoding="utf-8")

    del model; gc.collect(); tf.keras.backend.clear_session()
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train transfer models on E:\\dataset")
    p.add_argument("--model", default="all",
                   choices=["EfficientNetB0","ResNet50","VGG16","all"])
    p.add_argument("--no-fine-tune", action="store_true")
    p.add_argument("--epochs", type=int, default=INITIAL_EPOCHS)
    p.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    args = p.parse_args()
    INITIAL_EPOCHS   = args.epochs
    FINE_TUNE_EPOCHS = args.fine_tune_epochs

    print(f"\n{'='*70}")
    print(f"  RetinaCare AI - Transfer Model Trainer")
    print(f"  TensorFlow {tf.__version__}  |  GPU: {bool(tf.config.list_physical_devices('GPU'))}")
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    print(f"{'='*70}")

    targets = ["EfficientNetB0","ResNet50","VGG16"] if args.model=="all" else [args.model]
    results = {}
    for m in targets:
        try:
            results[m] = train_one_model(m, no_fine_tune=args.no_fine_tune)
        except Exception as e:
            import traceback; traceback.print_exc()

    print(f"\n{'='*70}  Training Complete")
    for n,r in results.items():
        print(f"  {n:<20} test acc: {r.get('test_accuracy','N/A')}")
    print(f"  Checkpoints in: {MODELS_DIR}")
    print(f"  App will show models as READY after training.\n")
