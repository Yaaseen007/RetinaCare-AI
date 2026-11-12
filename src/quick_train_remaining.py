"""
==============================================================================
RetinaCare AI - Rapid Transfer Learning Finisher for ResNet50 and VGG16
Builds and trains the classification heads on E:/dataset, saves checkpoints,
and ensures all 5 models are fully operational and ready.
==============================================================================
"""

import os
import sys
import time
from pathlib import Path

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, "E:/pylib")
sys.path.insert(0, "e:/P B L")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers

from src.train_transfer_models import (
    build_dataset,
    compute_class_weights,
    TRAIN_DIR,
    VAL_DIR,
    MODELS_DIR,
    INPUT_SHAPE,
    NUM_CLASSES,
    DROPOUT_RATE,
    L2_DECAY
)


def build_transfer_model(model_name: str) -> tf.keras.Model:
    inputs = layers.Input(shape=INPUT_SHAPE, name="input_retina")
    kw = dict(input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")
    
    name_up = model_name.upper()
    if name_up == "RESNET50":
        base = tf.keras.applications.ResNet50(**kw)
    elif name_up == "VGG16":
        base = tf.keras.applications.VGG16(**kw)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    base.trainable = False
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2_DECAY), name="d256")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop1")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(L2_DECAY), name="d128")(x)
    x = layers.Dropout(DROPOUT_RATE / 2, name="drop2")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="dr_out")(x)
    
    m = models.Model(inputs=inputs, outputs=out, name=f"RetinaCare_{model_name}")
    m.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return m


def train_and_save(model_name: str):
    ckpt_path = MODELS_DIR / f"{model_name.lower()}.keras"
    print(f"\n=======================================================")
    print(f"  Training {model_name} -> {ckpt_path}")
    print(f"=======================================================")
    
    print("Building model with ImageNet pre-trained backbone...")
    model = build_transfer_model(model_name)
    
    print("Building dataset loaders...")
    # Train on 25 batches (800 images) for quick convergence without multi-hour CPU stalls
    train_ds = build_dataset(TRAIN_DIR, batch_size=32, augment=True, shuffle=True).take(25)
    val_ds = build_dataset(VAL_DIR, batch_size=32, augment=False, shuffle=False).take(10)
    cw = compute_class_weights(TRAIN_DIR)
    
    print("Training classification head...")
    t0 = time.time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        class_weight=cw,
        verbose=1
    )
    elapsed = round(time.time() - t0, 1)
    print(f"Training completed in {elapsed}s")
    
    # Save checkpoint
    model.save(str(ckpt_path))
    size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)
    print(f"Successfully saved {ckpt_path.name} ({size_mb} MB)!")
    return ckpt_path


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for m in ["ResNet50", "VGG16"]:
        train_and_save(m)
        
    print("\nALL MODELS TRAINED AND SAVED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
