"""
==============================================================================
Train & Save VGG16 for RetinaCare AI
==============================================================================
"""
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, "E:/pylib")
sys.path.insert(0, "e:/P B L")

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


def main():
    ckpt_path = MODELS_DIR / "vgg16.keras"
    print(f"Building VGG16 model with pre-trained weights...")
    
    inputs = layers.Input(shape=INPUT_SHAPE, name="input_retina")
    base = tf.keras.applications.VGG16(input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")
    base.trainable = False
    
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2_DECAY), name="d256")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop1")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(L2_DECAY), name="d128")(x)
    x = layers.Dropout(DROPOUT_RATE / 2, name="drop2")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="dr_out")(x)
    
    model = models.Model(inputs=inputs, outputs=out, name="RetinaCare_VGG16")
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("Loading datasets...")
    train_ds = build_dataset(TRAIN_DIR, batch_size=32, augment=True, shuffle=True).take(15)
    val_ds = build_dataset(VAL_DIR, batch_size=32, augment=False, shuffle=False).take(5)
    cw = compute_class_weights(TRAIN_DIR)
    
    print("Training 1 epoch to calibrate weights...")
    model.fit(train_ds, validation_data=val_ds, epochs=1, class_weight=cw, verbose=1)
    
    print(f"Saving checkpoint to {ckpt_path}...")
    model.save(str(ckpt_path))
    size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)
    print(f"Successfully saved {ckpt_path.name} ({size_mb} MB)!")


if __name__ == "__main__":
    main()
