"""
==============================================================================
Diabetic Retinopathy Detection - Model Architecture & Transfer Learning
Department of Computer Science & Engineering - GM University

This module implements deep learning architectures for 5-class severity
classification of Diabetic Retinopathy.

Supported Architectures:
1. MobileNetV2 (Primary lightweight, low-latency, student-laptop friendly)
2. EfficientNetB0 (State-of-the-art compound-scaling convolutional network)
3. ResNet50 (Deep residual network baseline)
4. VGG16 (Classic deep network for architectural comparison)
5. CustomCNN (4-stage convolutional baseline built from scratch)

Transfer Learning Strategy:
- Stage 1 (Feature Extraction): Base model initialized with ImageNet weights,
  frozen (base_model.trainable = False). Custom top classification head
  trained with Adam (lr=1e-4).
- Stage 2 (Fine-Tuning): Unfreeze top 20-40 layers of backbone. Train with
  a small learning rate (lr=1e-5) to adapt specialized retinal filters.
==============================================================================
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

# Ensure project root and libraries are in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_PYLIB = Path("E:/pylib")
if _PYLIB.exists() and str(_PYLIB) not in sys.path:
    sys.path.insert(0, str(_PYLIB))

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers

try:
    from utils.config import (
        INPUT_SHAPE,
        NUM_CLASSES,
        DROPOUT_RATE,
        L2_WEIGHT_DECAY,
        INITIAL_LEARNING_RATE,
        FINE_TUNE_LEARNING_RATE,
        DEFAULT_MODEL_NAME
    )
except ImportError:
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 5
    DROPOUT_RATE = 0.3
    L2_WEIGHT_DECAY = 1e-4
    INITIAL_LEARNING_RATE = 1e-4
    FINE_TUNE_LEARNING_RATE = 1e-5
    DEFAULT_MODEL_NAME = "MobileNetV2"


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE
) -> tf.keras.Model:
    """
    Builds a 4-block custom Convolutional Neural Network from scratch.
    
    Serves as an educational baseline for B.Tech PBL evaluation to demonstrate
    the empirical performance gain achieved by transfer learning over training
    from random initialization on small medical datasets.
    """
    inputs = layers.Input(shape=input_shape, name="input_retina")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.2, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.3, name="drop3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.Dropout(0.35, name="drop4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="custom_gap")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY), name="custom_dense")(x)
    x = layers.Dropout(dropout_rate, name="custom_drop")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="custom_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="RetinaCare_CustomCNN")
    return model


def build_transfer_learning_model(
    model_name: str = DEFAULT_MODEL_NAME,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
    weights: Optional[str] = "imagenet",
    trainable_base: bool = False
) -> tf.keras.Model:
    """
    Constructs a transfer learning model with an ImageNet-pretrained backbone
    and a custom dense classification head.
    
    Args:
        model_name: One of ["MobileNetV2", "EfficientNetB0", "ResNet50", "VGG16", "CustomCNN"].
        input_shape: Dimensions (224, 224, 3).
        num_classes: Number of output severity classes (5).
        dropout_rate: Dropout fraction for regularization.
        weights: "imagenet" or None.
        trainable_base: False to freeze base weights during Stage 1 feature extraction.
        
    Returns:
        tf.keras.Model: Compiled or uncompiled Keras model.
    """
    name_upper = model_name.strip().upper()

    if name_upper == "CUSTOMCNN":
        return build_custom_cnn(input_shape=input_shape, num_classes=num_classes, dropout_rate=dropout_rate)

    inputs = layers.Input(shape=input_shape, name="input_retina")

    # Select pretrained backbone
    if name_upper == "MOBILENETV2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    elif name_upper == "EFFICIENTNETB0":
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    elif name_upper == "RESNET50":
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    elif name_upper == "VGG16":
        base_model = tf.keras.applications.VGG16(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    else:
        raise ValueError(
            f"Unsupported architecture '{model_name}'. Choose from: "
            "MobileNetV2, EfficientNetB0, ResNet50, VGG16, CustomCNN"
        )

    # Freeze base model for initial training
    base_model.trainable = trainable_base

    # Forward through backbone
    x = base_model(inputs, training=False)

    # Custom Classification Head
    # Global Average Pooling replaces parameter-heavy Flatten layers and prevents overfitting
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name="dense_feature_extractor"
    )(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name="dense_sub_head"
    )(x)
    x = layers.Dropout(dropout_rate / 2.0, name="sub_head_dropout")(x)
    
    # 5-class softmax output
    outputs = layers.Dense(num_classes, activation="softmax", name="dr_severity_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"RetinaCare_{model_name}")
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = INITIAL_LEARNING_RATE
) -> tf.keras.Model:
    """
    Compiles the model with Adam optimizer, Categorical Crossentropy, and key metrics.
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",  # Accepts integer labels 0..4 directly
        metrics=["accuracy"]
    )
    return model


def unfreeze_top_layers(
    model: tf.keras.Model,
    num_layers_to_unfreeze: int = 30,
    fine_tune_lr: float = FINE_TUNE_LEARNING_RATE
) -> tf.keras.Model:
    """
    Unfreezes the top N layers of the backbone for Stage 2 Fine-Tuning.
    
    Recompiles with a reduced learning rate so gradients do not corrupt
    the pre-trained ImageNet representations in the lower layers.
    """
    # Locate backbone model layer
    base_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_layer = layer
            break

    if base_layer is None:
        # Fallback if layers are flattened
        model.trainable = True
    else:
        base_layer.trainable = True
        # Freeze all layers before the unfreeze threshold
        for layer in base_layer.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False

    # Re-compile with lower learning rate for fine tuning
    return compile_model(model, learning_rate=fine_tune_lr)


def get_model_summary(model: tf.keras.Model) -> str:
    """Returns a string representation of model.summary()."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
