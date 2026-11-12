# Deep Learning Model Checkpoints

This directory houses trained Keras model checkpoints for Diabetic Retinopathy detection.

---

## 1. Primary Model File

* **`models/best_model.keras`**:  
  The top-performing model checkpoint saved automatically during training by the `tf.keras.callbacks.ModelCheckpoint` callback monitoring minimum validation loss (`val_loss`).

---

## 2. Generating the Model Locally or on Google Colab

To train the model and generate `best_model.keras`:

```bash
# 1. Train MobileNetV2 (Recommended for standard laptops and Colab)
python src/train.py --model MobileNetV2 --epochs 15 --fine-tune-epochs 10

# 2. Or train EfficientNetB0
python src/train.py --model EfficientNetB0 --epochs 15 --fine-tune-epochs 10
```

The script will:
1. Initialize the backbone with ImageNet pre-trained weights.
2. Freeze the base layers and train the classification head (Phase 1: Feature Extraction).
3. Unfreeze the top 30 layers and train with a lower learning rate (Phase 2: Fine-Tuning).
4. Save the optimal checkpoint automatically to `models/best_model.keras`.

---

## 3. Training on Google Colab

If you do not have a local NVIDIA GPU:
1. Open the included notebook: `notebooks/diabetic_retinopathy_training.ipynb` in Google Colab.
2. Set Runtime Type to **GPU (T4)**.
3. Run the notebook to train on the APTOS 2019 dataset.
4. Download the resulting `best_model.keras` and place it here in this `models/` directory.

---

## 4. Academic Integrity Notice

In compliance with B.Tech CSE PBL requirements:
* Model weights must be trained on authentic retinal datasets or generated via the verified pipeline.
* If no model file is found, the Streamlit UI gracefully alerts the user with training instructions rather than outputting fabricated or hardcoded results.
