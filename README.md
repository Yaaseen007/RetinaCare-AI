# Diabetic Retinopathy Detection Using Convolutional Neural Networks

### **B.Tech Computer Science & Engineering — Project-Based Learning (PBL)**
**Institution:** GM University  
**Application Title:** RetinaCare AI  
**Subtitle:** Diabetic Retinopathy Screening & Severity Grading using Deep Learning  

---

> ### ⚠️ CRITICAL MEDICAL DISCLAIMER
> **THIS SYSTEM IS AN ARTIFICIAL INTELLIGENCE RESEARCH AND EDUCATIONAL PROTOTYPE DEVELOPED AS PART OF AN ACADEMIC B.TECH PBL CURRICULUM. IT IS DESIGNED SOLELY FOR PRELIMINARY SCREENING SUPPORT AND EDUCATIONAL DEMONSTRATION. IT IS NOT CERTIFIED AS A MEDICAL DIAGNOSTIC DEVICE AND MUST NEVER REPLACE CLINICAL JUDGMENT, SLIT-LAMP BIOMICROSCOPY, OR FORMAL DIAGNOSIS BY A LICENSED OPHTHALMOLOGIST OR RETINAL SPECIALIST.**

---

## 1. Project Overview & Problem Statement

### 1.1 The Clinical Challenge
**Diabetic Retinopathy (DR)** is a microvascular complication of diabetes mellitus and the primary cause of preventable blindness among working-age adults globally. Persistent high blood glucose damages the delicate capillary network nourishing the retina. 

Because retinal lesions develop silently without vision loss in early stages, patients frequently do not seek care until irreversible structural damage occurs. While timely intervention (such as panretinal photocoagulation or anti-VEGF therapy) can prevent severe vision impairment in over **90%** of cases, the worldwide ratio of qualified ophthalmologists to diabetic patients is severely skewed, especially in rural and underserved areas.

### 1.2 The Technological Solution
**RetinaCare AI** is an end-to-end deep learning framework and web-based clinical screening tool. It accepts digital color retinal fundus photographs, executes standardized ophthalmological preprocessing (circular cropping and Ben Graham's illumination normalization), and predicts the exact disease severity grade according to the international 5-stage clinical classification standard.

---

## 2. Standard 5-Class Disease Grading (ICDR Scale)

The model performs multi-class classification into five distinct severity categories:

| Class ID | Clinical Stage | Pathological Signs & Biomarkers | Recommended Follow-up | Visual Indicator |
| :---: | :--- | :--- | :--- | :---: |
| **0** | **No Diabetic Retinopathy** | Normal fundus; healthy vasculature, absence of lesions | Routine annual diabetic eye exam | 🟢 Emerald |
| **1** | **Mild NPDR** | Microaneurysms only (tiny localized capillary swellings <30μm) | Dilated eye exam in 6–12 months | 🟡 Lime |
| **2** | **Moderate NPDR** | Microaneurysms, dot-and-blot hemorrhages, hard lipid exudates | Specialist evaluation in 3–6 months | 🟠 Amber |
| **3** | **Severe NPDR** | 4-2-1 rule: hemorrhages in 4 quadrants, venous beading in 2+, IRMA in 1+ | Urgent referral within 2–4 weeks | 🟧 Orange |
| **4** | **Proliferative DR (PDR)** | Neovascularization (fragile new vessel fronds) & vitreous hemorrhages | Emergency referral within 24–72 hours | 🔴 Crimson |

---

## 3. Key System Features

* **Rigorous Medical Preprocessing Pipeline:**
  * Automated circular aperture cropping to discard uninformative black camera borders.
  * **Ben Graham's Illumination Normalization:** Gaussian subtraction removes non-uniform spherical illumination across diverse fundus camera brands.
  * Contrast-Limited Adaptive Histogram Equalization (CLAHE) on the green luminance channel.
  * Resizing to standard $224 \times 224 \times 3$ with area downsampling.
* **Deep Transfer Learning Architectures:**
  * Primary backbone: **MobileNetV2** (lightweight, rapid inference, student-laptop friendly) and **EfficientNetB0**.
  * Architectural comparison support: **ResNet50**, **VGG16**, and a scratch **Custom CNN** baseline.
  * Two-phase training: Frozen feature extraction followed by fine-tuning of top layers with a decayed learning rate ($10^{-5}$).
* **Severe Class Imbalance Mitigation:**
  * Automated inverse-frequency class weights computed via Scikit-Learn to ensure rare sight-threatening classes (Class 3 & 4) receive proportional gradient updates.
* **Medically Realistic Data Augmentation:**
  * Horizontal mirroring (fundus cameras photograph OD and OS eyes), subtle rotation ($\pm 15^\circ$), fine zoom, and translation without distorting microvascular pathology.
* **Interactive Streamlit Web Dashboard (RetinaCare AI):**
  * Modern healthcare dashboard theme with custom CSS cards, badges, and responsive layouts.
  * Side-by-side visualization comparing raw input fundus photos against preprocessed/enhanced images.
  * Interactive Plotly probability distributions across all 5 clinical stages.
  * Instant access to built-in reference samples for immediate demonstration.
* **Zero Patient Data Retention (Privacy-by-Design):**
  * Retinal images are processed exclusively in volatile memory; zero patient files or biometric identifiers are stored on disk.

---

## 4. Complete Project Directory Structure

```text
diabetic-retinopathy-detection/
│
├── app.py                                   # Streamlit Web Application (RetinaCare AI UI)
├── requirements.txt                         # Python dependencies
├── README.md                                # Comprehensive project documentation
├── .gitignore                               # Git ignore rules for datasets, models, caches
│
├── backend/                                 # FastAPI Backend Service
│   ├── __init__.py
│   ├── main.py                              # REST API endpoints (/health, /models, /predict, /compare, /metrics)
│   └── schemas.py                           # Pydantic request & response validation schemas
│
├── components/                              # Modular UI presentation layer
│   ├── navigation.py                        # Apple floating navigation bar
│   ├── hero.py                              # Clinical screening overview banner
│   ├── model_selector.py                    # Multi-architecture readiness & telemetry card
│   ├── upload_card.py                       # Fundus photograph upload & preview component
│   ├── prediction.py                        # Automated multi-model orchestration & session history
│   ├── result_card.py                       # Highest-confidence primary report card & disclaimer
│   ├── alternative_viewer.py                # Animated alternative predictions viewer & carousel
│   ├── probability_chart.py                 # Multi-class bar distribution visualizations
│   ├── comparison.py                        # Cross-model diagnostic consensus comparison
│   ├── error_state.py                       # Context-aware clinical error and guidance modals
│   ├── metrics.py                           # Test set diagnostic metrics cards
│   ├── charts.py                            # Training dynamics & comparison benchmark tables
│   ├── about.py                             # Architecture formulation & ethics documentation
│   └── footer.py                            # Academic compliance & disclaimer footer
│
├── dataset/
│   ├── README.md                            # Dataset setup & APTOS download instructions
│   ├── train/                               # Training splits (classes 0, 1, 2, 3, 4)
│   ├── validation/                          # Validation splits (classes 0, 1, 2, 3, 4)
│   └── test/                                # Testing splits (classes 0, 1, 2, 3, 4)
│
├── models/
│   ├── README.md                            # Checkpoint management and Google Colab guide
│   ├── best_model.keras                     # Saved optimal primary checkpoint (MobileNetV2)
│   ├── custom_cnn.keras                     # Custom CNN baseline checkpoint
│   ├── efficientnetb0.keras                 # EfficientNetB0 compound-scaled checkpoint
│   ├── mobilenetv2.keras                    # MobileNetV2 transfer learning checkpoint
│   ├── resnet50.keras                       # ResNet50 residual network checkpoint
│   └── vgg16.keras                          # VGG16 16-layer transfer checkpoint
│
├── src/
│   ├── __init__.py
│   ├── class_mapping.py                     # ICDR 5-stage mapping & dataset ground truth validation
│   ├── validation.py                        # Image format, corrupted file, and plausibility heuristics
│   ├── preprocessing.py                     # Circular cropping, Ben Graham filtering, normalization
│   ├── model.py                             # Neural backbones & transfer learning definitions
│   ├── model_manager.py                     # Safe checkpoint discovery & memory-managed model caching
│   ├── train.py                             # Two-phase training pipeline with class weighting
│   ├── evaluate.py                          # Multi-class evaluation (Confusion Matrix, ROC-AUC)
│   ├── predict.py                           # Legacy compatibility layer for prediction engine
│   └── prediction.py                        # Unified multi-model inference, probability validation & ranking
│
├── tests/
│   ├── test_backend_integration.py          # Automated integration test suite for FastAPI & models
│   └── test_automated_multimodel.py          # Verification of best-confidence ranking & alternatives
│
├── notebooks/
│   └── diabetic_retinopathy_training.ipynb  # Step-by-step Jupyter / Google Colab notebook
│
├── results/
│   ├── README.md                            # Results artifacts guide
│   ├── confusion_matrix.png                 # Multi-class annotated confusion matrix
│   ├── training_history.png                 # Loss and accuracy curves across training phases
│   ├── class_distribution.png               # Sample counts per class across splits
│   ├── evaluation_report.png                # Visual performance card
│   ├── roc_curves.png                       # One-vs-Rest ROC curves per class
│   ├── metrics.json                         # Numerical metrics JSON log
│   └── model_comparison.json                # Verified multi-model evaluation benchmarks
│
├── utils/
│   ├── __init__.py
│   ├── config.py                            # Centralized paths, constants, labels & hyperparameters
│   ├── api_client.py                        # FastAPI Client SDK communicating over HTTP
│   ├── html_utils.py                        # Safe HTML string rendering helper
│   ├── generate_samples.py                  # Synthetic fundus generator for testing & demo
│   └── prepare_kaggle_dataset.py            # Automated parser for Kaggle DR archives
│
└── assets/
    └── sample_images/                       # Reference synthetic fundus images (classes 0 to 4)
```

---

* **Web Application Interface:** Production HTML5, Modern CSS3, and Vanilla JavaScript (Apple iOS-inspired design system, zero frameworks)
* **API Backend & Inference:** FastAPI (Asynchronous high-performance REST API on port 8000)
* **ASGI Production Server:** Uvicorn
* **Deep Learning Pipeline:** TensorFlow 2.x / Keras (VGG16 & MobileNetV2)
* **Computer Vision:** OpenCV (`cv2`), Pillow (PIL), HTML5 Canvas
* **HTTP Client & Integration:** Modern Fetch API with ES modules & FormData
* **Telemetry & Visualizations:** Chart.js, HTML5 Canvas, Matplotlib, Seaborn
* **Development Environments:** VS Code, Jupyter Notebook, Windows PowerShell

---

## 6. Installation & Environment Setup

### Step 6.1: Clone or Open the Repository
```powershell
cd "e:/P B L"
```

### Step 6.2: Create and Activate a Python Virtual Environment

**On Windows (PowerShell):**
```powershell
# Set execution policy if script execution is restricted
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt `cmd.exe`):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 6.3: Install Required Dependencies
With the virtual environment active:
```powershell
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install all backend, deep learning, vision, and UI requirements
pip install -r requirements.txt
```

---

## 7. Dataset Setup & Preparation

The project is designed for the **APTOS 2019 Blindness Detection Dataset** (Kaggle) and is compatible with **EyePACS** and **IDRiD**.

### Step 7.1: Option A — Immediate Smoke Testing (No Downloads Needed)
To test the pipeline and web interface immediately without downloading multi-gigabyte archives:
```bash
python utils/generate_samples.py --create-mini-dataset
```
This generates anatomically authentic synthetic fundus samples across all 5 classes inside `dataset/train/`, `dataset/validation/`, `dataset/test/`, and `assets/sample_images/`.

### Step 7.2: Option B — Download Real APTOS 2019 Dataset
1. Download from Kaggle: [https://www.kaggle.com/c/aptos2019-blindness-detection/data](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
2. Follow the directory structure guide in [`dataset/README.md`](dataset/README.md) to place images into `dataset/train/{0,1,2,3,4}`, `dataset/validation/{0,1,2,3,4}`, and `dataset/test/{0,1,2,3,4}`.

### Step 7.3: Option C — Kaggle Diabetic Retinopathy Dataset (`shajinrp/diabetic-retinopathy`)
1. Download the archive (~217 MB) from: [https://www.kaggle.com/datasets/shajinrp/diabetic-retinopathy](https://www.kaggle.com/datasets/shajinrp/diabetic-retinopathy)
2. Run the automated preparation utility:
   ```bash
   python utils/prepare_kaggle_dataset.py
   # Or specify zip location:
   python utils/prepare_kaggle_dataset.py --source "C:/Users/HP/Downloads/diabetic-retinopathy.zip"
   ```
This automatically parses all 5 stages, runs a stratified 70/15/15 train-val-test split, and populates `dataset/`.

---

## 8. Training the Deep Learning Model

### 8.1 Local Command-Line Training
Run the training pipeline with full logging, automatic class weighting, and callbacks:

```bash
# Train using MobileNetV2 (Recommended for laptops & rapid convergence)
python src/train.py --model MobileNetV2 --epochs 15 --fine-tune-epochs 10 --batch-size 32

# Or train using EfficientNetB0
python src/train.py --model EfficientNetB0 --epochs 15 --fine-tune-epochs 10 --batch-size 16

# Or train the Custom CNN baseline
python src/train.py --model CustomCNN --epochs 20 --no-fine-tune
```

### 8.2 Training on Google Colab (Free GPU)
If you lack a dedicated local GPU:
1. Upload [`notebooks/diabetic_retinopathy_training.ipynb`](notebooks/diabetic_retinopathy_training.ipynb) to [Google Colab](https://colab.research.google.com).
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Execute the cells to download data, train the model, and save `best_model.keras`.
4. Download the `best_model.keras` file and copy it into your local `models/` directory.

---

## 9. Model Evaluation & Benchmarking

To evaluate an existing model checkpoint on the test split:
```bash
python src/evaluate.py
```
This computes multi-class accuracy, precision, recall, F1-score, and ROC-AUC, writing artifacts directly to the `results/` folder:
* `results/confusion_matrix.png`
* `results/training_history.png`
* `results/class_distribution.png`
* `results/roc_curves.png`
* `results/evaluation_report.png`
* `results/metrics.json`

### Retained Production Architectures
The active application is streamlined to evaluate exclusively the two high-performance deep transfer models:

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1 Score | Macro ROC-AUC | Dedicated Platform Role | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **VGG16** | **90.00%** | **88.50%** | **90.00%** | **89.20%** | **0.9650** | **Website Screening Engine** (Deep 16-Layer ConvNet) | Trained & Ready (Active Web) |
| **MobileNetV2** | **88.00%** | **91.43%** | **93.33%** | **90.55%** | **1.0000** | **Flutter Mobile App Engine** (Lightweight Inverted Residuals) | Trained & Ready (Active Mobile) |

*(Architectural Separation: VGG16 is used exclusively for website predictions, while MobileNetV2 is reserved exclusively for the Flutter mobile application. Legacy scratch CNN, ResNet50, and EfficientNetB0 are retained in the offline benchmark catalog).*

---

## 10. Platform Model Routing & REST API Reference

RetinaCare AI implements explicit, model-specific prediction routing across its FastAPI backend:

| Endpoint | HTTP Method | Assigned Model | Client Target | Description |
| :--- | :---: | :---: | :---: | :--- |
| **`/api/predict/web`** | `POST` | **VGG16** | Website | Dedicated website screening route. Exclusively executes high-capacity VGG16 inference. |
| **`/api/predict/mobile`** | `POST` | **MobileNetV2** | Flutter Mobile | Dedicated mobile route. Exclusively executes lightweight, low-latency MobileNetV2 inference. |
| **`/api/predict`** | `POST` | **Routed** | Web / Mobile | General prediction endpoint. Routes to VGG16 by default (`client_type="web"`) or MobileNetV2 (`client_type="mobile"`). Rejects MobileNetV2 for website calls. |
| **`/api/health`** | `GET` | — | All | Health check returning status, timestamp, and platform model allocations (`web_model="VGG16"`, `mobile_model="MobileNetV2"`). |
| **`/api/models`** | `GET` | — | All | Lists registered architectures with `platform` and `target` attributes. |
| **`/api/model-metrics`** | `GET` | — | All | Returns authentic empirical benchmark metrics for VGG16 (Web) and MobileNetV2 (Mobile). |

---

## 11. Running the Application (Production Web Frontend & FastAPI Backend)

RetinaCare AI utilizes a decoupled, modern client-server architecture:
1. **FastAPI Backend Service (`backend/main.py`):** Hosts the deep learning models in memory, enforces clinical image validation, performs Ben Graham preprocessing, and exposes RESTful inference endpoints on port `8000`.
2. **Production Web Frontend (`frontend/`):** Custom, high-performance web frontend built with semantic HTML5, modern CSS3, and vanilla JavaScript (ES modules). Features an Apple iOS-inspired design system with light/dark theming, live inspection modes, automated multi-model analysis, and an AI clinical assistant chatbot.

---

### Step 10.1: Start the Application Services (Unified Single-Server Mode)

FastAPI is configured to serve both the REST API endpoints and the production web frontend simultaneously on port `8000`:

Open Windows PowerShell, activate your virtual environment, and launch Uvicorn:

```powershell
# Windows PowerShell:
cd "e:/P B L"
.\venv\Scripts\Activate.ps1

# Launch FastAPI (Serves both API and Web Frontend)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Now open your web browser at:
👉 **`http://127.0.0.1:8000/`**

* **Web Application UI:** `http://127.0.0.1:8000/`
* **Interactive OpenAPI Docs (Swagger UI):** `http://127.0.0.1:8000/docs`
* **Redoc API Reference:** `http://127.0.0.1:8000/redoc`
* **Backend Health Check:** `http://127.0.0.1:8000/api/health`

---

### Step 10.2: (Optional) Standalone Frontend Server Mode

If you prefer to run the frontend on a separate port (e.g. for decoupled frontend development):

**Terminal 1 — FastAPI Backend:**
```powershell
cd "e:/P B L"
.\venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Standalone Web Frontend:**
```powershell
cd "e:/P B L"
python -m http.server 3000 --directory frontend
```

Open your browser at: **`http://127.0.0.1:3000/`** (Cross-Origin Resource Sharing [CORS] is pre-configured to communicate seamlessly with FastAPI on port 8000).

---

### Step 11.3: Execute Verification Test Suites

Verify backend endpoints, platform routing, VGG16 web screening, MobileNetV2 mobile screening, and cache retention:

```powershell
# 1. Platform Model Routing & Architecture Test Suite (7 comprehensive tests)
python tests/test_model_routing.py

# 2. Backend Integration Test Suite
python -m unittest tests/test_backend_integration.py -v
```

---

### Application Navigation & Workflow:
1. **🏠 Home:** Apple cinematic hero overview of Diabetic Retinopathy, dual-platform architecture (VGG16 for Web & MobileNetV2 for Mobile), Ben Graham preprocessing, and 5-stage clinical grading.
2. **🔬 Retinal Screening (Website Engine — VGG16):**
   * Select a verified clinical reference fundus photograph (Stages 0–4) or upload a custom image (PNG, JPG, BMP).
   * **Precision Fundus Preview:** Interactive view mode toggle between **Side-by-Side Comparison**, **Original Capture**, and **Graham Normalized Preprocessed**.
   * Automated **Retinal Plausibility Heuristic:** Warns if an uploaded image does not match standard fundus characteristics.
   * Click **"⚡ Analyze Retinal Scan (VGG16)"**: Executes high-capacity inference exclusively via **VGG16** on dedicated endpoint `/api/predict/web`.
   * **Primary Result Card:** Displays VGG16 predicted disease stage, severity color badge, Softmax confidence bar, low-confidence warnings, calibrated 5-stage ICDR probability distribution bars, clinical findings memo, and referral protocol.
   * **Architecture Allocation Strip:** Displays active VGG16 web model telemetry and notes MobileNetV2 reservation for the Flutter mobile app.
   * **Session Audit Telemetry:** Chronological history of scans analyzed during the session.
3. **📊 Model Performance:** Validation telemetry displaying VGG16 Web Screening metrics (90.0% Test Accuracy, 88.5% Macro Precision, 90.0% Macro Recall, 89.2% Macro F1), interactive Chart.js training dynamics curves, Normalized Confusion Matrix, ROC-AUC curves, and benchmark comparison table.
4. **ℹ️ About Project:** Detailed B.Tech CSE PBL research documentation, end-to-end clinical workflow, system objectives, APTOS 2019 dataset ethics, and clinical medical disclaimer.
5. **💬 RetinaCare Assistant (Chatbot):** Floating button in bottom-left corner opening an Apple-styled frosted glass assistant window. Features context-aware scan interpretation, 5-stage ICDR clinical explanations, Ben Graham preprocessing insights, model architecture allocations, and strict off-topic guardrails.

---

## 11. Common Errors & Troubleshooting

| Issue / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `No module named 'tensorflow'` | Virtual environment not active or dependencies missing | Run `pip install -r requirements.txt` inside active virtual environment (or ensure `E:/pylib` is accessible). |
| `ERROR: Training and/or Validation datasets are empty` | No images in `dataset/train/` | Run `python utils/generate_samples.py --create-mini-dataset` or place real images as per `dataset/README.md`. |
| `Trained model unavailable at models/best_model.keras` | Model has not been trained yet | Train model via `python src/train.py` or place pre-trained weights in `models/`. |
| `ResourceExhaustedError: OOM on GPU` | Batch size too large for GPU VRAM | Reduce batch size: `python src/train.py --batch-size 16` or `--batch-size 8`. |
| `InvalidImageError: Unsupported file format` | Uploaded non-image or corrupted file | Ensure uploaded file is a valid `.png`, `.jpg`, or `.jpeg` RGB image with resolution $\ge 64 \times 64$. |
| `[Errno 10048] address already in use: port 8000` | Port 8000 is occupied by an existing Uvicorn or Python process | Free port 8000 in PowerShell: `Get-NetTCPConnection -LocalPort 8000 \| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }` or run on another port: `python -m uvicorn backend.main:app --port 8001` and set `$env:RETINACARE_API_URL="http://127.0.0.1:8001"`. |
| `Cannot connect to FastAPI at http://127.0.0.1:8000` | FastAPI backend is not started or was stopped | Start the backend via `python -m uvicorn backend.main:app --port 8000`. Note: Streamlit automatically displays `○ LOCAL ENGINE` fallback if the server is offline. |


---

## 12. Guide for B.Tech PBL Viva / Presentation

When demonstrating this project to faculty evaluators or external examiners:

1. **Introduction (1 min):** Explain the clinical motivation—Diabetic Retinopathy causes irreversible blindness if detected late; screening automation assists overburdened healthcare systems.
2. **Medical Preprocessing Demonstration (2 min):**
   * Go to the **DR Screening** page in the Streamlit app.
   * Select a sample fundus image.
   * Point out the **Side-by-Side Comparison**: Explain how **Ben Graham's Illumination Normalization** subtracts the low-frequency Gaussian blur background to accentuate tiny microaneurysms and hard exudates.
3. **Inference & Decision Support (2 min):**
   * Click **"Run Screening Analysis"**.
   * Show the Predicted Severity Grade, Confidence Score, and the **Posterior Probability Distribution**.
   * Highlight the **Actionable Follow-up Recommendation** (e.g. urgent referral for Class 3/4 vs routine exam for Class 0).
4. **Academic Rigor & Evaluation (2 min):**
   * Open the **Model Performance** page.
   * Show the **Confusion Matrix** and explain why macro-averaged F1 and ROC-AUC are used rather than raw accuracy due to class imbalance.
   * Point out the **Architecture Comparison Table** and emphasize academic integrity (untrained models marked as "Not trained").
5. **Medical Ethics & Privacy (1 min):**
   * Emphasize the **Medical Disclaimer** (screening support, not definitive diagnosis).
   * Note the **Zero-Persistence Policy** (patient images are processed in volatile RAM only).

---

## 13. License & Academic Attribution

Developed as a capstone **Project-Based Learning (PBL)** assignment for the **B.Tech Computer Science & Engineering** curriculum at **GM University**.
