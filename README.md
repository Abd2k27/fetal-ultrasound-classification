# 🎯 Fetal Ultrasound Plane Classification

## Overview
This project implements a professional-grade image classifier for fetal ultrasound planes using **Transfer Learning** and **Deep Learning**. Inspired by industry leaders in prenatal AI (like Sonio), this repository demonstrates a complete machine learning lifecycle—from robust data pipelining to advanced error analysis and model explainability.

The goal is to automatically classify ultrasound images into 6 key anatomical planes, which is a critical step in automating fetal screening and ensuring high-quality diagnostic data.

## 🏥 Medical Context
In prenatal care, clinicians must capture specific anatomical planes (e.g., Trans-thalamic Brain, Abdominal circumference) to measure fetal growth and detect anomalies. Manually identifying these planes is time-consuming and prone to operator variability. This project builds a tool to:
1.  **Standardize** plane identification.
2.  **Accelerate** the screening workflow.
3.  **Provide Explainability** to clinicians via feature attribution (Grad-CAM).

## 🛠️ Technical Stack
- **Framework:** PyTorch
- **Model Library:** `timm` (PyTorch Image Models) — Used for state-of-the-art EfficientNet backbones.
- **Explainability:** `pytorch-grad-cam` — Critical for medical trust.
- **Metrics:** `scikit-learn` (Confusion Matrices, F1-Score).
- **Visualization:** `Seaborn` & `Matplotlib`.

## 📊 Dataset: Fetal Planes DB
The project uses the **Fetal Planes DB** (Burgos-Artizzu et al.), containing 12,400+ images from 1,792 patients.
- **Classes:** Abdomen, Brain, Femur, Thorax, Cervix, Other.
- **Pre-defined Split:** Rigorous Train/Test splits as defined by the original authors to ensure valid benchmarking.

## 🚀 Key Features
- **Professional Data Pipeline:** Custom PyTorch Dataset with clinical-grade augmentations.
- **Modern Architecture:** Fine-tuned **EfficientNet-B0** via `timm` for optimal accuracy/speed trade-off.
- **Rigorous Error Analysis:**
  - Automated Confusion Matrix generation.
  - "Worst Prediction" identification (high-confidence errors) to find clinical edge cases.
- **Grad-CAM Explainability:** Visual overlays showing *where* the model is focusing (e.g., the cerebellum in a brain plane).

## 📂 Project Structure
```text
fetal-ultrasound-classification/
├── src/
│   ├── dataset.py      # Data loading & Augmentation
│   ├── model.py        # timm-based EfficientNet setup
│   ├── train.py        # Robust training loop
│   ├── evaluation.py   # Error analysis & Grad-CAM
│   └── utils.py        # Device & Helper functions
├── tests/              # Pytest suite
├── notebooks/          # Interactive Error Analysis
├── data/               # (Dataset placeholder)
└── requirements.txt    # Professional dependency list
```

## 🚦 Getting Started
1.  **Install dependencies:** `pip install -r requirements.txt`
2.  **Download Dataset:** Get the [Fetal Planes DB from Zenodo](https://zenodo.org/records/3904280) and place the CSV and Images in the `data/` directory.
3.  **Run Tests:** `pytest tests/`
4.  **Train:** (Configure paths in `src/train.py`) `python -m src.train`

## 📈 Error Analysis Highlights
(Once you run the model, you can include screenshots here)
- **Confusion Matrix:** Identifies if 'Abdomen' and 'Thorax' are frequently confused due to similar visual features.
- **Grad-CAM:** Confirms the model detects the skull boundary for brain planes and the femur bone for femur planes.

---
*Created for portfolio purposes to demonstrate expertise in Medical Computer Vision.*
