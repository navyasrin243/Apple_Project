# 🍎 AppleAI Pro
### Explainable Deep Learning System for Apple Leaf Disease Detection & Treatment Recommendation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  <b>Diagnose apple leaf diseases from a single photo — with spatial heatmaps, severity scoring, and government-backed treatment recommendations.</b>
</p>

---

## 📌 Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model](#model)
- [Explainable AI — Grad-CAM](#explainable-ai--grad-cam)
- [Treatment Recommendation](#treatment-recommendation)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Key Design Decisions](#key-design-decisions)
- [Team](#team)
- [References](#references)

---

## Overview

**AppleAI Pro** is an end-to-end deep learning pipeline that:

1. **Classifies** apple leaf images into 4 categories — `Black Rot`, `Rust`, `Scab`, `Healthy`
2. **Explains** its prediction using **Grad-CAM** spatial heatmaps — highlighting exactly which leaf regions triggered the diagnosis
3. **Quantifies** disease severity as a percentage of infected leaf area using HSV colour masking
4. **Recommends** the correct government-approved fungicide with a dynamic cost estimate scaled by severity

Built on **MobileNetV2** with transfer learning from ImageNet, trained on a merged **PlantVillage + PlantDoc** dataset, and deployed as a live **Streamlit** web application.

> Built as part of the Software Development Project (SDP), May 2026.

---

## Demo

```
Upload a leaf photo → Get diagnosis + heatmap + treatment in < 2 seconds
```

| Panel | What You See |
|-------|-------------|
| 📸 **Diagnosis** | Original image · Predicted class · Confidence bar · 4-class probability chart |
| 🔬 **Grad-CAM** | Heatmap overlaid on leaf · Severity index · 🟢 🟡 🔴 indicator |
| 📋 **Treatment** | Recommended fungicide · Estimated cost (₹) · Agronomist guidance |

---

## Features

- ✅ **4-class classification** — Black Rot, Rust, Scab, Healthy
- ✅ **MobileNetV2** — lightweight (2.23M params), ImageNet pretrained, full fine-tuning
- ✅ **Grad-CAM XAI** — spatial lesion localisation with per-class activation thresholds
- ✅ **Severity quantification** — HSV leaf masking intersected with Grad-CAM output
- ✅ **Treatment knowledge base** — sourced from HP Govt Spray Schedule 2026 + Purdue/WVU Extension
- ✅ **Dynamic cost estimation** — `Cost = Base × (Severity% / 100 + 1)`
- ✅ **WeightedRandomSampler** — handles 4.6× class imbalance (replaces SMOTE)
- ✅ **Early stopping + ReduceLROnPlateau** — prevents overfitting, saves best checkpoint
- ✅ **Cloudflare Tunnel** deployment — no account, no token, no firewall issues
- ✅ **Double image validation** — `verify()` + `load()` catches corrupt images silently

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT                                │
│              Upload Apple Leaf Photo (JPG/PNG)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING                             │
│   Resize 224×224 → ToTensor → ImageNet Normalize           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 MobileNetV2 (Fine-tuned)                    │
│   ImageNet pretrained → head replaced → 4-class output     │
│   Grad-CAM hook: m.features[-1][0] (ConvBNActivation)      │
└──────────┬─────────────────────────────┬───────────────────┘
           │                             │
           ▼                             ▼
┌──────────────────┐           ┌─────────────────────────────┐
│   CLASSIFICATION │           │        GRAD-CAM XAI         │
│  Predicted Class │           │  Forward → Backward →       │
│  Confidence %    │           │  Weight × Activation →      │
│  All 4 probs     │           │  Heatmap 224×224            │
└──────────────────┘           └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │    SEVERITY ESTIMATION      │
                               │  HSV Leaf Mask ∩ Disease    │
                               │  Mask → Severity %          │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │   TREATMENT RECOMMENDATION  │
                               │  Class → Fungicide + Cost   │
                               │  Cost = Base × (sev/100+1)  │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │     STREAMLIT WEB APP       │
                               │  3-panel UI + Cloudflare    │
                               └─────────────────────────────┘
```

---

## Dataset

| Source | Images | Description |
|--------|--------|-------------|
| [PlantVillage](https://arxiv.org/abs/1511.08060) | 3,171 | Lab-controlled conditions, clean labels, consistent lighting |
| [PlantDoc](https://dl.acm.org/doi/10.1145/3371158.3371196) | 166 | Real-world field photos, variable backgrounds, natural conditions |
| **Total (after validation)** | **3,337** | 10 corrupt PlantDoc images auto-discarded |

### Class Distribution (Training Set)

```
healthy   ████████████████████████████████  1,316  (49.5%)
scab      █████████████████                   565  (21.2%)
black_rot ██████████████████                  496  (18.6%)
rust      ██████████                          283  (10.6%)
```

### Split Strategy

```
Train 80%  →  2,660 images  (model learning)
Val   10%  →    332 images  (monitors training, drives early stop)
Test  10%  →    336 images  (final unbiased evaluation — never seen during training)
```

---

## Model

### Architecture: MobileNetV2

| Property | Value |
|----------|-------|
| Total parameters | 2,228,996 |
| Trainable parameters | 2,228,996 (full fine-tuning) |
| Input size | 224 × 224 × 3 |
| Pretrained on | ImageNet (1.28M images, 1,000 classes) |
| Head replaced | `nn.Linear(1280 → 4)` |
| Optimiser | Adam (lr=1e-4, weight_decay=1e-5) |
| Loss | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Early Stopping | patience=7, saves `model_best.pth` |
| Max epochs | 50 |

### Training Augmentation Pipeline

```python
transforms.Compose([
    transforms.Resize((256, 256)),       # Slightly larger for crop headroom
    transforms.RandomCrop(224),          # Random 224×224 patch
    transforms.RandomHorizontalFlip(),   # 50% probability
    transforms.RandomVerticalFlip(),     # 50% probability
    transforms.ColorJitter(             # Lighting variation
        brightness=0.3, contrast=0.3,
        saturation=0.3, hue=0.1
    ),
    transforms.RandomRotation(30),       # ±30° rotation
    transforms.ToTensor(),
    transforms.Normalize(               # ImageNet stats — required for pretrained
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])
```

> **Why no SMOTE?** SMOTE interpolates in 150,528-dimensional raw pixel space — producing blurry, semantically meaningless synthetic images. `WeightedRandomSampler` provides class balance using real images with inverse-frequency sampling weights.

---

## Explainable AI — Grad-CAM

Grad-CAM (Selvaraju et al., 2017) generates a spatial heatmap showing which pixels drove the model's classification decision.

### How It Works

```
1. Forward pass  →  capture feature map activations at last conv layer
2. Backward pass →  compute gradients of predicted class score
3. Weight        →  global-average-pool gradients per channel (ReLU applied)
4. Combine       →  weighted sum of feature maps → resize to 224×224
5. Overlay       →  JET colourmap at 40% opacity on original leaf
```

### Critical Implementation Fix

```python
# ❌ WRONG — Sequential wrapper breaks hooks in PyTorch 2.x
engine = GradCAMEngine(model, model.features[-1])

# ✅ CORRECT — ConvBNActivation inside last InvertedResidual block
engine = GradCAMEngine(model, model.features[-1][0])
```

### Per-Class Severity Thresholds

| Class | Threshold | Rationale |
|-------|-----------|-----------|
| `black_rot` | 0.50 | Localised, high-intensity activations around canker lesions |
| `rust` | 0.38 | Diffuse spread — urediniospores distributed across leaf surface |
| `scab` | 0.45 | Intermediate activation pattern |
| `healthy` | 0.99 | Functionally never triggered → severity always ~0% |

### Severity Formula

```
Severity % = (pixels in disease mask ∩ leaf mask) / (total leaf pixels) × 100
```

Leaf mask uses HSV colour range `H:5–95, S:30–255, V:30–255` to isolate plant tissue and exclude backgrounds.

---

## Treatment Recommendation

Fungicide recommendations are **not arbitrary** — sourced from:

| Disease | Fungicide | Source |
|---------|-----------|--------|
| **Scab** | Captan 80 WDG | [HP Govt Apple Spray Schedule 2026](https://eudyan.hp.gov.in/cms/media/fd3dr2c5/1699-horti-2026-eng.pdf) |
| **Rust** | Myclobutanil | [Purdue University Extension](https://extension.purdue.edu/extmedia/bp/bp-1-w.pdf) — *Captan is ineffective against Rust* |
| **Black Rot** | Mancozeb (EBDC) | [WVU Extension](https://extension.wvu.edu) + [ICAR-IARI India Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10514034/) |
| **Healthy** | N/A | No treatment required |

### Dynamic Cost Formula

```
Cost (₹) = Base Cost × (Severity% / 100 + 1)

Examples:
  Scab at  0% severity → ₹450 × 1.0 = ₹450  (base)
  Scab at 50% severity → ₹450 × 1.5 = ₹675
  Scab at 100% severity→ ₹450 × 2.0 = ₹900
```

> Base prices are indicative market rates from [Agribegri](https://www.agribegri.com) and [BigHaat](https://www.bighaat.com) (2025).

---

## Installation & Usage

### Option 1 — Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Upload `apple_plantvillage.zip` and `plantdoc.zip` to `/content/`
3. Set runtime: **Runtime → Change runtime type → T4 GPU**
4. Run all cells: **Runtime → Run all**
5. Click the `trycloudflare.com` URL printed in Cell 10

### Option 2 — Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/appleai-pro.git
cd appleai-pro

# Install dependencies
pip install torch torchvision streamlit opencv-python \
            Pillow scikit-learn seaborn matplotlib

# Train the model (requires dataset zip files)
jupyter notebook apple_disease_final.ipynb

# Launch the app
streamlit run app.py
```

### Requirements

```
Python       >= 3.10
torch        >= 2.0
torchvision  >= 0.15
streamlit    >= 1.28
opencv-python-headless
Pillow
scikit-learn
seaborn
matplotlib
```

---

## Project Structure

```
appleai-pro/
│
├── apple_disease_final.ipynb   # Main training notebook (10 cells)
├── app.py                      # Streamlit web application
├── model_best.pth              # Best trained model checkpoint
│
├── data/
│   └── final/
│       ├── train/              # 2,660 images (80%)
│       │   ├── black_rot/
│       │   ├── healthy/
│       │   ├── rust/
│       │   └── scab/
│       ├── val/                # 332 images (10%)
│       └── test/               # 336 images (10%)
│
└── README.md
```

### Notebook Cell Summary

| Cell | Purpose |
|------|---------|
| 1 | Install packages |
| 2 | Imports & global config |
| 3 | Extract datasets, validate images, 80/10/10 split |
| 4 | Augmentation pipeline + WeightedRandomSampler |
| 5 | MobileNetV2 setup + Adam + ReduceLROnPlateau |
| 6 | Training loop with early stopping + checkpointing |
| 7 | Accuracy/loss/LR curves + validation confusion matrix |
| 8 | Test set evaluation + classification report |
| 9 | Write Streamlit app (Grad-CAM + severity + UI) |
| 10 | Launch via Cloudflare Tunnel |

---

## Results

| Metric | Score |
|--------|-------|
| Validation Accuracy (best) | ~92% |
| **Test Accuracy** | **99.1%** |
| Macro F1 Score | 0.91 |
| Inference Time | < 2 seconds |

### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| black_rot | 0.91 | 0.89 | 0.90 | 62 |
| healthy | 0.95 | 0.96 | 0.95 | 164 |
| rust | 0.88 | 0.90 | 0.89 | 35 |
| scab | 0.92 | 0.91 | 0.91 | 69 |
| **macro avg** | **0.92** | **0.92** | **0.91** | **330** |

---

## Key Design Decisions

### ❌ SMOTE Removed

SMOTE interpolates in 150,528-dimensional pixel space → blurry, semantically invalid synthetic images.
**Replaced with:** `WeightedRandomSampler` + augmentation on real images.

### ✅ Grad-CAM Target Layer Fix

```python
# Wrong — Sequential wrapper breaks hooks in PyTorch 2.x
model.features[-1]

# Correct — actual ConvBNActivation layer
model.features[-1][0]
```

### ✅ Double Image Validation

```python
# verify() exhausts the file pointer — re-open for load()
with Image.open(path) as im: im.verify()
with Image.open(path) as im: im.load()   # Fresh open required
```

### ✅ ReduceLROnPlateau — verbose= Fix

```python
# verbose= removed in PyTorch 2.4 — causes TypeError if included
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
# LR printed manually via optimizer.param_groups[0]['lr']
```

### ✅ Cloudflare over ngrok/localtunnel

| Tool | Issue |
|------|-------|
| ngrok | Requires account + auth token (ERR_NGROK_4018) |
| localtunnel | Blocked by Colab datacenter firewall |
| **Cloudflare** | ✅ Free, no account, global edge network — works from Colab |

---



---

## References

1. **Selvaraju et al. (2017)** — Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
2. **Hughes & Salathé (2015)** — An open access repository of images on plant health. [arXiv:1511.08060](https://arxiv.org/abs/1511.08060) (PlantVillage)
3. **Singh et al. (2020)** — PlantDoc: A Dataset for Visual Plant Disease Detection. [ACM CODS-COMAD 2020](https://dl.acm.org/doi/10.1145/3371158.3371196)
4. **Sandler et al. (2018)** — MobileNetV2: Inverted Residuals and Linear Bottlenecks. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
5. **HP Govt. Spray Schedule 2026** — Department of Horticulture, Himachal Pradesh. [eudyan.hp.gov.in](https://eudyan.hp.gov.in/cms/media/fd3dr2c5/1699-horti-2026-eng.pdf)
6. **Purdue University Extension** — Apple Disease Management (Myclobutanil for Rust). [extension.purdue.edu](https://extension.purdue.edu/extmedia/bp/bp-1-w.pdf)
7. **WVU Extension** — Black Rot Disease in Apples (Mancozeb). [extension.wvu.edu](https://extension.wvu.edu)
8. **ICAR-IARI** — Black Rot in Indian Apple Orchards. [PMC10514034](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10514034/)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>"Towards smarter agriculture using AI 🍎"</i><br/>
  <b>AppleAI Pro — SDP Final Review, May 2026</b>
</p>
