# NeuroScan AI: Alzheimer's stage detection from brain MRI

A Flask web app that classifies a brain MRI slice into one of four stages of cognitive impairment using a ResNet-50 deep learning model, and keeps a scan history for each patient.

> **Research prototype.** Results are not a medical diagnosis. Any clinical decision needs a qualified clinician.

## Research background

This app accompanies our paper *Quantifying Neural Degeneration: An Information-Theoretic Approach to Alzheimer's Disease* (the **NeuroThermo** framework, VIT Bhopal University). The paper was selected for the International Congress on Khazar Scientific Research and Innovation.

The paper combines a ResNet-50 classifier with information-theory measures (Shannon entropy and mutual information over feature maps) as interpretable markers of neurodegeneration. This repository contains the **deployed classifier** and the training script. Entropy and mutual-information helpers are in `utils/train_model.py`; the full entropy-regularised training and biomarker analysis from the paper are not part of the web app.

## What it does

| Who | Can do |
|---|---|
| Patient | Register, log in, upload a JPG/PNG MRI slice, see the predicted stage with confidence, view past scans |
| Admin | See totals and recent scans, add or delete patients, print a report of the result distribution |

Predicted classes (the dataset's folder names):

| Index | Stage | Roughly corresponds to |
|---|---|---|
| 0 | Mild Impairment | Mild dementia (CDR 1) |
| 1 | Moderate Impairment | Moderate dementia (CDR 2) |
| 2 | No Impairment | Cognitively normal (CDR 0) |
| 3 | Very Mild Impairment | Very mild dementia (CDR 0.5) |

The order matters: Keras numbers classes alphabetically by folder name, and `CLASS_NAMES` in `app.py` must follow it.

## Model

- **Architecture:** ResNet-50 (ImageNet weights, frozen) → global average pooling → batch normalisation → dense 2048 → 512 → 256 → 64 → softmax 4.
- **Input:** 128 × 128 RGB, pixel values scaled to 0–1.
- **File:** `models/resnet50_alzheimer.keras` (about 160 MB, stored with Git LFS).
- **Measured on the dataset's test folder (1,279 images):**

| Class | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| Mild Impairment | 0.870 | 0.899 | 0.885 | 179 |
| Moderate Impairment | 0.714 | 0.833 | 0.769 | 12 |
| No Impairment | 0.892 | 0.931 | 0.911 | 640 |
| Very Mild Impairment | 0.867 | 0.797 | 0.830 | 448 |
| **Overall accuracy** | | | **0.879** | 1,279 |

Most errors are between neighbouring stages (Very Mild ↔ No Impairment). The Moderate class has only 12 test images, so its numbers are noisy.

## Run it

Requires Python 3.13 (other recent versions may work) and Git LFS for the model file.

```bash
git lfs install
git clone https://github.com/M0tracno/Alzheimer.git
cd Alzheimer
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The database (`database.db`) and a default admin account are created on first start.

### Configuration (`.env`)

| Variable | Purpose | Default |
|---|---|---|
| `SECRET_KEY` | Signs login sessions; set it so logins survive restarts | random per start |
| `ADMIN_USERNAME`, `ADMIN_EMAIL`, `ADMIN_PASSWORD` | First admin account | `admin`, `admin@neuroscan.ai`, `admin123` |
| `FLASK_DEBUG` | `1` enables debug mode | off |

Change the admin password before putting the app anywhere public.

## Test

```bash
python -m unittest discover tests
```

The tests cover page rendering, upload validation, safe file names, admin-only pages, POST-only deletion and the class-label order. The label-order test runs the real model on test images and is skipped when the model or dataset is missing.

## Train

Put the dataset at `dataset/Combined Dataset/{train,test}/<class>/` and run:

```bash
python utils/train_model.py
```

This trains the architecture above (80/20 train/validation split from `train/`), saves the best checkpoint to `models/resnet50_alzheimer.keras`, plots `training_history.png`, and prints a confusion matrix and classification report on `test/`.

## Project structure

```
app.py              Flask routes, model loading, prediction
config.py           Settings (paths, upload limits, admin defaults)
templates/          HTML pages (patient and admin)
static/             CSS, JS, uploaded scans
models/             Trained model (Git LFS)
utils/train_model.py  Training, evaluation, entropy / mutual-information helpers
tests/test_app.py   Automated tests
```

## Security notes

- Passwords are hashed with Werkzeug (salted PBKDF2/scrypt).
- Uploads are limited to JPG/PNG under 5 MB and renamed with `secure_filename`.
- Patient deletion requires an admin and a POST request.
- SQL uses parameterised queries.
- Not yet included: CSRF tokens, rate limiting, audit logging, encryption at rest. Add these before handling real patient data.
