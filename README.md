# Chicken Disease Classification

End-to-end CNN pipeline for classifying chicken fecal images into:
- `Coccidiosis`
- `Healthy`

The project includes:
- Data ingestion
- Transfer learning base model preparation (VGG16)
- Training with callbacks
- Evaluation and metric export
- Flask inference app for image upload

## Project Workflow

1. Configure paths in `config/config.yaml`
2. Configure hyperparameters in `params.yaml`
3. Run training + evaluation pipeline
4. Serve predictions through Flask app

## Setup

Use Python `3.10` or `3.11` (TensorFlow is typically not available for Python `3.13`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Full Pipeline

```bash
python3 main.py
```

Output artifacts:
- `artifacts/training/model.h5`
- `artifacts/scores.json`

## Run Individual Stages (DVC Style)

```bash
PYTHONPATH=src python -m cnnClassifier.pipeline.stage_01_data_ingestion
PYTHONPATH=src python -m cnnClassifier.pipeline.stage_02_prepare_base_model
PYTHONPATH=src python -m cnnClassifier.pipeline.stage_03_training
PYTHONPATH=src python -m cnnClassifier.pipeline.stage_04_evaluation
```

Or run through DVC:

```bash
dvc repro
```

## Run Inference API

```bash
python3 app.py
```

Open `http://localhost:8080`, upload an image, and get prediction + confidence.

For API calls:
- Endpoint: `POST /predict`
- Form key: `image`
