# Fraud Detection Project Progress Report

## Current Status

The pipeline supports dataset ingestion, baseline training, SLM fine-tuning, and JSON-formatted inference output.

## What Has Been Completed

### Data Ingestion
- Added Kaggle-based dataset ingestion for SMS Spam Collection and phishing-related datasets.
- Implemented graceful fallback to synthetic data when real datasets are unavailable.
- Added dataset statistics reporting for sample count and label distribution.
- Documented dataset setup and download instructions.

### Baseline Model
- Implemented TF-IDF + SVM baseline training.
- Ensured the model save path is created automatically.
- Verified baseline training and inference in containerized and local workflows.

### SLM Fine-Tuning
- Replaced the placeholder SLM flow with a real HuggingFace Trainer-based fine-tuning pipeline.
- Added tokenization, PyTorch dataset handling, model saving, and model loading.
- Updated inference to prefer the SLM model and fall back to the baseline when needed.
- Confirmed local GPU availability with CUDA-enabled PyTorch in the Conda env.

### Inference and Output
- Inference now returns clean JSON output.
- Added `model_used` metadata to show whether SLM or baseline produced the result.
- Verified inference on both suspicious and benign sample texts.

### Environment and Infrastructure
- Docker build and container validation are working.
- A local Conda environment was created and tested on the host machine.
- The host machine has an NVIDIA GeForce RTX 3060 Ti and CUDA is available in the local env.

## Key Issues Resolved

- Fixed missing `PYTHONPATH` behavior in Docker.
- Fixed JSON serialization issues from NumPy scalar values.
- Fixed model directory creation before saving.
- Fixed dataset filename and encoding mismatches.
- Fixed `Trainer` dependency issues by adding `accelerate`.
- Recovered from Conda package collision issues by recreating a clean local environment.

## Validation Results

- Docker validation: `python -m pytest -q` passed with `1 passed, 3 warnings`.
- Docker training: baseline trained on synthetic fallback data and SLM fine-tuning completed successfully.
- Docker inference: JSON output returned correctly, including `model_used` for both SLM and baseline paths.
- Kaggle ingestion: both datasets downloaded successfully using the provided `kaggle.json` file.
- Local GPU check: `torch.cuda.is_available() == True` in `fraud_env`, with `NVIDIA GeForce RTX 3060 Ti` detected.

## Results

The project now has working end-to-end validation across both execution modes.

- Docker run: tests, baseline training, SLM training, and inference all completed successfully.
- Local Conda run: the environment was rebuilt cleanly, CUDA-enabled PyTorch was installed, and the host GPU is visible to PyTorch.
- Data ingestion: Kaggle downloads succeeded, and the SMS dataset now loads locally for training.
- Model behavior: inference returns clean JSON and uses the SLM by default, with baseline fallback still available.

## Current Limitation

The phishing dataset downloaded from Kaggle is feature-based rather than text-based, so it is not yet useful for the text fine-tuning path. The SMS Spam Collection dataset is text-based and is currently the practical dataset for local SLM training.

## Next Steps

1. Run a full local GPU training pass with the new Conda env.
2. Decide whether to add a text-based phishing email dataset or adapt the feature-based phishing dataset into a separate branch of the pipeline.
3. Continue with the RAG pipeline: embeddings, FAISS retrieval, and context-aware explanations.
4. Expand the explanation layer beyond rule-based output.

## One-Line Summary

The project is now operational, validated, and GPU-ready locally; the next major feature is completing the RAG pipeline.
