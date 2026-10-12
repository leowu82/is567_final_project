# Dataset Setup Guide

This document explains how to download and prepare the training datasets for the fraud detection system.

## Overview

The system supports three datasets:

1. **SMS Spam Collection** – 5,574 SMS messages (87% legitimate, 13% spam)
2. **Phishing Email Dataset** – ~82,500 emails (well-balanced)
3. **RAG Knowledge Base** – FTC, PhishTank, r/Scams data (future integration)

## Prerequisites

- `kaggle` CLI installed (included in the Docker environment)
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

## Downloading Datasets

### Option 1: Using Docker with Kaggle Credentials

If you have Kaggle credentials, mount them into the Docker container and download datasets:

```bash
# Windows PowerShell
docker run --rm \
  -v "C:\Users\YourUsername\.kaggle:/root/.kaggle:ro" \
  -v "C:\Users\User\Desktop\sp26\is567\project:/workspace" \
  -w /workspace \
  fraud-detect:latest bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate fraud_env && \
    python data/ingest_kaggle.py --dataset uciml/sms-spam-collection-dataset && \
    python data/ingest_kaggle.py --dataset shashwatwork/phishing-dataset-for-machine-learning'
```

### Option 2: Manual Download

#### SMS Spam Collection

1. Visit: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Download and extract to: `data/raw/SMS Spam Collection/`
3. Expected file: `data/raw/SMS Spam Collection/SMSSpamCollection`

#### Phishing Email Dataset

1. Visit: https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning
2. Download and extract to: `data/raw/`
3. Expected file structure: `data/raw/emails.csv` (or similar)

## File Structure After Download

```
data/
├── raw/
│   ├── SMS Spam Collection/
│   │   └── SMSSpamCollection       (tab-separated: label \t message)
│   └── emails.csv                  (CSV with text and label columns)
└── ingest_kaggle.py
```

## Using the Datasets

### Automatic Loading

The training scripts automatically detect and load downloaded datasets:

```bash
# Load SMS or Phishing datasets (whichever are available)
python scripts/train_baseline.py

# Same for SLM training
python scripts/train_slm.py
```

### Programmatic Usage

```python
from fraud_detect.data_utils import load_combined_dataset, get_dataset_stats

# Load both datasets (if available)
texts, labels = load_combined_dataset()

# Check dataset composition
stats = get_dataset_stats(texts, labels)
print(stats)
# Output: {'total_samples': 5574, 'label_distribution': {0: 4827, 1: 747}, ...}
```

### Custom Loading

```python
from fraud_detect.data_utils import load_sms_spam_collection, load_phishing_emails

# Load only SMS data
sms_texts, sms_labels = load_sms_spam_collection()

# Load only Phishing data
phishing_texts, phishing_labels = load_phishing_emails()
```

## Dataset Formats

### SMS Spam Collection

- **Format**: Tab-separated text file (no header)
- **Columns**: `label \t text`
- **Labels**: `ham` (legitimate) → 0, `spam` → 1
- **Size**: 5,574 messages
- **Class Distribution**: ~87% ham, ~13% spam

Example:
```
ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121. C Ts C Ex cond ol D16S45ox2op3tagM18 11
```

### Phishing Email Dataset

- **Format**: CSV with header
- **Required Columns**: 
  - Email/Text (message content)
  - Prediction/Label (0=legitimate, 1=phishing)
- **Size**: ~82,500 emails
- **Class Distribution**: Balanced (50/50)

## Fallback Behavior

If no datasets are downloaded, the training scripts will:

1. Print a warning message
2. Fall back to small synthetic datasets (4-2 samples)
3. Complete training successfully (useful for testing)

This allows the system to run and validate without requiring large dataset downloads.

## Dataset Statistics

After downloading and training, you'll see output like:

```
✓ Loaded 5574 SMS messages
✓ Loaded 82500 phishing emails
Dataset loaded: 88074 samples
Label distribution: {0: 45137, 1: 42937}
Average text length: 145.3 chars
```

## Storage Recommendations

- SMS dataset: ~1 MB
- Phishing email dataset: ~100-500 MB (depending on which one you use)
- Total: Plan for ~1-2 GB free space in `data/raw/`

## Troubleshooting

### "Dataset not found" error

Make sure you've extracted the files to the correct location. Check:

```bash
# On host (Windows)
dir C:\Users\User\Desktop\sp26\is567\project\data\raw

# In Docker
docker run --rm -v "C:\Users\User\Desktop\sp26\is567\project:/workspace" \
  -w /workspace fraud-detect:latest ls -la data/raw/
```

### Kaggle authentication issues

Ensure `kaggle.json` is in the right location:

- **Host**: `%USERPROFILE%/.kaggle/kaggle.json`
- **Docker**: `/root/.kaggle/kaggle.json`

When running in Docker, mount the credentials:

```bash
docker run -v "C:\Users\YourUsername\.kaggle:/root/.kaggle:ro" ...
```

### Missing columns in CSV

If you use a different phishing dataset, the script will try to auto-detect text and label columns. If it fails, you can manually load and transform the data:

```python
import pandas as pd
from fraud_detect.gatekeeper.preprocessing import normalize_text

df = pd.read_csv('data/raw/custom_dataset.csv')
texts = df['your_text_column'].tolist()
labels = df['your_label_column'].tolist()

# Normalize and train
texts = [normalize_text(t) for t in texts]
from fraud_detect.gatekeeper.vectorizer import train_baseline, save_model
model = train_baseline(texts, labels)
save_model(model, 'models/baseline.joblib')
```

## Next Steps

Once datasets are downloaded:

1. Train the baseline model: `python scripts/train_baseline.py`
2. Train the SLM: `python scripts/train_slm.py`
3. Test inference: `python scripts/run_inference.py --text "your test message"`

For RAG knowledge base integration, see the RAG documentation (coming soon).
