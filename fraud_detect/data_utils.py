"""Utilities for loading fraud detection datasets."""
import os
from typing import Tuple, List
import pandas as pd


def load_sms_spam_collection(data_dir: str = "data/raw") -> Tuple[List[str], List[int]]:
    """Load SMS Spam Collection dataset from Kaggle.
    
    Dataset: uciml/sms-spam-collection-dataset
    Looks for: SMSSpamCollection (tab-separated) or spam.csv (CSV format)
    Columns: label (ham/spam), text
    
    Returns:
        (texts, labels) where labels are 0=ham (legitimate), 1=spam
    """
    # Try both naming patterns
    candidates = [
        os.path.join(data_dir, "SMS Spam Collection", "SMSSpamCollection"),
        os.path.join(data_dir, "SMSSpamCollection"),
        os.path.join(data_dir, "spam.csv"),  # Alternative extracted name
    ]
    
    spam_file = None
    for candidate in candidates:
        if os.path.exists(candidate):
            spam_file = candidate
            break
    
    if spam_file is None:
        raise FileNotFoundError(
            f"SMS Spam Collection dataset not found in {data_dir}. "
            "Download it with: python data/ingest_kaggle.py --dataset uciml/sms-spam-collection-dataset"
        )
    
    texts = []
    labels = []
    
    # Handle CSV format (spam.csv)
    if spam_file.endswith('.csv'):
        df = None
        # Try multiple encodings for CSV files
        for encoding in ['utf-8', 'iso-8859-1', 'cp1252', 'latin-1']:
            try:
                df = pd.read_csv(spam_file, encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if df is None:
            raise ValueError(f"Could not read {spam_file} with any standard encoding")
        
        # Detect columns
        text_cols = [col for col in df.columns if col.lower() in ['v2', 'text', 'message', 'email']]
        label_cols = [col for col in df.columns if col.lower() in ['v1', 'label', 'class', 'type']]
        
        if text_cols and label_cols:
            text_col = text_cols[0]
            label_col = label_cols[0]
            for _, row in df.iterrows():
                text = str(row[text_col]).strip()
                label = 1 if str(row[label_col]).lower() == 'spam' else 0
                if text:
                    texts.append(text)
                    labels.append(label)
    else:
        # Handle tab-separated format (SMSSpamCollection)
        with open(spam_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    label_str, text = parts
                    label = 1 if label_str.lower() == 'spam' else 0
                    texts.append(text)
                    labels.append(label)
    
    return texts, labels


def load_phishing_emails(data_dir: str = "data/raw") -> Tuple[List[str], List[int]]:
    """Load Phishing Email dataset from Kaggle.
    
    Dataset: shashwatwork/phishing-dataset-for-machine-learning
    Note: The downloaded dataset is actually feature vectors, not email text.
    For text-based phishing detection, we return synthetic examples.
    
    Returns:
        (texts, labels) tuple
    """
    # Check if feature-based phishing dataset exists
    phishing_feature_file = os.path.join(data_dir, "Phishing_Legitimate_full.csv")
    
    if os.path.exists(phishing_feature_file):
        # This dataset has features, not text. Skip it.
        raise FileNotFoundError(
            f"Phishing dataset at {phishing_feature_file} contains feature vectors, not text. "
            "For this project, we use SMS Spam Collection (text-based). "
            "To add text-based phishing email detection, download a text-based phishing email dataset."
        )
    
    # Try other naming patterns for text-based datasets
    candidates = [
        os.path.join(data_dir, "Phishing Emails", "emails.csv"),
        os.path.join(data_dir, "phishing", "emails.csv"),
        os.path.join(data_dir, "phishing_dataset", "emails.csv"),
        os.path.join(data_dir, "emails.csv"),
    ]
    
    found_file = None
    for candidate in candidates:
        if os.path.exists(candidate):
            found_file = candidate
            break
    
    if found_file is None:
        raise FileNotFoundError(
            f"Text-based Phishing Email dataset not found. "
            "Download a phishing email dataset from Kaggle with email text content."
        )
    
    # Try multiple encodings
    df = None
    for encoding in ['utf-8', 'iso-8859-1', 'cp1252', 'latin-1']:
        try:
            df = pd.read_csv(found_file, encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if df is None:
        raise ValueError(f"Could not read {found_file} with any standard encoding")
    
    # Detect text column (try common names)
    text_cols = [col for col in df.columns if col.lower() in ['email', 'text', 'message', 'content', 'body']]
    label_cols = [col for col in df.columns if col.lower() in ['label', 'prediction', 'is_phishing', 'class']]
    
    if not text_cols or not label_cols:
        raise ValueError(
            f"Could not detect text and label columns in {found_file}. "
            f"Found columns: {df.columns.tolist()}"
        )
    
    text_col = text_cols[0]
    label_col = label_cols[0]
    
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[label_col].tolist()
    
    # Normalize labels to 0/1 if needed
    labels = [1 if (label == 1 or str(label).lower() == 'phishing') else 0 for label in labels]
    
    return texts, labels


def load_combined_dataset(
    include_sms: bool = True,
    include_phishing: bool = True,
    sms_dir: str = "data/raw/SMS Spam Collection",
    phishing_dir: str = "data/raw",
) -> Tuple[List[str], List[int]]:
    """Load and combine SMS and Phishing datasets.
    
    Args:
        include_sms: Whether to include SMS Spam Collection
        include_phishing: Whether to include Phishing Email dataset
        sms_dir: Directory containing SMS dataset
        phishing_dir: Directory containing Phishing dataset
    
    Returns:
        (texts, labels) combined dataset
    """
    all_texts = []
    all_labels = []
    
    if include_sms:
        try:
            texts, labels = load_sms_spam_collection(sms_dir if os.path.exists(sms_dir) else phishing_dir)
            all_texts.extend(texts)
            all_labels.extend(labels)
            print(f"✓ Loaded {len(texts)} SMS messages")
        except FileNotFoundError as e:
            print(f"⚠ SMS dataset not available: {e}")
    
    if include_phishing:
        try:
            texts, labels = load_phishing_emails(phishing_dir)
            all_texts.extend(texts)
            all_labels.extend(labels)
            print(f"✓ Loaded {len(texts)} phishing emails")
        except FileNotFoundError as e:
            print(f"⚠ Phishing dataset not available: {e}")
    
    if not all_texts:
        raise RuntimeError("No datasets available. Please download at least one dataset.")
    
    return all_texts, all_labels


def get_dataset_stats(texts: List[str], labels: List[int]) -> dict:
    """Return basic statistics about the dataset."""
    unique_labels = set(labels)
    label_counts = {label: labels.count(label) for label in unique_labels}
    
    return {
        "total_samples": len(texts),
        "label_distribution": label_counts,
        "avg_text_length": sum(len(t) for t in texts) / len(texts) if texts else 0,
        "unique_labels": len(unique_labels),
    }
