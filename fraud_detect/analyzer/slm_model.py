import os
import json
import numpy as np
from typing import Optional, Tuple, Dict, Any
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
import torch


def _select_device() -> torch.device:
    """Select the best available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextClassificationDataset(Dataset):
    """PyTorch Dataset for text classification with tokenization."""
    
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def fine_tune(
    train_texts: list,
    train_labels: list,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "models/slm",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 128
) -> str:
    """Fine-tune a transformer model on text classification task.
    
    Args:
        train_texts: List of input texts
        train_labels: List of binary labels (0 or 1)
        model_name: HuggingFace model identifier
        output_dir: Directory to save model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        max_length: Max token length for tokenizer
    
    Returns:
        Path to saved model directory
    """
    os.makedirs(output_dir, exist_ok=True)
    device = _select_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Create dataset
    print(f"Tokenizing {len(train_texts)} samples...")
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length=max_length
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="epoch",
        logging_steps=max(1, len(train_dataset) // (batch_size * 2)),
        weight_decay=0.01,
        seed=42,
        fp16=(device.type == "cuda"),
        dataloader_pin_memory=(device.type == "cuda"),
        use_cpu=(device.type != "cuda"),
        report_to=[],  # Disable wandb/tensorboard
    )
    
    # Data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"Training for {num_epochs} epochs...")
    trainer.train()
    
    # Save model and tokenizer
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    meta = {
        "model_name": model_name,
        "n_samples": len(train_texts),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    return output_dir

def load_model(path: str) -> Optional[Dict[str, Any]]:
    """Load fine-tuned model and tokenizer from directory.
    
    Args:
        path: Path to saved model directory
    
    Returns:
        Dictionary with 'model' and 'tokenizer', or None if not found
    """
    if not os.path.exists(path):
        return None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        
        # Load metadata
        meta_path = os.path.join(path, "meta.json")
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'metadata': metadata,
            'path': path
        }
    except Exception as e:
        print(f"Failed to load SLM model: {e}")
        return None

def predict_slm(model_dict: Dict[str, Any], texts: list, max_length: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference with fine-tuned model.
    
    Args:
        model_dict: Output from load_model()
        texts: List of input texts
        max_length: Max token length
    
    Returns:
        Tuple of (predictions, probabilities) where predictions are class indices
    """
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = _select_device()
    
    # Set to eval mode
    model = model.to(device)
    model.eval()
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            encoding = {key: value.to(device) for key, value in encoding.items()}
            
            outputs = model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)
            
            predictions.append(pred.item())
            probabilities.append(probs[0].detach().cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)
