"""
Model Fine-tuning Script
========================
Fine-tune DistilBERT on the red-team test suite.
"""

import torch
import pandas as pd
import numpy as np
import logging
import os
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptDataset(Dataset):
    """Dataset for prompt classification."""

    CATEGORIES = ["safe", "jailbreak", "pii_extraction", "toxic_content", "prompt_injection"]
    CATEGORY_TO_ID = {cat: idx for idx, cat in enumerate(CATEGORIES)}

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        """
        Args:
            df: DataFrame with columns ['prompt', 'category', 'label']
            tokenizer: HuggingFace tokenizer
            max_length: Max token length
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = row['prompt']
        category = row['category']

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get label (0 = safe, 1-4 = attacks)
        label = self.CATEGORY_TO_ID[category]

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class FineTuner:
    """Fine-tune DistilBERT on safety classification."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        output_dir: str = "./models/fine_tuned_model",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load tokenizer and model
        logger.info(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=5  # 5 categories
        )
        self.model.to(device)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0
    ) -> dict:
        """
        Fine-tune the model.

        Args:
            train_df: Training data
            val_df: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            max_grad_norm: Max gradient norm for clipping

        Returns:
            Training history
        """
        logger.info(f"Training on {len(train_df)} examples, validating on {len(val_df)}")

        # Create datasets and dataloaders
        train_dataset = PromptDataset(train_df, self.tokenizer)
        val_dataset = PromptDataset(val_df, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'timestamps': []
        }

        for epoch in range(epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"{'='*50}")

            # Training
            train_loss = self._train_epoch(
                train_loader,
                optimizer,
                scheduler,
                max_grad_norm
            )
            history['train_loss'].append(train_loss)

            # Validation
            val_loss, val_accuracy = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['timestamps'].append(datetime.now().isoformat())

            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {val_accuracy:.4f}")

        # Save model and tokenizer
        logger.info(f"\nSaving model to {self.output_dir}...")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # Save training history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info("✅ Fine-tuning complete!")
        return history

    def _train_epoch(self, train_loader, optimizer, scheduler, max_grad_norm):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def _validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy


def main():
    """Main fine-tuning script."""

    # Load data
    logger.info("Loading test suite...")
    test_suite = pd.read_csv("data/test_suite.csv")
    logger.info(f"Loaded {len(test_suite)} examples")

    # Split into train/val
    np.random.seed(42)
    indices = np.random.permutation(len(test_suite))
    train_size = int(0.8 * len(test_suite))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_df = test_suite.iloc[train_indices].reset_index(drop=True)
    val_df = test_suite.iloc[val_indices].reset_index(drop=True)

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Print distribution
    logger.info("\nTraining set distribution:")
    for cat, count in train_df['category'].value_counts().items():
        logger.info(f"  {cat}: {count}")

    # Fine-tune
    tuner = FineTuner(
        model_name="distilbert-base-uncased",
        output_dir="./models/fine_tuned_model",
        device="cpu"  # Change to "cuda" if GPU available
    )

    history = tuner.train(
        train_df=train_df,
        val_df=val_df,
        epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        warmup_steps=500
    )

    logger.info("\n" + "="*50)
    logger.info("Training Summary")
    logger.info("="*50)
    logger.info(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Final Val Accuracy: {history['val_accuracy'][-1]:.4f}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
