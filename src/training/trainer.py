"""
Trainer for XABSA model.

Custom training loop with:
- Multi-task loss
- Early stopping
- Checkpoint saving
- Metrics tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Optional, Any
import json

from ..evaluation.metrics import MetricsTracker
from ..utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for XABSA model.

    Handles:
    - Training loop with gradient accumulation
    - Validation and early stopping
    - Checkpoint management
    - Logging and metrics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: torch.device,
        scheduler: Optional[Any] = None
    ):
        """
        Initialize trainer.

        Args:
            model: XABSA model
            train_dataloader: Training dataloader
            eval_dataloader: Evaluation dataloader
            optimizer: Optimizer
            config: Training configuration
            device: Device (cuda/cpu)
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.scheduler = scheduler

        # Training config
        self.num_epochs = config.get("training", {}).get("num_epochs", 10)
        self.max_grad_norm = config.get("training", {}).get("max_grad_norm", 1.0)
        self.loss_weights = config.get("training", {}).get("loss_weights", {
            "ate": 1.0, "category": 1.0, "polarity": 1.0
        })

        # Early stopping
        self.early_stopping_patience = config.get("training", {}).get("early_stopping_patience", 3)
        self.early_stopping_metric = config.get("training", {}).get("early_stopping_metric", "triplet_f1")
        self.save_best_by = config.get("training", {}).get("save_best_by", "triplet_f1")

        # Output dirs
        self.output_dir = Path(config.get("output", {}).get("output_dir", "results/default"))
        self.checkpoint_dir = Path(config.get("output", {}).get("checkpoint_dir", "results/checkpoints/default"))

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_dir)

        # Tracking
        self.best_metric = -float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self.epoch = 0

        # Metrics history
        self.train_history = []
        self.eval_history = []

        # BIO tag mapping
        self.bio2id = {"O": 0, "B": 1, "I": 2}

    def train(self):
        """
        Main training loop.
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"{'='*50}")

            # Train one epoch
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # Evaluate
            eval_metrics = self.evaluate()
            self.eval_history.append(eval_metrics)

            # Log metrics
            logger.info(f"\nEpoch {epoch + 1} Summary:")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Eval Metrics:")
            for key, value in eval_metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")

            # Check for improvement
            current_metric = eval_metrics.get(self.save_best_by, 0.0)
            if current_metric > self.best_metric:
                logger.info(f"\n*** New best {self.save_best_by}: {current_metric:.4f} (previous: {self.best_metric:.4f}) ***")
                self.best_metric = current_metric
                self.patience_counter = 0

                # Save best checkpoint
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                logger.info(f"No improvement for {self.patience_counter} epochs")

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # Save epoch checkpoint
            self.save_checkpoint(is_best=False)

        # Save final results
        self.save_results()
        logger.info(f"\nTraining completed!")
        logger.info(f"Best {self.save_best_by}: {self.best_metric:.4f}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {self.epoch + 1}")

        for batch in progress_bar:
            # Move batch to device
            batch = self._move_to_device(batch)

            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                term_masks=batch.get("term_masks"),
                bio_labels=batch["bio_labels"],
                category_labels=batch.get("category_labels"),
                polarity_labels=batch.get("polarity_labels"),
                term_category_labels=batch["term_category_labels"],
                term_polarity_labels=batch["term_polarity_labels"],
                num_terms=batch["num_terms"]
            )

            loss = outputs["loss"]

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Optimizer step
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            "loss": avg_loss,
            "epoch": self.epoch + 1,
            "global_step": self.global_step
        }

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on evaluation set.

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = self._move_to_device(batch)

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    term_masks=batch.get("term_masks"),
                    bio_labels=batch["bio_labels"],
                    category_labels=batch.get("category_labels"),
                    polarity_labels=batch.get("polarity_labels"),
                    term_category_labels=batch["term_category_labels"],
                    term_polarity_labels=batch["term_polarity_labels"],
                    num_terms=batch["num_terms"]
                )

                loss = outputs["loss"]
                total_loss += loss.item()
                num_batches += 1

                # Get predictions
                pred_bio = torch.argmax(outputs["ate_logits"], dim=-1)
                pred_cat = torch.argmax(outputs["category_logits"], dim=-1)
                pred_pol = torch.argmax(outputs["polarity_logits"], dim=-1)

                # Update metrics
                metrics_tracker.update(
                    pred_bio=pred_bio,
                    gold_bio=batch["bio_labels"],
                    pred_cat=pred_cat,
                    gold_cat=batch["term_category_labels"],
                    pred_pol=pred_pol,
                    gold_pol=batch["term_polarity_labels"],
                    n_terms=batch["num_terms"]
                )

        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = metrics_tracker.compute(bio2id=self.bio2id)
        metrics["loss"] = avg_loss

        return metrics

    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            "epoch": self.epoch + 1,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best checkpoint to {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.debug(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", -float('inf'))

        logger.info(f"Loaded checkpoint from epoch {self.epoch}")

    def save_results(self):
        """
        Save training results and metrics history.
        """
        results = {
            "best_metric": self.best_metric,
            "best_metric_name": self.save_best_by,
            "total_epochs": self.epoch + 1,
            "total_steps": self.global_step,
            "train_history": self.train_history,
            "eval_history": self.eval_history,
            "config": self.config
        }

        results_path = self.output_dir / "training_results.json"
        save_json(results, str(results_path))
        logger.info(f"Saved training results to {results_path}")

        # Save final metrics
        if self.eval_history:
            final_metrics = self.eval_history[-1]
            metrics_path = self.output_dir / "metrics.json"
            save_json(final_metrics, str(metrics_path))
            logger.info(f"Saved final metrics to {metrics_path}")

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch to device.

        Args:
            batch: Batch dictionary

        Returns:
            Batch on device
        """
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch
