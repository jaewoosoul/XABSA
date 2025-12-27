"""
Training script for XABSA model.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/experiments/exp1_baseline_en.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data.taxonomy import Taxonomy
from src.data.dataset import XABSADataset
from src.models.student import XABSAModel
from src.training.trainer import Trainer
from src.utils import setup_logging, set_seed, load_jsonl, init_wandb

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train XABSA model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Overrides config."
    )
    return parser.parse_args()


def load_data(config: Config, taxonomy: Taxonomy, tokenizer: AutoTokenizer):
    """
    Load training and evaluation datasets.

    Args:
        config: Configuration
        taxonomy: Taxonomy instance
        tokenizer: Tokenizer

    Returns:
        train_dataloader, eval_dataloader
    """
    logger.info("Loading datasets...")

    # Load training data
    train_data = []
    train_paths = config.get("data.train_paths", [])
    train_splits = config.get("data.train_splits", ["train"])

    for path in train_paths:
        logger.info(f"Loading training data from {path}")
        data = load_jsonl(path)
        # Filter by split
        filtered_data = [d for d in data if d.get("split") in train_splits]
        train_data.extend(filtered_data)
        logger.info(f"Loaded {len(filtered_data)} samples (total: {len(data)})")

    logger.info(f"Total training samples: {len(train_data)}")

    # Load evaluation data
    eval_data = []
    eval_paths = config.get("data.eval_paths", [])
    eval_splits = config.get("data.eval_splits", ["dev", "test"])

    for path in eval_paths:
        logger.info(f"Loading evaluation data from {path}")
        data = load_jsonl(path)
        # Filter by split
        filtered_data = [d for d in data if d.get("split") in eval_splits]
        eval_data.extend(filtered_data)
        logger.info(f"Loaded {len(filtered_data)} samples (total: {len(data)})")

    logger.info(f"Total evaluation samples: {len(eval_data)}")

    # Create datasets
    max_length = config.get("data.max_length", 128)
    term_pooling = config.get("model.term_pooling", "mean")
    match_normalize_whitespace = config.get("data.match_normalize_whitespace", True)
    match_all_occurrences = config.get("data.match_all_occurrences", False)

    train_dataset = XABSADataset(
        data=train_data,
        tokenizer=tokenizer,
        taxonomy=taxonomy,
        max_length=max_length,
        term_pooling=term_pooling,
        match_normalize_whitespace=match_normalize_whitespace,
        match_all_occurrences=match_all_occurrences
    )

    eval_dataset = XABSADataset(
        data=eval_data,
        tokenizer=tokenizer,
        taxonomy=taxonomy,
        max_length=max_length,
        term_pooling=term_pooling,
        match_normalize_whitespace=match_normalize_whitespace,
        match_all_occurrences=match_all_occurrences
    )

    # Log dataset statistics
    train_stats = train_dataset.get_stats()
    eval_stats = eval_dataset.get_stats()
    logger.info(f"Training dataset stats: {train_stats}")
    logger.info(f"Evaluation dataset stats: {eval_stats}")

    # Create dataloaders
    batch_size = config.get("training.batch_size", 16)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=XABSADataset.collate_fn,
        num_workers=0  # Windows compatibility
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=XABSADataset.collate_fn,
        num_workers=0
    )

    return train_dataloader, eval_dataloader


def create_model(config: Config, device: torch.device):
    """
    Create XABSA model.

    Args:
        config: Configuration
        device: Device

    Returns:
        model
    """
    logger.info("Creating model...")

    backbone = config.get("model.backbone", "xlm-roberta-base")
    num_ate_labels = config.get("model.num_ate_labels", 3)
    num_category_labels = config.get("model.num_category_labels", 13)
    num_polarity_labels = config.get("model.num_polarity_labels", 3)
    dropout = config.get("model.dropout", 0.1)
    term_pooling = config.get("model.term_pooling", "mean")

    model = XABSAModel(
        backbone=backbone,
        num_ate_labels=num_ate_labels,
        num_category_labels=num_category_labels,
        num_polarity_labels=num_polarity_labels,
        dropout=dropout,
        term_pooling=term_pooling
    )

    model.to(device)

    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {backbone}")
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable_params:,}")

    return model


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: Config,
    num_training_steps: int
):
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: Model
        config: Configuration
        num_training_steps: Total number of training steps

    Returns:
        optimizer, scheduler
    """
    lr = config.get("training.learning_rate", 2e-5)
    weight_decay = config.get("training.weight_decay", 0.01)
    warmup_ratio = config.get("training.warmup_ratio", 0.1)

    # Convert to float if string (YAML sometimes parses scientific notation as string)
    if isinstance(lr, str):
        lr = float(lr)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    if isinstance(warmup_ratio, str):
        warmup_ratio = float(warmup_ratio)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Create scheduler
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    logger.info(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
    logger.info(f"Scheduler: Linear with warmup (warmup_steps={num_warmup_steps}, total_steps={num_training_steps})")

    return optimizer, scheduler


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    config = Config(args.config)
    config_dict = config.to_dict()

    # Setup logging
    log_level = config.get("logging.log_level", "INFO")
    setup_logging(log_level)
    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Experiment: {config.get('experiment.name', 'unknown')}")

    # Set seed
    seed = config.get("experiment.seed", 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device_name = config.get("device", "cuda")
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load taxonomy
    taxonomy_path = config.get("taxonomy.taxonomy_path", "configs/taxonomy.yaml")
    taxonomy = Taxonomy(taxonomy_path)
    logger.info(f"Loaded taxonomy: {taxonomy}")

    # Load tokenizer
    backbone = config.get("model.backbone", "xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    logger.info(f"Loaded tokenizer: {backbone}")

    # Load data
    train_dataloader, eval_dataloader = load_data(config, taxonomy, tokenizer)

    # Create model
    model = create_model(config, device)

    # Calculate total training steps
    num_epochs = config.get("training.num_epochs", 10)
    num_training_steps = len(train_dataloader) * num_epochs

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, num_training_steps
    )

    # Initialize wandb (optional)
    init_wandb(config_dict, "train")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        config=config_dict,
        device=device,
        scheduler=scheduler
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("\n" + "="*50)
    logger.info("Starting training...")
    logger.info("="*50 + "\n")

    trainer.train()

    logger.info("\n" + "="*50)
    logger.info("Training complete!")
    logger.info("="*50)


if __name__ == "__main__":
    main()
