"""
Evaluation script for XABSA model.

Usage:
    python scripts/eval.py --config configs/baseline.yaml --ckpt results/checkpoints/baseline/best_model.pt
    python scripts/eval.py --config configs/experiments/exp1_baseline_en.yaml --ckpt path/to/checkpoint.pt
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data.taxonomy import Taxonomy
from src.data.dataset import XABSADataset
from src.models.student import XABSAModel
from src.evaluation.metrics import MetricsTracker
from src.utils import setup_logging, set_seed, load_jsonl, save_json, ensure_dir

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate XABSA model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to evaluation data (overrides config)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split to evaluate on (e.g., 'test'). Overrides config."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Overrides config."
    )
    return parser.parse_args()


def load_data(config: Config, taxonomy: Taxonomy, tokenizer: AutoTokenizer, args):
    """
    Load evaluation dataset.

    Args:
        config: Configuration
        taxonomy: Taxonomy instance
        tokenizer: Tokenizer
        args: Command line arguments

    Returns:
        eval_dataloader
    """
    logger.info("Loading evaluation dataset...")

    # Load evaluation data
    if args.data:
        # Use specified data file
        eval_paths = [args.data]
    else:
        # Use config
        eval_paths = config.get("data.eval_paths", [])

    eval_data = []
    eval_splits = config.get("data.eval_splits", ["dev", "test"])

    # Override split if specified
    if args.split:
        eval_splits = [args.split]

    for path in eval_paths:
        logger.info(f"Loading data from {path}")
        data = load_jsonl(path)

        # Filter by split
        if args.split or eval_splits:
            filtered_data = [d for d in data if d.get("split") in eval_splits]
            eval_data.extend(filtered_data)
            logger.info(f"Loaded {len(filtered_data)} samples with splits {eval_splits} (total: {len(data)})")
        else:
            eval_data.extend(data)
            logger.info(f"Loaded {len(data)} samples (all splits)")

    logger.info(f"Total evaluation samples: {len(eval_data)}")

    if len(eval_data) == 0:
        logger.error("No evaluation data found!")
        sys.exit(1)

    # Create dataset
    max_length = config.get("data.max_length", 128)
    term_pooling = config.get("model.term_pooling", "mean")
    match_normalize_whitespace = config.get("data.match_normalize_whitespace", True)
    match_all_occurrences = config.get("data.match_all_occurrences", False)

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
    eval_stats = eval_dataset.get_stats()
    logger.info(f"Evaluation dataset stats: {eval_stats}")

    # Create dataloader
    batch_size = config.get("training.batch_size", 16)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=XABSADataset.collate_fn,
        num_workers=0  # Windows compatibility
    )

    return eval_dataloader, eval_data


def load_model(config: Config, checkpoint_path: str, device: torch.device):
    """
    Load model from checkpoint.

    Args:
        config: Configuration
        checkpoint_path: Path to checkpoint
        device: Device

    Returns:
        model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Create model
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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    return model


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_predictions: bool = False
):
    """
    Evaluate model.

    Args:
        model: Model
        dataloader: Evaluation dataloader
        device: Device
        save_predictions: Whether to save predictions

    Returns:
        metrics, predictions (if save_predictions=True)
    """
    logger.info("Evaluating...")

    model.eval()
    metrics_tracker = MetricsTracker()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []

    bio2id = {"O": 0, "B": 1, "I": 2}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device)
                else:
                    batch_device[key] = value

            # Forward pass
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
                term_masks=batch_device.get("term_masks"),
                bio_labels=batch_device["bio_labels"],
                category_labels=batch_device.get("category_labels"),
                polarity_labels=batch_device.get("polarity_labels"),
                term_category_labels=batch_device["term_category_labels"],
                term_polarity_labels=batch_device["term_polarity_labels"],
                num_terms=batch_device["num_terms"]
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
                gold_bio=batch_device["bio_labels"],
                pred_cat=pred_cat,
                gold_cat=batch_device["term_category_labels"],
                pred_pol=pred_pol,
                gold_pol=batch_device["term_polarity_labels"],
                n_terms=batch_device["num_terms"]
            )

            # Save predictions if requested
            if save_predictions:
                for i in range(len(batch_device["input_ids"])):
                    all_predictions.append({
                        "sample_id": batch["sample_ids"][i],
                        "text": batch["texts"][i],
                        "lang": batch["langs"][i],
                        "gold_triplets": batch["triplets"][i],
                        "pred_bio": pred_bio[i].cpu().tolist(),
                        "pred_categories": pred_cat[i].cpu().tolist(),
                        "pred_polarities": pred_pol[i].cpu().tolist()
                    })

    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = metrics_tracker.compute(bio2id=bio2id)
    metrics["loss"] = avg_loss

    return metrics, all_predictions if save_predictions else None


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load config
    config = Config(args.config)
    config_dict = config.to_dict()

    # Setup logging
    log_level = config.get("logging.log_level", "INFO")
    setup_logging(log_level)
    logger.info(f"Loaded config from {args.config}")

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
    eval_dataloader, eval_data = load_data(config, taxonomy, tokenizer, args)

    # Load model
    model = load_model(config, args.ckpt, device)

    # Evaluate
    save_predictions = args.save_predictions or config.get("evaluation.save_predictions", False)
    metrics, predictions = evaluate(model, eval_dataloader, device, save_predictions)

    # Print metrics
    logger.info("\n" + "="*50)
    logger.info("Evaluation Results")
    logger.info("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(config.get("output.output_dir", "results/eval"))

    ensure_dir(output_dir)

    metrics_path = output_dir / "eval_metrics.json"
    save_json(metrics, str(metrics_path))
    logger.info(f"\nSaved metrics to {metrics_path}")

    if predictions:
        predictions_path = output_dir / "predictions.jsonl"
        with open(predictions_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')
        logger.info(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
