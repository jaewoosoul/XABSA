"""
Utility functions for XABSA project
"""

import yaml
import logging
import csv
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List

# Optional imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    Note: For more advanced config loading with inheritance,
    use src.config.Config class instead.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(config: Dict[str, Any], task_name: str):
    """Initialize wandb logging."""
    logger = logging.getLogger(__name__)

    if config.get("logging", {}).get("use_wandb", False):
        if not HAS_WANDB:
            logger.warning("wandb is not installed. Skipping wandb initialization.")
            return
        wandb.init(
            project=config["logging"].get("wandb_project", f"xabsa-{task_name}"),
            entity=config["logging"].get("wandb_entity"),
            config=config
        )


def log_to_csv(metrics: Dict[str, float], csv_path: str):
    """Log metrics to CSV file."""
    file_exists = Path(csv_path).exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_jsonl(path: str) -> List[Dict]:
    """
    Load JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """
    Save data to JSONL file.

    Args:
        data: List of dictionaries
        path: Output path
    """
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, path: str, indent: int = 2):
    """Save data to JSON file."""
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

