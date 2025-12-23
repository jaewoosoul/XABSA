#!/usr/bin/env python3
"""
Run LLM Teacher to generate pseudo-labels.
"""

import argparse
import logging
from pathlib import Path
import sys
import os
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.taxonomy import Taxonomy
from src.teacher import OpenAITeacher, ClaudeTeacher, GeminiTeacher, MockTeacher, TripletValidator, PseudoLabelFilter
from src.utils import setup_logging, load_jsonl, save_jsonl, save_json, ensure_dir
from src.config import load_config


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels using LLM Teacher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate pseudo-labels with OpenAI
  python scripts/run_teacher.py \\
    --input data/processed/ko_raw.jsonl \\
    --output data/pseudo/ko_pseudo.jsonl \\
    --teacher openai \\
    --model gpt-4-turbo-preview

  # Generate with Claude
  python scripts/run_teacher.py \\
    --input data/processed/ko_raw.jsonl \\
    --output data/pseudo/ko_pseudo.jsonl \\
    --teacher claude \\
    --model claude-3-sonnet-20240229

  # Use config file
  python scripts/run_teacher.py \\
    --input data/processed/ko_raw.jsonl \\
    --output data/pseudo/ko_pseudo.jsonl \\
    --config configs/teacher.yaml

  # With filtering
  python scripts/run_teacher.py \\
    --input data/processed/ko_raw.jsonl \\
    --output data/pseudo/ko_pseudo.jsonl \\
    --teacher openai \\
    --filter
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file (unlabeled data)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file (pseudo-labeled data)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to teacher config file (optional)"
    )

    parser.add_argument(
        "--teacher",
        type=str,
        choices=["openai", "claude", "gemini", "mock"],
        default="gemini",
        help="Teacher type (default: gemini, use 'mock' for testing without API)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides config)"
    )

    parser.add_argument(
        "--taxonomy",
        type=str,
        default="configs/taxonomy.yaml",
        help="Path to taxonomy file (default: configs/taxonomy.yaml)"
    )

    parser.add_argument(
        "--filter",
        action="store_true",
        help="Apply filtering to pseudo-labels"
    )

    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw pseudo-labels before filtering"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    return parser.parse_args()


def create_teacher(teacher_type: str, config: Dict, model: str = None):
    """
    Create teacher instance.

    Args:
        teacher_type: Teacher type (openai, claude, gemini, mock)
        config: Config dictionary
        model: Model name (optional, overrides config)

    Returns:
        Teacher instance
    """
    teacher_config = config.get("teacher", {})

    # Get model name
    if model is None:
        model = teacher_config.get("model", "gemini-1.5-flash")

    # Get common settings
    temperature = teacher_config.get("temperature", 0.0)
    max_tokens = teacher_config.get("max_tokens", 1000)
    max_retries = teacher_config.get("max_retries", 3)

    # Create teacher
    if teacher_type == "openai":
        api_key = os.getenv(teacher_config.get("api_key_env", "OPENAI_API_KEY"))
        teacher = OpenAITeacher(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries
        )
    elif teacher_type == "claude":
        api_key = os.getenv(teacher_config.get("api_key_env", "ANTHROPIC_API_KEY"))
        teacher = ClaudeTeacher(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries
        )
    elif teacher_type == "gemini":
        api_key = os.getenv(teacher_config.get("api_key_env", "GOOGLE_API_KEY"))
        teacher = GeminiTeacher(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries
        )
    elif teacher_type == "mock":
        teacher = MockTeacher(
            model=model or "mock",
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries
        )
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")

    logger.info(f"Created {teacher}")
    return teacher


def create_filter(config: Dict, taxonomy: Taxonomy):
    """
    Create filter instance.

    Args:
        config: Config dictionary
        taxonomy: Taxonomy instance

    Returns:
        Filter instance
    """
    filter_config = config.get("filtering", {})

    filter = PseudoLabelFilter(
        taxonomy=taxonomy,
        check_term_existence=filter_config.get("check_term_existence", True),
        remove_duplicates=filter_config.get("remove_duplicates", True),
        normalize_whitespace=filter_config.get("normalize_whitespace", True),
        validate_category=filter_config.get("validate_category", True),
        map_to_etc=filter_config.get("map_to_etc", True),
        max_triplets_per_text=filter_config.get("max_triplets_per_text", 8),
        use_self_consistency=filter_config.get("use_self_consistency", False),
        consistency_threshold=filter_config.get("consistency_threshold", 0.6)
    )

    logger.info(f"Created filter with settings: {filter_config}")
    return filter


def generate_pseudo_labels(
    teacher,
    samples: List[Dict],
    lang: str = "ko"
) -> List[Dict]:
    """
    Generate pseudo-labels for samples.

    Args:
        teacher: Teacher instance
        samples: List of samples
        lang: Language

    Returns:
        Samples with pseudo-labels
    """
    from tqdm import tqdm

    pseudo_labeled = []

    for sample in tqdm(samples, desc="Generating pseudo-labels"):
        text = sample["text"]

        try:
            # Generate triplets
            triplets = teacher.generate_triplets(text, lang=lang)

            # Create pseudo-labeled sample
            pseudo_sample = sample.copy()
            pseudo_sample["gold_triplets"] = triplets

            pseudo_labeled.append(pseudo_sample)

        except Exception as e:
            logger.error(f"Error generating triplets for sample {sample.get('id')}: {e}")
            # Add empty triplets
            pseudo_sample = sample.copy()
            pseudo_sample["gold_triplets"] = []
            pseudo_labeled.append(pseudo_sample)

    return pseudo_labeled


def main():
    """Main function."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)
    else:
        logger.info("Using default config")
        config = {
            "teacher": {
                "type": args.teacher,
                "model": args.model or "gpt-4-turbo-preview",
                "temperature": 0.0,
                "max_tokens": 1000,
                "max_retries": 3
            },
            "filtering": {
                "check_term_existence": True,
                "remove_duplicates": True,
                "validate_category": True,
                "max_triplets_per_text": 8
            }
        }

    # Load taxonomy
    logger.info(f"Loading taxonomy from {args.taxonomy}")
    taxonomy = Taxonomy(args.taxonomy)

    # Load input data
    logger.info(f"Loading input data from {args.input}")
    samples = load_jsonl(args.input)

    # Limit samples if specified
    if args.max_samples:
        samples = samples[:args.max_samples]
        logger.info(f"Limited to {len(samples)} samples")

    logger.info(f"Loaded {len(samples)} samples")

    # Detect language from first sample
    lang = samples[0].get("lang", "ko") if samples else "ko"
    logger.info(f"Language: {lang}")

    # Create teacher
    teacher = create_teacher(
        args.teacher,
        config,
        model=args.model
    )

    # Generate pseudo-labels
    logger.info("Generating pseudo-labels...")
    pseudo_labeled = generate_pseudo_labels(teacher, samples, lang=lang)

    # Save raw pseudo-labels
    if args.save_raw or args.filter:
        raw_output_path = str(Path(args.output).parent / "ko_pseudo_raw.jsonl")
        ensure_dir(Path(raw_output_path).parent)
        save_jsonl(pseudo_labeled, raw_output_path)
        logger.info(f"Saved raw pseudo-labels to {raw_output_path}")

    # Apply filtering
    if args.filter:
        logger.info("Applying filtering...")
        filter = create_filter(config, taxonomy)

        filtered = filter.filter_batch(pseudo_labeled)

        # Save filtered
        ensure_dir(Path(args.output).parent)
        save_jsonl(filtered, args.output)
        logger.info(f"Saved {len(filtered)} filtered samples to {args.output}")

        # Save statistics
        stats = filter.get_summary()
        stats_path = str(Path(args.output).parent / "summary.json")
        save_json(stats, stats_path)
        logger.info(f"Saved filtering statistics to {stats_path}")

        # Print summary
        logger.info("\n=== Filtering Summary ===")
        logger.info(f"Total before: {stats['total_before']}")
        logger.info(f"Total after: {stats['total_after']}")
        logger.info(f"Removed: {stats['removed']}")
        logger.info(f"Retention rate: {stats['retention_rate']:.2%}")
        logger.info("\nBreakdown:")
        for key, value in stats['breakdown'].items():
            logger.info(f"  {key}: {value}")

    else:
        # Save without filtering
        ensure_dir(Path(args.output).parent)
        save_jsonl(pseudo_labeled, args.output)
        logger.info(f"Saved {len(pseudo_labeled)} pseudo-labeled samples to {args.output}")

    logger.info("Pseudo-label generation completed!")


if __name__ == "__main__":
    main()
