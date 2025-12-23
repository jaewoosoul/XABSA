#!/usr/bin/env python3
"""
Data preparation script.

Parses raw data (SemEval, Korean) and converts to unified JSONL format.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.taxonomy import Taxonomy
from src.data.semeval_parser import SemEvalParser, parse_semeval_dataset
from src.data.korean_parser import KoreanParser, parse_korean_data
from src.utils import setup_logging, save_jsonl, ensure_dir


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare XABSA datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse SemEval data
  python scripts/prepare_data.py \\
    --semeval data/raw/semeval/restaurant \\
    --out data/processed

  # Parse Korean data
  python scripts/prepare_data.py \\
    --korean data/raw/korean/reviews.csv \\
    --korean-format csv \\
    --out data/processed

  # Parse both
  python scripts/prepare_data.py \\
    --semeval data/raw/semeval/restaurant \\
    --korean data/raw/korean/reviews.json \\
    --out data/processed
        """
    )

    parser.add_argument(
        "--semeval",
        type=str,
        help="Path to SemEval data directory (containing XML files)"
    )

    parser.add_argument(
        "--korean",
        type=str,
        help="Path to Korean data file (CSV/JSON/JSONL)"
    )

    parser.add_argument(
        "--korean-format",
        type=str,
        default="auto",
        choices=["auto", "csv", "json", "jsonl"],
        help="Korean data format (default: auto-detect from extension)"
    )

    parser.add_argument(
        "--korean-text-column",
        type=str,
        default="text",
        help="Text column name for CSV format (default: text)"
    )

    parser.add_argument(
        "--korean-has-labels",
        action="store_true",
        help="Whether Korean data has labels"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="data/processed",
        help="Output directory (default: data/processed)"
    )

    parser.add_argument(
        "--taxonomy",
        type=str,
        default="configs/taxonomy.yaml",
        help="Path to taxonomy file (default: configs/taxonomy.yaml)"
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language for SemEval data (default: en)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    return parser.parse_args()


def prepare_semeval(
    semeval_dir: str,
    output_dir: str,
    taxonomy: Taxonomy,
    lang: str = "en"
):
    """
    Prepare SemEval dataset.

    Args:
        semeval_dir: Directory containing SemEval XML files
        output_dir: Output directory
        taxonomy: Taxonomy instance
        lang: Language
    """
    logger.info(f"Processing SemEval data from {semeval_dir}")

    parser = SemEvalParser(taxonomy)
    splits = parser.parse_directory(semeval_dir, lang=lang)

    # Save each split
    for split, samples in splits.items():
        output_path = Path(output_dir) / f"{lang}_{split}.jsonl"
        save_jsonl(samples, str(output_path))
        logger.info(f"Saved {len(samples)} samples to {output_path}")


def prepare_korean(
    korean_path: str,
    output_dir: str,
    taxonomy: Taxonomy,
    format: str = "auto",
    has_labels: bool = False,
    text_column: str = "text"
):
    """
    Prepare Korean dataset.

    Args:
        korean_path: Path to Korean data file
        output_dir: Output directory
        taxonomy: Taxonomy instance
        format: Data format
        has_labels: Whether data has labels
        text_column: Text column name (for CSV)
    """
    logger.info(f"Processing Korean data from {korean_path}")

    # Determine split based on filename or default to "train"
    filename = Path(korean_path).stem
    if "train" in filename.lower():
        split = "train"
    elif "dev" in filename.lower() or "val" in filename.lower():
        split = "dev"
    elif "test" in filename.lower():
        split = "test"
    else:
        split = "train" if has_labels else "unlabeled"

    # Parse
    samples = parse_korean_data(
        korean_path,
        taxonomy_path=taxonomy.taxonomy_path,
        format=format,
        split=split,
        has_labels=has_labels,
        text_column=text_column
    )

    # Save
    if has_labels:
        output_path = Path(output_dir) / f"ko_{split}.jsonl"
    else:
        output_path = Path(output_dir) / "ko_raw.jsonl"

    save_jsonl(samples, str(output_path))
    logger.info(f"Saved {len(samples)} samples to {output_path}")


def main():
    """Main function."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load taxonomy
    logger.info(f"Loading taxonomy from {args.taxonomy}")
    taxonomy = Taxonomy(args.taxonomy)
    logger.info(f"Loaded {len(taxonomy.categories)} categories, "
                f"{len(taxonomy.polarities)} polarities")

    # Ensure output directory exists
    ensure_dir(args.out)

    # Process SemEval data
    if args.semeval:
        try:
            prepare_semeval(
                args.semeval,
                args.out,
                taxonomy,
                lang=args.lang
            )
        except Exception as e:
            logger.error(f"Error processing SemEval data: {e}")
            import traceback
            traceback.print_exc()

    # Process Korean data
    if args.korean:
        try:
            prepare_korean(
                args.korean,
                args.out,
                taxonomy,
                format=args.korean_format,
                has_labels=args.korean_has_labels,
                text_column=args.korean_text_column
            )
        except Exception as e:
            logger.error(f"Error processing Korean data: {e}")
            import traceback
            traceback.print_exc()

    if not args.semeval and not args.korean:
        logger.error("No input data specified. Use --semeval or --korean")
        sys.exit(1)

    logger.info("Data preparation completed!")


if __name__ == "__main__":
    main()
