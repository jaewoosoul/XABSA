#!/usr/bin/env python3
"""
Download SemEval ABSA datasets.

This script provides instructions and utilities for downloading SemEval datasets.
Note: Due to licensing, we cannot auto-download SemEval data.
Users must manually download from the official sources.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, ensure_dir


logger = logging.getLogger(__name__)


SEMEVAL_INFO = {
    "2014": {
        "name": "SemEval-2014 Task 4",
        "url": "http://alt.qcri.org/semeval2014/task4/",
        "datasets": ["Restaurant", "Laptop"],
        "description": "Aspect-Based Sentiment Analysis"
    },
    "2015": {
        "name": "SemEval-2015 Task 12",
        "url": "http://alt.qcri.org/semeval2015/task12/",
        "datasets": ["Restaurant"],
        "description": "Aspect-Based Sentiment Analysis"
    },
    "2016": {
        "name": "SemEval-2016 Task 5",
        "url": "http://alt.qcri.org/semeval2016/task5/",
        "datasets": ["Restaurant", "Laptop"],
        "description": "Aspect-Based Sentiment Analysis"
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SemEval ABSA dataset download helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script provides download instructions for SemEval ABSA datasets.

Available datasets:
  - SemEval 2014 Task 4 (Restaurant, Laptop)
  - SemEval 2015 Task 12 (Restaurant)
  - SemEval 2016 Task 5 (Restaurant, Laptop)

After downloading, place the XML files in:
  data/raw/semeval/<dataset_name>/

Then run:
  python scripts/prepare_data.py --semeval data/raw/semeval/<dataset_name>
        """
    )

    parser.add_argument(
        "--year",
        type=str,
        choices=["2014", "2015", "2016", "all"],
        default="all",
        help="SemEval year (default: all)"
    )

    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create directory structure for downloaded data"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    return parser.parse_args()


def print_download_instructions(year: str = None):
    """
    Print download instructions.

    Args:
        year: SemEval year (or None for all)
    """
    if year and year != "all":
        years = [year]
    else:
        years = sorted(SEMEVAL_INFO.keys())

    print("\n" + "=" * 80)
    print("SemEval ABSA Dataset Download Instructions")
    print("=" * 80 + "\n")

    for year in years:
        info = SEMEVAL_INFO[year]

        print(f"\n{info['name']}")
        print("-" * 80)
        print(f"Description: {info['description']}")
        print(f"Datasets: {', '.join(info['datasets'])}")
        print(f"Official URL: {info['url']}")
        print("\nSteps:")
        print(f"  1. Visit: {info['url']}")
        print("  2. Download the dataset files (XML format)")
        print(f"  3. Extract and place in: data/raw/semeval/{year.lower()}_<dataset>/")
        print("     Example: data/raw/semeval/2014_restaurant/")
        print("  4. Run: python scripts/prepare_data.py --semeval data/raw/semeval/2014_restaurant")
        print()

    print("=" * 80)
    print("\nNote: SemEval datasets may require registration or agreement to license terms.")
    print("Please follow the official instructions on the SemEval website.")
    print("=" * 80 + "\n")


def create_directory_structure():
    """Create directory structure for SemEval data."""
    base_dir = Path("data/raw/semeval")

    dirs_to_create = [
        "2014_restaurant",
        "2014_laptop",
        "2015_restaurant",
        "2016_restaurant",
        "2016_laptop"
    ]

    for dir_name in dirs_to_create:
        dir_path = base_dir / dir_name
        ensure_dir(str(dir_path))
        logger.info(f"Created directory: {dir_path}")

        # Create README
        readme_path = dir_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"# {dir_name.upper()}\n\n")
            f.write("Place SemEval dataset files (XML format) in this directory.\n\n")
            f.write("Expected files:\n")
            f.write("- *_Train*.xml (training data)\n")
            f.write("- *_Test*.xml (test data)\n")
            f.write("- *_Dev*.xml (development data, optional)\n")

    logger.info(f"\nCreated directory structure in {base_dir}")
    logger.info("Place your downloaded SemEval XML files in the appropriate subdirectories.")


def verify_dataset(dataset_dir: str) -> bool:
    """
    Verify if dataset directory contains XML files.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        True if XML files found
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        logger.warning(f"Directory does not exist: {dataset_dir}")
        return False

    xml_files = list(dataset_path.glob("*.xml"))

    if not xml_files:
        logger.warning(f"No XML files found in {dataset_dir}")
        return False

    logger.info(f"Found {len(xml_files)} XML files in {dataset_dir}:")
    for xml_file in xml_files:
        logger.info(f"  - {xml_file.name}")

    return True


def main():
    """Main function."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Print instructions
    print_download_instructions(args.year)

    # Create directory structure if requested
    if args.create_dirs:
        create_directory_structure()

    # Check if any datasets exist
    base_dir = Path("data/raw/semeval")
    if base_dir.exists():
        print("\n" + "=" * 80)
        print("Checking existing datasets...")
        print("=" * 80 + "\n")

        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                verify_dataset(str(subdir))


if __name__ == "__main__":
    main()
