"""
Korean data parser.

Parses Korean review data (CSV/JSON) to unified JSONL format.
"""

import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from .taxonomy import Taxonomy

logger = logging.getLogger(__name__)


class KoreanParser:
    """Parser for Korean review data."""

    def __init__(self, taxonomy: Taxonomy):
        """
        Initialize parser.

        Args:
            taxonomy: Taxonomy instance
        """
        self.taxonomy = taxonomy

    def parse_csv(
        self,
        csv_path: str,
        text_column: str = "text",
        id_column: Optional[str] = None,
        has_labels: bool = False,
        split: str = "train"
    ) -> List[Dict[str, Any]]:
        """
        Parse CSV file.

        Args:
            csv_path: Path to CSV file
            text_column: Name of text column
            id_column: Name of ID column (optional)
            has_labels: Whether CSV contains labels
            split: Data split

        Returns:
            List of samples in JSONL format
        """
        samples = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                # Get text
                text = row.get(text_column, "").strip()
                if not text:
                    continue

                # Get ID
                sample_id = row.get(id_column, f"ko_{split}_{idx}") if id_column else f"ko_{split}_{idx}"

                # Parse labels if available
                triplets = []
                if has_labels:
                    triplets = self._parse_labels_from_row(row, text)

                samples.append({
                    "id": sample_id,
                    "lang": "ko",
                    "text": text,
                    "gold_triplets": triplets,
                    "split": split
                })

        logger.info(f"Parsed {len(samples)} samples from {csv_path}")
        return samples

    def parse_json(
        self,
        json_path: str,
        split: str = "train"
    ) -> List[Dict[str, Any]]:
        """
        Parse JSON file.

        Expected format:
        [
          {
            "id": "...",
            "text": "...",
            "triplets": [{"term": "...", "category": "...", "polarity": "..."}]
          }
        ]

        Args:
            json_path: Path to JSON file
            split: Data split

        Returns:
            List of samples in JSONL format
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        for idx, item in enumerate(data):
            text = item.get("text", "").strip()
            if not text:
                continue

            # Get triplets
            triplets = []
            if "triplets" in item:
                for t in item["triplets"]:
                    triplet = {
                        "term": t.get("term", ""),
                        "category": self.taxonomy.normalize_category(
                            t.get("category", "ETC")
                        ),
                        "polarity": self.taxonomy.normalize_polarity(
                            t.get("polarity", "neutral")
                        )
                    }
                    triplets.append(triplet)

            samples.append({
                "id": item.get("id", f"ko_{split}_{idx}"),
                "lang": "ko",
                "text": text,
                "gold_triplets": triplets,
                "split": split
            })

        logger.info(f"Parsed {len(samples)} samples from {json_path}")
        return samples

    def parse_jsonl(
        self,
        jsonl_path: str,
        split: str = "train"
    ) -> List[Dict[str, Any]]:
        """
        Parse JSONL file (already in unified format).

        Args:
            jsonl_path: Path to JSONL file
            split: Data split

        Returns:
            List of samples
        """
        samples = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)

                # Validate and normalize
                text = item.get("text", "").strip()
                if not text:
                    continue

                # Normalize triplets
                triplets = []
                for t in item.get("gold_triplets", []):
                    triplet = {
                        "term": t.get("term", ""),
                        "category": self.taxonomy.normalize_category(
                            t.get("category", "ETC")
                        ),
                        "polarity": self.taxonomy.normalize_polarity(
                            t.get("polarity", "neutral")
                        )
                    }
                    triplets.append(triplet)

                samples.append({
                    "id": item.get("id", f"ko_{split}_{idx}"),
                    "lang": item.get("lang", "ko"),
                    "text": text,
                    "gold_triplets": triplets,
                    "split": item.get("split", split)
                })

        logger.info(f"Parsed {len(samples)} samples from {jsonl_path}")
        return samples

    def _parse_labels_from_row(
        self,
        row: Dict[str, str],
        text: str
    ) -> List[Dict[str, str]]:
        """
        Parse labels from CSV row.

        Expects columns like: term, category, polarity
        Or: triplets (JSON string)

        Args:
            row: CSV row
            text: Review text

        Returns:
            List of triplets
        """
        triplets = []

        # Check if triplets are in JSON format
        if "triplets" in row:
            try:
                triplets_data = json.loads(row["triplets"])
                for t in triplets_data:
                    triplet = {
                        "term": t.get("term", ""),
                        "category": self.taxonomy.normalize_category(
                            t.get("category", "ETC")
                        ),
                        "polarity": self.taxonomy.normalize_polarity(
                            t.get("polarity", "neutral")
                        )
                    }
                    triplets.append(triplet)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse triplets JSON: {row['triplets']}")

        # Otherwise, check for individual columns
        elif "term" in row and "category" in row and "polarity" in row:
            triplet = {
                "term": row["term"].strip(),
                "category": self.taxonomy.normalize_category(row["category"]),
                "polarity": self.taxonomy.normalize_polarity(row["polarity"])
            }
            triplets.append(triplet)

        return triplets

    def create_raw_samples(
        self,
        texts: List[str],
        split: str = "train"
    ) -> List[Dict[str, Any]]:
        """
        Create samples from raw text list (no labels).

        Args:
            texts: List of texts
            split: Data split

        Returns:
            List of samples
        """
        samples = []
        for idx, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue

            samples.append({
                "id": f"ko_raw_{idx}",
                "lang": "ko",
                "text": text,
                "gold_triplets": [],  # No labels
                "split": split
            })

        logger.info(f"Created {len(samples)} raw samples")
        return samples


def parse_korean_data(
    input_path: str,
    taxonomy_path: str = "configs/taxonomy.yaml",
    format: str = "auto",  # auto, csv, json, jsonl
    split: str = "train",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to parse Korean data.

    Args:
        input_path: Path to input file
        taxonomy_path: Path to taxonomy file
        format: Input format (auto-detect from extension if "auto")
        split: Data split
        **kwargs: Additional arguments for parser

    Returns:
        List of samples
    """
    taxonomy = Taxonomy(taxonomy_path)
    parser = KoreanParser(taxonomy)

    # Auto-detect format
    if format == "auto":
        ext = Path(input_path).suffix.lower()
        if ext == ".csv":
            format = "csv"
        elif ext == ".json":
            format = "json"
        elif ext == ".jsonl":
            format = "jsonl"
        else:
            raise ValueError(f"Cannot auto-detect format for extension: {ext}")

    # Parse based on format
    if format == "csv":
        return parser.parse_csv(input_path, split=split, **kwargs)
    elif format == "json":
        return parser.parse_json(input_path, split=split)
    elif format == "jsonl":
        return parser.parse_jsonl(input_path, split=split)
    else:
        raise ValueError(f"Unsupported format: {format}")
