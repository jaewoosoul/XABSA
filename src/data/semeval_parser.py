"""
SemEval ABSA dataset parser.

Parses SemEval XML format to unified JSONL format.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from .taxonomy import Taxonomy

logger = logging.getLogger(__name__)


class SemEvalParser:
    """Parser for SemEval ABSA datasets."""

    def __init__(self, taxonomy: Taxonomy):
        """
        Initialize parser.

        Args:
            taxonomy: Taxonomy instance for category normalization
        """
        self.taxonomy = taxonomy

    def parse_xml_file(
        self,
        xml_path: str,
        lang: str = "en",
        split: str = "train"
    ) -> List[Dict[str, Any]]:
        """
        Parse SemEval XML file.

        Args:
            xml_path: Path to XML file
            lang: Language (en, es, etc.)
            split: Data split (train, dev, test)

        Returns:
            List of samples in JSONL format
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        samples = []

        # SemEval format: <sentences><sentence><text>...</text><aspectTerms>...</aspectTerms></sentence></sentences>
        for sentence in root.findall(".//sentence"):
            sample = self._parse_sentence(sentence, lang, split)
            if sample:
                samples.append(sample)

        logger.info(f"Parsed {len(samples)} samples from {xml_path}")
        return samples

    def _parse_sentence(
        self,
        sentence_elem: ET.Element,
        lang: str,
        split: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse single sentence element.

        Args:
            sentence_elem: XML sentence element
            lang: Language
            split: Data split

        Returns:
            Sample dictionary or None
        """
        # Get sentence ID
        sent_id = sentence_elem.get("id", "unknown")

        # Get text
        text_elem = sentence_elem.find("text")
        if text_elem is None or text_elem.text is None:
            logger.warning(f"No text found for sentence {sent_id}")
            return None

        text = text_elem.text.strip()

        # Parse aspect terms
        triplets = []
        aspect_terms_elem = sentence_elem.find("aspectTerms")

        if aspect_terms_elem is not None:
            for aspect_term in aspect_terms_elem.findall("aspectTerm"):
                triplet = self._parse_aspect_term(aspect_term, text)
                if triplet:
                    triplets.append(triplet)

        # Parse aspect categories (if available)
        aspect_categories_elem = sentence_elem.find("aspectCategories")
        if aspect_categories_elem is not None and not triplets:
            # If no aspect terms but has categories, create implicit triplets
            for aspect_category in aspect_categories_elem.findall("aspectCategory"):
                triplet = self._parse_aspect_category(aspect_category)
                if triplet:
                    # Use "NULL" as term for implicit aspects
                    triplet["term"] = "NULL"
                    triplets.append(triplet)

        return {
            "id": f"semeval_{lang}_{split}_{sent_id}",
            "lang": lang,
            "text": text,
            "gold_triplets": triplets,
            "split": split
        }

    def _parse_aspect_term(
        self,
        aspect_term_elem: ET.Element,
        text: str
    ) -> Optional[Dict[str, str]]:
        """
        Parse aspect term element.

        Args:
            aspect_term_elem: XML aspect term element
            text: Sentence text

        Returns:
            Triplet dictionary or None
        """
        term = aspect_term_elem.get("term", "").strip()
        polarity = aspect_term_elem.get("polarity", "neutral").strip()
        category = aspect_term_elem.get("category", "ETC").strip()

        # Validate term exists in text
        if term and term not in text:
            logger.warning(f"Term '{term}' not found in text: {text}")
            return None

        # Normalize category and polarity
        category = self.taxonomy.normalize_category(category)
        polarity = self.taxonomy.normalize_polarity(polarity)

        return {
            "term": term,
            "category": category,
            "polarity": polarity
        }

    def _parse_aspect_category(
        self,
        aspect_category_elem: ET.Element
    ) -> Optional[Dict[str, str]]:
        """
        Parse aspect category element (for implicit aspects).

        Args:
            aspect_category_elem: XML aspect category element

        Returns:
            Triplet dictionary or None
        """
        category = aspect_category_elem.get("category", "ETC").strip()
        polarity = aspect_category_elem.get("polarity", "neutral").strip()

        # Normalize
        category = self.taxonomy.normalize_category(category)
        polarity = self.taxonomy.normalize_polarity(polarity)

        return {
            "category": category,
            "polarity": polarity
        }

    def parse_directory(
        self,
        data_dir: str,
        lang: str = "en"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse all XML files in directory.

        Args:
            data_dir: Directory containing XML files
            lang: Language

        Returns:
            Dictionary mapping split to samples
        """
        data_dir = Path(data_dir)
        splits = {}

        # Common filename patterns
        patterns = {
            "train": ["train.xml", "*_train.xml", "Restaurants_Train*.xml", "Laptop*Train*.xml"],
            "dev": ["dev.xml", "*_dev.xml"],
            "test": ["test.xml", "*_test.xml", "Restaurants_Test*.xml", "Laptop*Test*.xml"]
        }

        for split, file_patterns in patterns.items():
            for pattern in file_patterns:
                xml_files = list(data_dir.glob(pattern))
                if xml_files:
                    # Use first matching file
                    xml_file = xml_files[0]
                    logger.info(f"Parsing {split} from {xml_file}")
                    splits[split] = self.parse_xml_file(
                        str(xml_file), lang=lang, split=split
                    )
                    break

        return splits


def parse_semeval_dataset(
    data_dir: str,
    taxonomy_path: str = "configs/taxonomy.yaml",
    lang: str = "en"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to parse SemEval dataset.

    Args:
        data_dir: Directory containing SemEval XML files
        taxonomy_path: Path to taxonomy file
        lang: Language

    Returns:
        Dictionary mapping split to samples
    """
    taxonomy = Taxonomy(taxonomy_path)
    parser = SemEvalParser(taxonomy)
    return parser.parse_directory(data_dir, lang=lang)
