"""
Validator for generated triplets.
"""

from typing import List, Dict, Any, Optional
import logging

from ..data.taxonomy import Taxonomy

logger = logging.getLogger(__name__)


class TripletValidator:
    """Validator for triplet schema and content."""

    def __init__(self, taxonomy: Taxonomy):
        """
        Initialize validator.

        Args:
            taxonomy: Taxonomy instance
        """
        self.taxonomy = taxonomy

    def validate_triplet(
        self,
        triplet: Dict[str, str],
        text: Optional[str] = None,
        strict: bool = False
    ) -> bool:
        """
        Validate single triplet.

        Args:
            triplet: Triplet dictionary
            text: Original text (for term existence check)
            strict: If True, invalid triplets raise ValueError

        Returns:
            True if valid
        """
        # Check required fields
        required_fields = ["term", "category", "polarity"]
        for field in required_fields:
            if field not in triplet:
                if strict:
                    raise ValueError(f"Missing required field: {field}")
                logger.warning(f"Triplet missing field '{field}': {triplet}")
                return False

        # Validate term
        term = triplet["term"]
        if not isinstance(term, str) or not term.strip():
            if strict:
                raise ValueError(f"Invalid term: {term}")
            logger.warning(f"Invalid term in triplet: {triplet}")
            return False

        # Check term exists in text
        if text is not None:
            if term.lower() not in text.lower() and term != "NULL":
                if strict:
                    raise ValueError(f"Term '{term}' not found in text: {text}")
                logger.warning(f"Term '{term}' not found in text: {text[:100]}...")
                return False

        # Validate category
        category = triplet["category"]
        if not self.taxonomy.is_valid_category(category):
            # Try to normalize
            try:
                normalized_category = self.taxonomy.normalize_category(category)
                triplet["category"] = normalized_category
                logger.debug(f"Normalized category: {category} -> {normalized_category}")
            except Exception as e:
                if strict:
                    raise ValueError(f"Invalid category: {category}")
                logger.warning(f"Invalid category in triplet: {category}")
                return False

        # Validate polarity
        polarity = triplet["polarity"]
        if not self.taxonomy.is_valid_polarity(polarity):
            # Try to normalize
            try:
                normalized_polarity = self.taxonomy.normalize_polarity(polarity)
                triplet["polarity"] = normalized_polarity
                logger.debug(f"Normalized polarity: {polarity} -> {normalized_polarity}")
            except Exception as e:
                if strict:
                    raise ValueError(f"Invalid polarity: {polarity}")
                logger.warning(f"Invalid polarity in triplet: {polarity}")
                return False

        return True

    def validate_triplets(
        self,
        triplets: List[Dict[str, str]],
        text: Optional[str] = None,
        strict: bool = False,
        remove_invalid: bool = True
    ) -> List[Dict[str, str]]:
        """
        Validate list of triplets.

        Args:
            triplets: List of triplets
            text: Original text
            strict: If True, invalid triplets raise ValueError
            remove_invalid: If True, remove invalid triplets from list

        Returns:
            Validated triplets (with invalid ones removed if remove_invalid=True)
        """
        valid_triplets = []

        for triplet in triplets:
            if self.validate_triplet(triplet, text=text, strict=strict):
                valid_triplets.append(triplet)
            elif not remove_invalid:
                valid_triplets.append(triplet)

        if remove_invalid and len(valid_triplets) < len(triplets):
            logger.info(
                f"Removed {len(triplets) - len(valid_triplets)} invalid triplets"
            )

        return valid_triplets

    def validate_schema(self, data: Any) -> bool:
        """
        Validate data schema.

        Args:
            data: Data to validate

        Returns:
            True if valid
        """
        # Check if data is a list
        if not isinstance(data, list):
            logger.warning(f"Expected list, got {type(data)}")
            return False

        # Check each item
        for item in data:
            if not isinstance(item, dict):
                logger.warning(f"Expected dict, got {type(item)}")
                return False

            required_fields = ["term", "category", "polarity"]
            for field in required_fields:
                if field not in item:
                    logger.warning(f"Missing field '{field}' in item: {item}")
                    return False

        return True
