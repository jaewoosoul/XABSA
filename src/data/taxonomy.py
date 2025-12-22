"""
Category taxonomy management.
"""

import yaml
from typing import Dict


class Taxonomy:
    """Category taxonomy manager."""

    def __init__(self, taxonomy_path: str = "configs/taxonomy.yaml"):
        """
        Initialize taxonomy from YAML file.

        Args:
            taxonomy_path: Path to taxonomy YAML file
        """
        self.taxonomy_path = taxonomy_path
        self.taxonomy = self.load_taxonomy(taxonomy_path)

        # Extract categories and polarities
        self.categories = list(self.taxonomy["categories"].keys())
        self.polarities = self.taxonomy["polarities"]

        # Category to index mapping
        self.category2id = {cat: idx for idx, cat in enumerate(self.categories)}
        self.id2category = {idx: cat for cat, idx in self.category2id.items()}

        # Polarity to index mapping
        self.polarity2id = {pol: idx for idx, pol in enumerate(self.polarities)}
        self.id2polarity = {idx: pol for pol, idx in self.polarity2id.items()}

        # Category mappings
        self.category_mappings = self.taxonomy.get("category_mappings", {})

    def load_taxonomy(self, path: str) -> Dict:
        """
        Load taxonomy from YAML file.

        Args:
            path: Path to taxonomy file

        Returns:
            Taxonomy dictionary
        """
        with open(path, 'r', encoding='utf-8') as f:
            taxonomy = yaml.safe_load(f)
        return taxonomy

    def normalize_category(self, category: str) -> str:
        """
        Normalize category name.

        Args:
            category: Category name

        Returns:
            Normalized category name
        """
        # Convert to uppercase
        category = category.upper()

        # Apply mappings
        if category in self.category_mappings:
            category = self.category_mappings[category].upper()

        # If not in taxonomy, map to ETC
        if category not in self.categories:
            category = "ETC"

        return category

    def normalize_polarity(self, polarity: str) -> str:
        """
        Normalize polarity name.

        Args:
            polarity: Polarity name

        Returns:
            Normalized polarity name
        """
        polarity = polarity.lower()

        # Handle variations
        if polarity in ["pos", "positive", "긍정"]:
            return "positive"
        elif polarity in ["neg", "negative", "부정"]:
            return "negative"
        elif polarity in ["neu", "neutral", "중립"]:
            return "neutral"
        else:
            # Default to neutral if unknown
            return "neutral"

    def is_valid_category(self, category: str) -> bool:
        """
        Check if category is valid.

        Args:
            category: Category name

        Returns:
            True if valid
        """
        return category.upper() in self.categories

    def is_valid_polarity(self, polarity: str) -> bool:
        """
        Check if polarity is valid.

        Args:
            polarity: Polarity name

        Returns:
            True if valid
        """
        return polarity.lower() in self.polarities

    def get_category_id(self, category: str) -> int:
        """
        Get category ID.

        Args:
            category: Category name

        Returns:
            Category ID
        """
        category = self.normalize_category(category)
        return self.category2id[category]

    def get_polarity_id(self, polarity: str) -> int:
        """
        Get polarity ID.

        Args:
            polarity: Polarity name

        Returns:
            Polarity ID
        """
        polarity = self.normalize_polarity(polarity)
        return self.polarity2id[polarity]

    def get_num_categories(self) -> int:
        """Get number of categories."""
        return len(self.categories)

    def get_num_polarities(self) -> int:
        """Get number of polarities."""
        return len(self.polarities)

    def __repr__(self) -> str:
        return f"Taxonomy(categories={len(self.categories)}, polarities={len(self.polarities)})"
