"""
Pseudo-label filtering module.
"""

from typing import List, Dict, Any, Set
from collections import Counter
import logging

from ..data.taxonomy import Taxonomy
from .validator import TripletValidator

logger = logging.getLogger(__name__)


class PseudoLabelFilter:
    """
    Filter for pseudo-labeled triplets.

    Implements various filtering strategies:
    1. Term existence check
    2. Deduplication
    3. Category validation
    4. Triplet count limit
    5. Self-consistency
    """

    def __init__(
        self,
        taxonomy: Taxonomy,
        check_term_existence: bool = True,
        remove_duplicates: bool = True,
        normalize_whitespace: bool = True,
        validate_category: bool = True,
        map_to_etc: bool = True,
        max_triplets_per_text: int = 8,
        use_self_consistency: bool = False,
        consistency_threshold: float = 0.6
    ):
        """
        Initialize filter.

        Args:
            taxonomy: Taxonomy instance
            check_term_existence: Check if term exists in text
            remove_duplicates: Remove duplicate triplets
            normalize_whitespace: Normalize whitespace in terms
            validate_category: Validate category against taxonomy
            map_to_etc: Map invalid categories to ETC
            max_triplets_per_text: Maximum triplets per text
            use_self_consistency: Use self-consistency filtering
            consistency_threshold: Minimum agreement ratio for self-consistency
        """
        self.taxonomy = taxonomy
        self.validator = TripletValidator(taxonomy)

        # Filter flags
        self.check_term_existence = check_term_existence
        self.remove_duplicates = remove_duplicates
        self.normalize_whitespace = normalize_whitespace
        self.validate_category = validate_category
        self.map_to_etc = map_to_etc
        self.max_triplets_per_text = max_triplets_per_text
        self.use_self_consistency = use_self_consistency
        self.consistency_threshold = consistency_threshold

        # Statistics
        self.stats = {
            "total_before": 0,
            "removed_no_term": 0,
            "removed_duplicates": 0,
            "removed_invalid_category": 0,
            "removed_too_many": 0,
            "removed_low_consistency": 0,
            "total_after": 0
        }

    def filter_sample(
        self,
        text: str,
        triplets: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Filter triplets for a single sample.

        Args:
            text: Original text
            triplets: List of triplets

        Returns:
            Filtered triplets
        """
        self.stats["total_before"] += len(triplets)

        if not triplets:
            return []

        filtered = triplets

        # Filter 1: Term existence check
        if self.check_term_existence:
            filtered = self._filter_term_existence(text, filtered)

        # Filter 2: Deduplication
        if self.remove_duplicates:
            filtered = self._filter_duplicates(filtered)

        # Filter 3: Category validation
        if self.validate_category:
            filtered = self._filter_invalid_categories(filtered)

        # Filter 4: Triplet count limit
        if self.max_triplets_per_text > 0:
            filtered = self._filter_too_many_triplets(filtered)

        # Normalize
        if self.normalize_whitespace:
            filtered = self._normalize_triplets(filtered)

        self.stats["total_after"] += len(filtered)

        return filtered

    def filter_batch(
        self,
        samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter batch of samples.

        Args:
            samples: List of samples with text and triplets

        Returns:
            Filtered samples
        """
        filtered_samples = []

        for sample in samples:
            text = sample.get("text", "")
            triplets = sample.get("gold_triplets", [])

            filtered_triplets = self.filter_sample(text, triplets)

            # Only keep samples with at least one triplet
            if filtered_triplets:
                sample_copy = sample.copy()
                sample_copy["gold_triplets"] = filtered_triplets
                filtered_samples.append(sample_copy)

        logger.info(
            f"Filtered {len(samples)} samples -> {len(filtered_samples)} samples"
        )

        return filtered_samples

    def _filter_term_existence(
        self,
        text: str,
        triplets: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Filter triplets where term doesn't exist in text.

        Args:
            text: Original text
            triplets: Triplets

        Returns:
            Filtered triplets
        """
        filtered = []
        text_lower = text.lower()

        for triplet in triplets:
            term = triplet.get("term", "").strip()

            # Allow NULL terms (implicit aspects)
            if term == "NULL" or term.lower() in text_lower:
                filtered.append(triplet)
            else:
                self.stats["removed_no_term"] += 1
                logger.debug(f"Removed triplet with non-existent term: {term}")

        return filtered

    def _filter_duplicates(
        self,
        triplets: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Remove duplicate triplets.

        Args:
            triplets: Triplets

        Returns:
            Deduplicated triplets
        """
        seen = set()
        filtered = []

        for triplet in triplets:
            # Create a normalized key
            key = (
                triplet.get("term", "").strip().lower(),
                triplet.get("category", "").upper(),
                triplet.get("polarity", "").lower()
            )

            if key not in seen:
                seen.add(key)
                filtered.append(triplet)
            else:
                self.stats["removed_duplicates"] += 1

        return filtered

    def _filter_invalid_categories(
        self,
        triplets: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Filter or fix invalid categories.

        Args:
            triplets: Triplets

        Returns:
            Filtered triplets
        """
        filtered = []

        for triplet in triplets:
            category = triplet.get("category", "")

            if self.taxonomy.is_valid_category(category):
                filtered.append(triplet)
            elif self.map_to_etc:
                # Map to ETC
                triplet["category"] = "ETC"
                filtered.append(triplet)
                logger.debug(f"Mapped invalid category '{category}' to ETC")
            else:
                # Remove
                self.stats["removed_invalid_category"] += 1
                logger.debug(f"Removed triplet with invalid category: {category}")

        return filtered

    def _filter_too_many_triplets(
        self,
        triplets: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Limit number of triplets per text.

        Args:
            triplets: Triplets

        Returns:
            Limited triplets
        """
        if len(triplets) <= self.max_triplets_per_text:
            return triplets

        # Keep first N triplets (or use some other selection strategy)
        filtered = triplets[:self.max_triplets_per_text]

        removed_count = len(triplets) - len(filtered)
        self.stats["removed_too_many"] += removed_count
        logger.debug(
            f"Removed {removed_count} triplets (exceeded max {self.max_triplets_per_text})"
        )

        return filtered

    def _normalize_triplets(
        self,
        triplets: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Normalize triplets (whitespace, case, etc.).

        Args:
            triplets: Triplets

        Returns:
            Normalized triplets
        """
        normalized = []

        for triplet in triplets:
            normalized_triplet = {
                "term": triplet.get("term", "").strip(),
                "category": self.taxonomy.normalize_category(
                    triplet.get("category", "ETC")
                ),
                "polarity": self.taxonomy.normalize_polarity(
                    triplet.get("polarity", "neutral")
                )
            }
            normalized.append(normalized_triplet)

        return normalized

    def apply_self_consistency(
        self,
        text: str,
        multiple_triplets: List[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Apply self-consistency filtering.

        Given multiple generations for the same text,
        only keep triplets that appear in at least threshold% of generations.

        Args:
            text: Original text
            multiple_triplets: List of triplet lists from multiple generations

        Returns:
            Consensus triplets
        """
        if not multiple_triplets:
            return []

        num_generations = len(multiple_triplets)

        # Count occurrences of each triplet
        triplet_counts = Counter()

        for triplets in multiple_triplets:
            # Normalize first
            triplets = self._normalize_triplets(triplets)

            for triplet in triplets:
                # Create key
                key = (
                    triplet["term"],
                    triplet["category"],
                    triplet["polarity"]
                )
                triplet_counts[key] += 1

        # Filter by threshold
        min_count = int(num_generations * self.consistency_threshold)
        consensus_triplets = []

        for key, count in triplet_counts.items():
            if count >= min_count:
                consensus_triplets.append({
                    "term": key[0],
                    "category": key[1],
                    "polarity": key[2]
                })
            else:
                self.stats["removed_low_consistency"] += 1

        logger.debug(
            f"Self-consistency: {len(triplet_counts)} unique -> "
            f"{len(consensus_triplets)} consensus (threshold={self.consistency_threshold})"
        )

        return consensus_triplets

    def get_stats(self) -> Dict[str, int]:
        """Get filtering statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        for key in self.stats:
            self.stats[key] = 0

    def get_summary(self) -> Dict[str, Any]:
        """
        Get filtering summary.

        Returns:
            Summary statistics
        """
        total_before = self.stats["total_before"]
        total_after = self.stats["total_after"]

        if total_before == 0:
            retention_rate = 0.0
        else:
            retention_rate = total_after / total_before

        return {
            "total_before": total_before,
            "total_after": total_after,
            "removed": total_before - total_after,
            "retention_rate": retention_rate,
            "breakdown": {
                "removed_no_term": self.stats["removed_no_term"],
                "removed_duplicates": self.stats["removed_duplicates"],
                "removed_invalid_category": self.stats["removed_invalid_category"],
                "removed_too_many": self.stats["removed_too_many"],
                "removed_low_consistency": self.stats["removed_low_consistency"]
            }
        }
