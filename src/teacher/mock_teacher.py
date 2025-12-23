"""
Mock teacher for testing without API keys.
"""

from typing import List, Dict
import random
import logging

from .base import BaseTeacher

logger = logging.getLogger(__name__)


class MockTeacher(BaseTeacher):
    """
    Mock teacher that generates dummy triplets.

    Useful for testing the pipeline without API keys.
    """

    def __init__(
        self,
        model: str = "mock",
        **kwargs
    ):
        """Initialize mock teacher."""
        super().__init__(model=model, **kwargs)

        # Mock patterns for different languages
        self.patterns = {
            "ko": [
                {"term": "배송", "category": "DELIVERY", "polarity": "positive"},
                {"term": "가격", "category": "PRICE", "polarity": "positive"},
                {"term": "품질", "category": "QUALITY", "polarity": "positive"},
                {"term": "포장", "category": "PACKAGING", "polarity": "negative"},
                {"term": "서비스", "category": "SERVICE", "polarity": "negative"},
            ],
            "en": [
                {"term": "price", "category": "PRICE", "polarity": "positive"},
                {"term": "quality", "category": "QUALITY", "polarity": "positive"},
                {"term": "delivery", "category": "DELIVERY", "polarity": "positive"},
                {"term": "packaging", "category": "PACKAGING", "polarity": "negative"},
                {"term": "service", "category": "SERVICE", "polarity": "negative"},
            ]
        }

    def generate_triplets(
        self,
        text: str,
        lang: str = "en",
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Generate mock triplets.

        Tries to extract terms from text, falls back to patterns.

        Args:
            text: Input text
            lang: Language
            **kwargs: Additional arguments (ignored)

        Returns:
            List of triplets
        """
        patterns = self.patterns.get(lang, self.patterns["en"])
        triplets = []

        # Try to find matching terms in text
        text_lower = text.lower()

        for pattern in patterns:
            term = pattern["term"]

            # Check if term exists in text
            if term.lower() in text_lower:
                triplets.append(pattern.copy())

        # If no matches, generate random triplets
        if not triplets:
            num_triplets = random.randint(1, 3)
            triplets = random.sample(patterns, min(num_triplets, len(patterns)))

            # Try to extract actual terms from text
            words = text.split()
            if len(words) >= 2:
                for i, triplet in enumerate(triplets):
                    if i < len(words):
                        triplet["term"] = words[i]

        logger.debug(f"Generated {len(triplets)} mock triplets for text: {text[:50]}...")

        return triplets
