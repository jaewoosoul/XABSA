"""
Base Teacher interface for LLM-based pseudo-label generation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseTeacher(ABC):
    """
    Abstract base class for LLM Teachers.

    Subclasses should implement the generate_triplets method.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize teacher.

        Args:
            model: Model name/identifier
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            max_retries: Maximum retry attempts
            **kwargs: Additional model-specific arguments
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.kwargs = kwargs

    @abstractmethod
    def generate_triplets(
        self,
        text: str,
        lang: str = "en",
        prompt_template: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate triplets for given text.

        Args:
            text: Input text
            lang: Language (en, ko)
            prompt_template: Optional custom prompt template

        Returns:
            List of triplets [{"term": "...", "category": "...", "polarity": "..."}]
        """
        pass

    def batch_generate(
        self,
        texts: List[str],
        lang: str = "en",
        show_progress: bool = True
    ) -> List[List[Dict[str, str]]]:
        """
        Generate triplets for batch of texts.

        Args:
            texts: List of input texts
            lang: Language
            show_progress: Whether to show progress bar

        Returns:
            List of triplet lists
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(texts, desc="Generating pseudo-labels") if show_progress else texts

        for text in iterator:
            try:
                triplets = self.generate_triplets(text, lang=lang)
                results.append(triplets)
            except Exception as e:
                logger.error(f"Error generating triplets for text: {text[:50]}... Error: {e}")
                results.append([])  # Empty list on failure

        return results

    def _parse_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse LLM response to extract triplets.

        Args:
            response: LLM response string

        Returns:
            List of triplets
        """
        import json
        import re

        # Try to extract JSON from response
        # Look for JSON object or array
        json_match = re.search(r'\{.*\}|\[.*\]', response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())

                # Handle different response formats
                if isinstance(data, dict):
                    if "triplets" in data:
                        return data["triplets"]
                    elif "aspects" in data:
                        return data["aspects"]
                elif isinstance(data, list):
                    return data

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                raise

        raise ValueError(f"Could not parse triplets from response: {response[:100]}...")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
