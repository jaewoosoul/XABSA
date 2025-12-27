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

        # Remove markdown code blocks if present
        # Handle ```json ... ``` or ``` ... ```
        response = re.sub(r'```json\s*\n?', '', response)
        response = re.sub(r'```\s*\n?', '', response)
        response = response.strip()

        # Try multiple parsing strategies
        # Strategy 1: Try to parse the entire response as JSON
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                if "triplets" in data:
                    return data["triplets"]
                elif "aspects" in data:
                    return data["aspects"]
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON object using balanced braces
        # Find the first { and match until the closing }
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(response):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    json_str = response[start_idx:i+1]
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, dict):
                            if "triplets" in data:
                                return data["triplets"]
                            elif "aspects" in data:
                                return data["aspects"]
                        elif isinstance(data, list):
                            return data
                    except json.JSONDecodeError as e:
                        logger.debug(f"Balanced braces JSON decode error: {e}")
                        logger.debug(f"Attempted JSON: {json_str[:300]}")
                    break

        # Strategy 3: Use regex to find JSON (fallback)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, dict):
                    if "triplets" in data:
                        return data["triplets"]
                    elif "aspects" in data:
                        return data["aspects"]
                elif isinstance(data, list):
                    return data
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode error: {e}")
                logger.debug(f"Attempted to parse: {json_match.group()[:200]}")

        # If all strategies fail, log the full response for debugging
        logger.error(f"Could not parse triplets from response. Full response: {response[:500]}")
        raise ValueError(f"Could not parse triplets from response: {response[:100]}...")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
