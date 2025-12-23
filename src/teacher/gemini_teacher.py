"""
Google Gemini-based teacher for pseudo-label generation.
"""

import os
import time
from typing import List, Dict, Optional
import logging

from .base import BaseTeacher
from .prompts import format_prompt, get_few_shot_prompt

logger = logging.getLogger(__name__)


class GeminiTeacher(BaseTeacher):
    """Teacher using Google Gemini API."""

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize Gemini teacher.

        Args:
            model: Gemini model name (gemini-pro, gemini-1.5-pro, gemini-1.5-flash)
            api_key: API key (or set GOOGLE_API_KEY or GEMINI_API_KEY env var)
            temperature: Generation temperature
            max_tokens: Maximum tokens
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            **kwargs: Additional arguments
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            **kwargs
        )

        self.retry_delay = retry_delay

        # Get API key (try both GOOGLE_API_KEY and GEMINI_API_KEY)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not provided. "
                "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable or pass api_key argument."
            )

        # Initialize Gemini client
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)

            # Create model configuration (without JSON mode for compatibility)
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens
            }

            # Add "models/" prefix if not present
            model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"

            self.client = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config
            )

        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

    def generate_triplets(
        self,
        text: str,
        lang: str = "en",
        prompt_template: Optional[str] = None,
        use_few_shot: bool = False
    ) -> List[Dict[str, str]]:
        """
        Generate triplets using Gemini API.

        Args:
            text: Input text
            lang: Language
            prompt_template: Custom prompt template
            use_few_shot: Whether to use few-shot prompting

        Returns:
            List of triplets
        """
        # Format prompt
        if use_few_shot:
            prompt = get_few_shot_prompt(text, lang=lang)
        else:
            prompt = format_prompt(
                text, lang=lang, template=prompt_template
            )

        # Generate with retries
        for attempt in range(self.max_retries):
            try:
                # Generate content
                response = self.client.generate_content(prompt)

                # Extract content
                content = response.text

                # Parse triplets
                triplets = self._parse_response(content)
                logger.debug(f"Generated {len(triplets)} triplets for text: {text[:50]}...")

                return triplets

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed for text: {text[:50]}...")
                    raise

        return []
