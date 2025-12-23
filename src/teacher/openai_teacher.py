"""
OpenAI-based teacher for pseudo-label generation.
"""

import os
import time
from typing import List, Dict, Optional
import logging

from .base import BaseTeacher
from .prompts import format_prompt, get_few_shot_prompt

logger = logging.getLogger(__name__)


class OpenAITeacher(BaseTeacher):
    """Teacher using OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize OpenAI teacher.

        Args:
            model: OpenAI model name
            api_key: API key (or set OPENAI_API_KEY env var)
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

        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass api_key argument."
            )

        # Initialize client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    def generate_triplets(
        self,
        text: str,
        lang: str = "en",
        prompt_template: Optional[str] = None,
        use_few_shot: bool = False
    ) -> List[Dict[str, str]]:
        """
        Generate triplets using OpenAI API.

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
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in aspect-based sentiment analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}  # Force JSON output
                )

                # Extract content
                content = response.choices[0].message.content

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
