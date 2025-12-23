"""
Claude-based teacher for pseudo-label generation.
"""

import os
import time
from typing import List, Dict, Optional
import logging

from .base import BaseTeacher
from .prompts import format_prompt, get_few_shot_prompt

logger = logging.getLogger(__name__)


class ClaudeTeacher(BaseTeacher):
    """Teacher using Claude API (Anthropic)."""

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize Claude teacher.

        Args:
            model: Claude model name
            api_key: API key (or set ANTHROPIC_API_KEY env var)
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
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key argument."
            )

        # Initialize client
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

    def generate_triplets(
        self,
        text: str,
        lang: str = "en",
        prompt_template: Optional[str] = None,
        use_few_shot: bool = False
    ) -> List[Dict[str, str]]:
        """
        Generate triplets using Claude API.

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
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                # Extract content
                content = response.content[0].text

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
