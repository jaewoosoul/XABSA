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
        max_tokens: int = 2000,  # Increased default to prevent truncation
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
            # Note: max_output_tokens must be set high enough for complete JSON responses
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": 1.0
            }
            
            logger.debug(f"Generation config: max_output_tokens={self.max_tokens}")

            # Create model - GenerativeModel accepts model name WITHOUT models/ prefix
            # The library handles the prefix internally
            # Common working model names: "gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"
            self.client = genai.GenerativeModel(
                model_name=self.model,  # Use model name directly without prefix
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

                # Extract content - check if response is complete
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    
                    # finish_reason can be enum, string, or integer
                    # 1 = STOP, 2 = MAX_TOKENS, 3 = SAFETY, etc.
                    is_max_tokens = (
                        finish_reason == 'MAX_TOKENS' or 
                        finish_reason == 2 or 
                        str(finish_reason) == '2' or
                        (hasattr(finish_reason, 'name') and finish_reason.name == 'MAX_TOKENS')
                    )
                    
                    if is_max_tokens:
                        logger.warning(f"Response was truncated (MAX_TOKENS). Consider increasing max_tokens (current: {self.max_tokens})")
                    elif finish_reason and finish_reason != 'STOP' and finish_reason != 1:
                        logger.warning(f"Response finished with reason: {finish_reason}")
                
                content = response.text
                
                # Debug: log full response for troubleshooting
                if attempt == 0:  # Only log on first attempt to avoid spam
                    logger.debug(f"Full API response length: {len(content)} chars")
                    logger.debug(f"First 500 chars: {content[:500]}")

                # Parse triplets
                triplets = self._parse_response(content)
                logger.debug(f"Generated {len(triplets)} triplets for text: {text[:50]}...")

                return triplets

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                # Log full response on error for debugging
                if 'response' in locals() and hasattr(response, 'text'):
                    logger.debug(f"Failed response length: {len(response.text)} chars")
                    logger.debug(f"Failed response (full): {response.text}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed for text: {text[:50]}...")
                    raise

        return []
