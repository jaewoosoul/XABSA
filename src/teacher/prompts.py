"""
Prompt templates for LLM Teachers.
"""

from typing import Dict, List


# Category list for prompts
CATEGORIES = [
    "PRICE", "QUALITY", "DELIVERY", "SERVICE", "DESIGN",
    "PERFORMANCE", "DURABILITY", "USABILITY", "PACKAGING",
    "SIZE", "RETURN", "VALUE", "ETC"
]

POLARITIES = ["positive", "negative", "neutral"]


# English prompt template
ENGLISH_PROMPT_TEMPLATE = """You are an expert in aspect-based sentiment analysis.

Given a review text, extract ALL aspect-based sentiment triplets (term, category, polarity).

**Categories**: {categories}
**Polarities**: {polarities}

**Important rules**:
1. The "term" MUST be an exact substring from the review text
2. Extract ALL aspects mentioned in the review
3. If a term is not explicitly mentioned, use the most relevant term from the text
4. Category should be one of the predefined categories
5. Polarity should be positive, negative, or neutral

**Review**: {text}

**Output format** (JSON only, no explanation):
{{
  "triplets": [
    {{"term": "exact term from text", "category": "CATEGORY", "polarity": "positive/negative/neutral"}}
  ]
}}

Output:"""


# Korean prompt template
KOREAN_PROMPT_TEMPLATE = """당신은 aspect-based sentiment analysis 전문가입니다.

주어진 리뷰 텍스트에서 모든 aspect 기반 감정 triplet(용어, 카테고리, 극성)을 추출하세요.

**카테고리**: {categories}
**극성**: {polarities}

**중요한 규칙**:
1. "term"은 반드시 리뷰 텍스트에서 정확히 나온 부분 문자열이어야 합니다
2. 리뷰에 언급된 모든 aspect를 추출하세요
3. 용어가 명시적으로 언급되지 않은 경우, 텍스트에서 가장 관련성 높은 용어를 사용하세요
4. 카테고리는 미리 정의된 카테고리 중 하나여야 합니다
5. 극성은 positive, negative, neutral 중 하나여야 합니다

**리뷰**: {text}

**출력 형식** (설명 없이 JSON만):
{{
  "triplets": [
    {{"term": "텍스트에서 정확한 용어", "category": "CATEGORY", "polarity": "positive/negative/neutral"}}
  ]
}}

출력:"""


def get_prompt_template(lang: str = "en") -> str:
    """
    Get prompt template for language.

    Args:
        lang: Language code (en, ko)

    Returns:
        Prompt template string
    """
    if lang == "ko":
        return KOREAN_PROMPT_TEMPLATE
    else:
        return ENGLISH_PROMPT_TEMPLATE


def format_prompt(
    text: str,
    lang: str = "en",
    categories: List[str] = None,
    polarities: List[str] = None,
    template: str = None
) -> str:
    """
    Format prompt with text and categories.

    Args:
        text: Review text
        lang: Language
        categories: List of categories (optional)
        polarities: List of polarities (optional)
        template: Custom template (optional)

    Returns:
        Formatted prompt
    """
    if template is None:
        template = get_prompt_template(lang)

    if categories is None:
        categories = CATEGORIES

    if polarities is None:
        polarities = POLARITIES

    return template.format(
        text=text,
        categories=", ".join(categories),
        polarities=", ".join(polarities)
    )


# Few-shot examples
FEW_SHOT_EXAMPLES = {
    "en": [
        {
            "text": "The food was excellent but the service was slow.",
            "triplets": [
                {"term": "food", "category": "QUALITY", "polarity": "positive"},
                {"term": "service", "category": "SERVICE", "polarity": "negative"}
            ]
        },
        {
            "text": "Great price for such good quality!",
            "triplets": [
                {"term": "price", "category": "PRICE", "polarity": "positive"},
                {"term": "quality", "category": "QUALITY", "polarity": "positive"}
            ]
        }
    ],
    "ko": [
        {
            "text": "배송은 빨랐는데 포장이 너무 부실했어요.",
            "triplets": [
                {"term": "배송", "category": "DELIVERY", "polarity": "positive"},
                {"term": "포장", "category": "PACKAGING", "polarity": "negative"}
            ]
        },
        {
            "text": "가격대비 품질이 훌륭합니다!",
            "triplets": [
                {"term": "가격", "category": "VALUE", "polarity": "positive"},
                {"term": "품질", "category": "QUALITY", "polarity": "positive"}
            ]
        }
    ]
}


def get_few_shot_prompt(
    text: str,
    lang: str = "en",
    num_examples: int = 2
) -> str:
    """
    Create few-shot prompt with examples.

    Args:
        text: Review text
        lang: Language
        num_examples: Number of examples to include

    Returns:
        Few-shot prompt
    """
    import json

    examples = FEW_SHOT_EXAMPLES.get(lang, FEW_SHOT_EXAMPLES["en"])[:num_examples]

    prompt_parts = []

    # Add instruction
    if lang == "ko":
        prompt_parts.append("다음 예시를 참고하여 리뷰에서 triplet을 추출하세요.\n")
    else:
        prompt_parts.append("Extract triplets from the review following these examples.\n")

    # Add examples
    for i, example in enumerate(examples, 1):
        prompt_parts.append(f"\nExample {i}:")
        prompt_parts.append(f"Review: {example['text']}")
        prompt_parts.append(f"Output: {json.dumps({'triplets': example['triplets']}, ensure_ascii=False)}\n")

    # Add actual query
    if lang == "ko":
        prompt_parts.append(f"\n이제 다음 리뷰를 분석하세요:")
    else:
        prompt_parts.append(f"\nNow analyze this review:")

    prompt_parts.append(f"Review: {text}")
    prompt_parts.append("Output:")

    return "\n".join(prompt_parts)
