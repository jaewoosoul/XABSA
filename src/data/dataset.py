"""
PyTorch Dataset for XABSA with improved labeling.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
import logging
import re

from .taxonomy import Taxonomy

logger = logging.getLogger(__name__)


class XABSADataset(Dataset):
    """
    Dataset for XABSA task.

    Handles tokenization and label alignment for triplet extraction.
    Supports multiple term occurrences, term pooling, and robust matching.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        taxonomy: Taxonomy,
        max_length: int = 128,
        task_type: str = "joint",  # joint or pipeline
        term_pooling: str = "mean",  # mean or start
        match_normalize_whitespace: bool = True,
        match_all_occurrences: bool = False
    ):
        """
        Initialize dataset.

        Args:
            data: List of samples in JSONL format
            tokenizer: HuggingFace tokenizer
            taxonomy: Taxonomy instance
            max_length: Maximum sequence length
            task_type: Task type (joint or pipeline)
            term_pooling: Term pooling strategy (mean or start)
            match_normalize_whitespace: Normalize whitespace for matching
            match_all_occurrences: Match all term occurrences or just first
        """
        self.data = data
        self.tokenizer = tokenizer
        self.taxonomy = taxonomy
        self.max_length = max_length
        self.task_type = task_type
        self.term_pooling = term_pooling
        self.match_normalize_whitespace = match_normalize_whitespace
        self.match_all_occurrences = match_all_occurrences

        # BIO tag mapping
        self.bio2id = {"O": 0, "B": 1, "I": 2}
        self.id2bio = {v: k for k, v in self.bio2id.items()}

        # Statistics
        self.skipped_terms = 0
        self.total_terms = 0

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Dictionary containing input_ids, attention_mask, labels
        """
        sample = self.data[idx]

        text = sample["text"]
        triplets = sample.get("gold_triplets", [])

        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        # Create labels
        if self.task_type == "joint":
            labels = self._create_joint_labels(
                text, triplets, offset_mapping
            )
        else:
            labels = self._create_pipeline_labels(
                text, triplets, offset_mapping
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            **labels,
            "text": text,
            "triplets": triplets,
            "sample_id": sample.get("id", f"sample_{idx}"),
            "lang": sample.get("lang", "unknown")
        }

    def _normalize_for_matching(self, text: str) -> str:
        """
        Normalize text for matching.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        if self.match_normalize_whitespace:
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        return text

    def _find_term_spans(self, text: str, term: str) -> List[Tuple[int, int]]:
        """
        Find all occurrences of term in text.

        Args:
            text: Input text
            term: Term to find

        Returns:
            List of (start, end) tuples
        """
        spans = []

        # Normalize if needed
        if self.match_normalize_whitespace:
            text_norm = self._normalize_for_matching(text)
            term_norm = self._normalize_for_matching(term)
        else:
            text_norm = text
            term_norm = term

        # Find all occurrences
        start = 0
        while True:
            idx = text_norm.find(term_norm, start)
            if idx == -1:
                break

            # Map back to original text positions
            # (In normalized text, positions should match)
            spans.append((idx, idx + len(term_norm)))

            if not self.match_all_occurrences:
                break

            start = idx + 1

        return spans

    def _align_span_to_tokens(
        self,
        span_start: int,
        span_end: int,
        offset_mapping: torch.Tensor
    ) -> List[int]:
        """
        Align character span to token indices.

        Args:
            span_start: Character start position
            span_end: Character end position
            offset_mapping: Token offset mapping

        Returns:
            List of token indices
        """
        token_indices = []

        for token_idx, (start, end) in enumerate(offset_mapping):
            start, end = start.item(), end.item()

            # Skip special tokens
            if start == 0 and end == 0:
                continue

            # Check if token overlaps with span
            if start >= span_start and end <= span_end:
                token_indices.append(token_idx)
            elif start < span_end and end > span_start:
                # Partial overlap - include token
                token_indices.append(token_idx)

        return token_indices

    def _create_joint_labels(
        self,
        text: str,
        triplets: List[Dict],
        offset_mapping: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Create labels for joint model.

        Handles multiple triplets with proper BIO tagging.

        Args:
            text: Input text
            triplets: Gold triplets
            offset_mapping: Token offset mapping

        Returns:
            Dictionary of labels
        """
        seq_len = len(offset_mapping)

        # Initialize BIO tags (for ATE)
        bio_tags = torch.zeros(seq_len, dtype=torch.long)  # Default: O

        # Initialize term-level labels
        # We'll collect all terms and their labels
        term_spans = []  # List of (token_indices, category_id, polarity_id)

        # Process each triplet
        for triplet in triplets:
            self.total_terms += 1

            term = triplet["term"]
            category = triplet["category"]
            polarity = triplet["polarity"]

            # Find term spans in text
            char_spans = self._find_term_spans(text, term)

            if not char_spans:
                # Term not found in text
                self.skipped_terms += 1
                logger.debug(f"Term not found in text: '{term}' in '{text[:50]}...'")
                continue

            # Process each occurrence
            for span_start, span_end in char_spans:
                # Align to tokens
                token_indices = self._align_span_to_tokens(
                    span_start, span_end, offset_mapping
                )

                if not token_indices:
                    continue

                # Mark BIO tags
                for i, token_idx in enumerate(token_indices):
                    if i == 0:
                        bio_tags[token_idx] = self.bio2id["B"]
                    else:
                        bio_tags[token_idx] = self.bio2id["I"]

                # Store term info
                category_id = self.taxonomy.get_category_id(category)
                polarity_id = self.taxonomy.get_polarity_id(polarity)
                term_spans.append((token_indices, category_id, polarity_id))

        # Create term-level labels
        # For joint model, we need sentence-level category and polarity
        # Strategy: Use the first valid triplet's labels
        if term_spans:
            _, category_label, polarity_label = term_spans[0]
        else:
            category_label = -100  # Ignore index
            polarity_label = -100

        # Store term span info for later use (e.g., term pooling)
        # We'll store a list of term token indices
        term_masks = []
        term_category_labels = []
        term_polarity_labels = []

        for token_indices, cat_id, pol_id in term_spans:
            # Create a mask tensor for this term
            mask = torch.zeros(seq_len, dtype=torch.bool)
            mask[token_indices] = True
            term_masks.append(mask)
            term_category_labels.append(cat_id)
            term_polarity_labels.append(pol_id)

        # Pad to max number of terms (e.g., 8)
        max_terms = 8
        while len(term_masks) < max_terms:
            term_masks.append(torch.zeros(seq_len, dtype=torch.bool))
            term_category_labels.append(-100)
            term_polarity_labels.append(-100)

        # Truncate if too many
        term_masks = term_masks[:max_terms]
        term_category_labels = term_category_labels[:max_terms]
        term_polarity_labels = term_polarity_labels[:max_terms]

        return {
            "bio_labels": bio_tags,
            "category_labels": torch.tensor(category_label, dtype=torch.long),
            "polarity_labels": torch.tensor(polarity_label, dtype=torch.long),
            # Additional term-level info
            "term_masks": torch.stack(term_masks),  # [max_terms, seq_len]
            "term_category_labels": torch.tensor(term_category_labels, dtype=torch.long),
            "term_polarity_labels": torch.tensor(term_polarity_labels, dtype=torch.long),
            "num_terms": torch.tensor(len(term_spans), dtype=torch.long)
        }

    def _create_pipeline_labels(
        self,
        text: str,
        triplets: List[Dict],
        offset_mapping: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Create labels for pipeline model.

        Similar to joint labels but structured for pipeline.

        Args:
            text: Input text
            triplets: Gold triplets
            offset_mapping: Token offset mapping

        Returns:
            Dictionary of labels
        """
        # For pipeline, we use the same structure as joint
        # The difference is in how the model processes them
        return self._create_joint_labels(text, triplets, offset_mapping)

    def get_stats(self) -> Dict[str, int]:
        """
        Get dataset statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "total_samples": len(self.data),
            "total_terms": self.total_terms,
            "skipped_terms": self.skipped_terms,
            "matched_terms": self.total_terms - self.skipped_terms
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.

        Args:
            batch: List of samples

        Returns:
            Batched tensors
        """
        keys = ["input_ids", "attention_mask", "bio_labels",
                "category_labels", "polarity_labels",
                "term_masks", "term_category_labels", "term_polarity_labels",
                "num_terms", "offset_mapping"]

        batched = {}
        for key in keys:
            if key in batch[0]:
                batched[key] = torch.stack([item[key] for item in batch])

        # Keep text and other metadata as lists
        batched["texts"] = [item["text"] for item in batch]
        batched["triplets"] = [item["triplets"] for item in batch]
        batched["sample_ids"] = [item["sample_id"] for item in batch]
        batched["langs"] = [item["lang"] for item in batch]

        return batched
