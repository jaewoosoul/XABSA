"""
PyTorch Dataset for XABSA.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
import logging

from .taxonomy import Taxonomy

logger = logging.getLogger(__name__)


class XABSADataset(Dataset):
    """
    Dataset for XABSA task.

    Handles tokenization and label alignment for triplet extraction.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        taxonomy: Taxonomy,
        max_length: int = 128,
        task_type: str = "joint"  # joint or pipeline
    ):
        """
        Initialize dataset.

        Args:
            data: List of samples in JSONL format
            tokenizer: HuggingFace tokenizer
            taxonomy: Taxonomy instance
            max_length: Maximum sequence length
            task_type: Task type (joint or pipeline)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.taxonomy = taxonomy
        self.max_length = max_length
        self.task_type = task_type

        # BIO tag mapping
        self.bio2id = {"O": 0, "B": 1, "I": 2}
        self.id2bio = {v: k for k, v in self.bio2id.items()}

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

        # Tokenize
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

    def _create_joint_labels(
        self,
        text: str,
        triplets: List[Dict],
        offset_mapping: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Create labels for joint model.

        Args:
            text: Input text
            triplets: Gold triplets
            offset_mapping: Token offset mapping

        Returns:
            Dictionary of labels
        """
        seq_len = len(offset_mapping)

        # Initialize BIO tags (for ATE)
        bio_tags = torch.zeros(seq_len, dtype=torch.long)

        # Initialize category and polarity labels
        # For simplicity, we use the first triplet's category/polarity
        # In a full implementation, you might want to handle multiple triplets
        category_label = -100  # Ignore index
        polarity_label = -100

        if triplets:
            # Process triplets to create BIO tags
            for triplet in triplets:
                term = triplet["term"]
                category = triplet["category"]
                polarity = triplet["polarity"]

                # Find term span in text
                term_start = text.find(term)
                if term_start == -1:
                    continue

                term_end = term_start + len(term)

                # Align with tokens
                for token_idx, (start, end) in enumerate(offset_mapping):
                    start, end = start.item(), end.item()

                    # Skip special tokens
                    if start == 0 and end == 0:
                        continue

                    # Check if token overlaps with term
                    if start >= term_start and end <= term_end:
                        if start == term_start or bio_tags[token_idx] == 0:
                            bio_tags[token_idx] = self.bio2id["B"]
                        else:
                            bio_tags[token_idx] = self.bio2id["I"]

            # Use first triplet for category/polarity
            first_triplet = triplets[0]
            category_label = self.taxonomy.get_category_id(
                first_triplet["category"]
            )
            polarity_label = self.taxonomy.get_polarity_id(
                first_triplet["polarity"]
            )

        return {
            "bio_labels": bio_tags,
            "category_labels": torch.tensor(category_label, dtype=torch.long),
            "polarity_labels": torch.tensor(polarity_label, dtype=torch.long)
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

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.

        Args:
            batch: List of samples

        Returns:
            Batched tensors
        """
        # This is a simple version, see collator.py for full implementation
        keys = ["input_ids", "attention_mask", "bio_labels",
                "category_labels", "polarity_labels"]

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
