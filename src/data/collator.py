"""
Data collator for XABSA.
"""

import torch
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class XABSACollator:
    """
    Collator for XABSA dataset.

    Handles batching and padding of samples.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch of samples.

        Args:
            batch: List of samples from dataset

        Returns:
            Batched dictionary
        """
        # Collect tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])

        # Labels
        bio_labels = torch.stack([item["bio_labels"] for item in batch])
        category_labels = torch.stack([item["category_labels"] for item in batch])
        polarity_labels = torch.stack([item["polarity_labels"] for item in batch])

        # Metadata (keep as lists)
        texts = [item["text"] for item in batch]
        triplets = [item["triplets"] for item in batch]
        sample_ids = [item["sample_id"] for item in batch]
        langs = [item["lang"] for item in batch]

        # Offset mapping (optional, for inference)
        offset_mappings = None
        if "offset_mapping" in batch[0]:
            offset_mappings = torch.stack([item["offset_mapping"] for item in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bio_labels": bio_labels,
            "category_labels": category_labels,
            "polarity_labels": polarity_labels,
            "offset_mappings": offset_mappings,
            "texts": texts,
            "triplets": triplets,
            "sample_ids": sample_ids,
            "langs": langs
        }
