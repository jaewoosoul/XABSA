"""
Evaluation metrics for XABSA task.

Metrics:
- ATE F1: Aspect Term Extraction (span-level)
- Category F1/Accuracy: Category classification
- Polarity F1/Accuracy: Polarity classification
- Triplet F1: Exact match (term span + category + polarity)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Set, Any
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)


def extract_spans_from_bio(
    bio_tags: List[int],
    bio2id: Dict[str, int] = None
) -> List[Tuple[int, int]]:
    """
    Extract spans from BIO tags.

    Args:
        bio_tags: List of BIO tag IDs
        bio2id: Mapping from tag to ID (default: {"O": 0, "B": 1, "I": 2})

    Returns:
        List of (start, end) tuples
    """
    if bio2id is None:
        bio2id = {"O": 0, "B": 1, "I": 2}

    id2bio = {v: k for k, v in bio2id.items()}
    spans = []
    current_start = None

    for i, tag_id in enumerate(bio_tags):
        tag = id2bio.get(tag_id, "O")

        if tag == "B":
            # Start new span
            if current_start is not None:
                # End previous span
                spans.append((current_start, i))
            current_start = i
        elif tag == "I":
            # Continue span
            if current_start is None:
                # Invalid: I without B, treat as new span
                current_start = i
        elif tag == "O":
            # End span if active
            if current_start is not None:
                spans.append((current_start, i))
                current_start = None

    # Handle final span
    if current_start is not None:
        spans.append((current_start, len(bio_tags)))

    return spans


def compute_ate_f1(
    pred_bio_tags: List[List[int]],
    gold_bio_tags: List[List[int]],
    bio2id: Dict[str, int] = None
) -> Dict[str, float]:
    """
    Compute ATE (Aspect Term Extraction) F1 score.

    Span-level evaluation: a span is correct if both start and end match.

    Args:
        pred_bio_tags: Predicted BIO tags (batch)
        gold_bio_tags: Gold BIO tags (batch)
        bio2id: BIO tag mapping

    Returns:
        Dictionary with precision, recall, f1
    """
    if bio2id is None:
        bio2id = {"O": 0, "B": 1, "I": 2}

    total_pred = 0
    total_gold = 0
    total_correct = 0

    for pred_tags, gold_tags in zip(pred_bio_tags, gold_bio_tags):
        pred_spans = set(extract_spans_from_bio(pred_tags, bio2id))
        gold_spans = set(extract_spans_from_bio(gold_tags, bio2id))

        total_pred += len(pred_spans)
        total_gold += len(gold_spans)
        total_correct += len(pred_spans & gold_spans)

    # Compute metrics
    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "ate_precision": precision,
        "ate_recall": recall,
        "ate_f1": f1,
        "ate_pred_count": total_pred,
        "ate_gold_count": total_gold,
        "ate_correct_count": total_correct
    }


def compute_category_metrics(
    pred_categories: List[int],
    gold_categories: List[int],
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Compute category classification metrics.

    Args:
        pred_categories: Predicted category IDs
        gold_categories: Gold category IDs
        ignore_index: Index to ignore

    Returns:
        Dictionary with accuracy and f1
    """
    # Filter out ignore_index
    valid_mask = np.array(gold_categories) != ignore_index
    pred_valid = np.array(pred_categories)[valid_mask]
    gold_valid = np.array(gold_categories)[valid_mask]

    if len(gold_valid) == 0:
        return {
            "category_accuracy": 0.0,
            "category_f1_macro": 0.0,
            "category_f1_weighted": 0.0
        }

    accuracy = accuracy_score(gold_valid, pred_valid)
    f1_macro = f1_score(gold_valid, pred_valid, average="macro", zero_division=0)
    f1_weighted = f1_score(gold_valid, pred_valid, average="weighted", zero_division=0)

    return {
        "category_accuracy": accuracy,
        "category_f1_macro": f1_macro,
        "category_f1_weighted": f1_weighted
    }


def compute_polarity_metrics(
    pred_polarities: List[int],
    gold_polarities: List[int],
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Compute polarity classification metrics.

    Args:
        pred_polarities: Predicted polarity IDs
        gold_polarities: Gold polarity IDs
        ignore_index: Index to ignore

    Returns:
        Dictionary with accuracy and f1
    """
    # Filter out ignore_index
    valid_mask = np.array(gold_polarities) != ignore_index
    pred_valid = np.array(pred_polarities)[valid_mask]
    gold_valid = np.array(gold_polarities)[valid_mask]

    if len(gold_valid) == 0:
        return {
            "polarity_accuracy": 0.0,
            "polarity_f1_macro": 0.0,
            "polarity_f1_weighted": 0.0
        }

    accuracy = accuracy_score(gold_valid, pred_valid)
    f1_macro = f1_score(gold_valid, pred_valid, average="macro", zero_division=0)
    f1_weighted = f1_score(gold_valid, pred_valid, average="weighted", zero_division=0)

    return {
        "polarity_accuracy": accuracy,
        "polarity_f1_macro": f1_macro,
        "polarity_f1_weighted": f1_weighted
    }


def compute_triplet_f1(
    pred_triplets: List[List[Tuple[Tuple[int, int], int, int]]],
    gold_triplets: List[List[Tuple[Tuple[int, int], int, int]]]
) -> Dict[str, float]:
    """
    Compute triplet F1 score.

    A triplet is correct if:
    - Term span matches (start, end)
    - Category matches
    - Polarity matches

    Args:
        pred_triplets: List of predicted triplets per sample
                      Each triplet: ((start, end), category_id, polarity_id)
        gold_triplets: List of gold triplets per sample

    Returns:
        Dictionary with precision, recall, f1
    """
    total_pred = 0
    total_gold = 0
    total_correct = 0

    for pred, gold in zip(pred_triplets, gold_triplets):
        pred_set = set(pred)
        gold_set = set(gold)

        total_pred += len(pred_set)
        total_gold += len(gold_set)
        total_correct += len(pred_set & gold_set)

    # Compute metrics
    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "triplet_precision": precision,
        "triplet_recall": recall,
        "triplet_f1": f1,
        "triplet_pred_count": total_pred,
        "triplet_gold_count": total_gold,
        "triplet_correct_count": total_correct
    }


def build_triplets_from_predictions(
    bio_tags: List[int],
    term_category_labels: List[int],
    term_polarity_labels: List[int],
    num_terms: int,
    bio2id: Dict[str, int] = None
) -> List[Tuple[Tuple[int, int], int, int]]:
    """
    Build triplets from model predictions.

    Args:
        bio_tags: BIO tag sequence
        term_category_labels: Category labels for each term (can be longer than predicted spans)
        term_polarity_labels: Polarity labels for each term (can be longer than predicted spans)
        num_terms: Number of valid terms (for gold, use actual num_terms; for pred, use len(spans))
        bio2id: BIO tag mapping

    Returns:
        List of triplets ((start, end), category_id, polarity_id)
    """
    spans = extract_spans_from_bio(bio_tags, bio2id)
    triplets = []

    # Use the number of predicted spans, not num_terms
    # num_terms is for gold labels, but for predictions we use actual span count
    num_predicted_spans = len(spans)
    
    # Use minimum of predicted spans and available labels
    max_terms_to_use = min(num_predicted_spans, len(term_category_labels), len(term_polarity_labels))

    for i in range(max_terms_to_use):
        if i < len(spans):
            start, end = spans[i]
            category_id = term_category_labels[i]
            polarity_id = term_polarity_labels[i]

            # Skip if labels are invalid (e.g., -100)
            if category_id >= 0 and polarity_id >= 0:
                triplets.append(((start, end), category_id, polarity_id))

    return triplets


def compute_all_metrics(
    pred_bio_tags: List[List[int]],
    gold_bio_tags: List[List[int]],
    pred_term_categories: List[List[int]],
    gold_term_categories: List[List[int]],
    pred_term_polarities: List[List[int]],
    gold_term_polarities: List[List[int]],
    num_terms: List[int],
    bio2id: Dict[str, int] = None
) -> Dict[str, float]:
    """
    Compute all metrics.

    Args:
        pred_bio_tags: Predicted BIO tags (batch)
        gold_bio_tags: Gold BIO tags (batch)
        pred_term_categories: Predicted categories per term (batch)
        gold_term_categories: Gold categories per term (batch)
        pred_term_polarities: Predicted polarities per term (batch)
        gold_term_polarities: Gold polarities per term (batch)
        num_terms: Number of terms per sample
        bio2id: BIO tag mapping

    Returns:
        Dictionary with all metrics
    """
    if bio2id is None:
        bio2id = {"O": 0, "B": 1, "I": 2}

    # ATE metrics
    ate_metrics = compute_ate_f1(pred_bio_tags, gold_bio_tags, bio2id)

    # Flatten term-level labels for category/polarity metrics
    pred_cats_flat = []
    gold_cats_flat = []
    pred_pols_flat = []
    gold_pols_flat = []

    for pred_cat, gold_cat, pred_pol, gold_pol, n_terms in zip(
        pred_term_categories, gold_term_categories,
        pred_term_polarities, gold_term_polarities, num_terms
    ):
        pred_cats_flat.extend(pred_cat[:n_terms])
        gold_cats_flat.extend(gold_cat[:n_terms])
        pred_pols_flat.extend(pred_pol[:n_terms])
        gold_pols_flat.extend(gold_pol[:n_terms])

    # Category and polarity metrics
    category_metrics = compute_category_metrics(pred_cats_flat, gold_cats_flat)
    polarity_metrics = compute_polarity_metrics(pred_pols_flat, gold_pols_flat)

    # Build triplets for triplet F1
    pred_triplets = []
    gold_triplets = []

    for pred_bio, gold_bio, pred_cat, gold_cat, pred_pol, gold_pol, n_terms in zip(
        pred_bio_tags, gold_bio_tags,
        pred_term_categories, gold_term_categories,
        pred_term_polarities, gold_term_polarities, num_terms
    ):
        # For predictions: use actual number of predicted spans (ignore n_terms)
        # For gold: use n_terms to limit to actual gold terms
        pred_trip = build_triplets_from_predictions(pred_bio, pred_cat, pred_pol, len(extract_spans_from_bio(pred_bio, bio2id)), bio2id)
        gold_trip = build_triplets_from_predictions(gold_bio, gold_cat, gold_pol, n_terms, bio2id)
        pred_triplets.append(pred_trip)
        gold_triplets.append(gold_trip)

    triplet_metrics = compute_triplet_f1(pred_triplets, gold_triplets)

    # Combine all metrics
    all_metrics = {
        **ate_metrics,
        **category_metrics,
        **polarity_metrics,
        **triplet_metrics
    }

    return all_metrics


class MetricsTracker:
    """
    Track metrics during training and evaluation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.pred_bio_tags = []
        self.gold_bio_tags = []
        self.pred_term_categories = []
        self.gold_term_categories = []
        self.pred_term_polarities = []
        self.gold_term_polarities = []
        self.num_terms = []

    def update(
        self,
        pred_bio: torch.Tensor,
        gold_bio: torch.Tensor,
        pred_cat: torch.Tensor,
        gold_cat: torch.Tensor,
        pred_pol: torch.Tensor,
        gold_pol: torch.Tensor,
        n_terms: torch.Tensor
    ):
        """
        Update metrics with batch predictions.

        Args:
            pred_bio: [batch_size, seq_len]
            gold_bio: [batch_size, seq_len]
            pred_cat: [batch_size, max_terms]
            gold_cat: [batch_size, max_terms]
            pred_pol: [batch_size, max_terms]
            gold_pol: [batch_size, max_terms]
            n_terms: [batch_size]
        """
        # Convert to lists
        self.pred_bio_tags.extend(pred_bio.cpu().tolist())
        self.gold_bio_tags.extend(gold_bio.cpu().tolist())
        self.pred_term_categories.extend(pred_cat.cpu().tolist())
        self.gold_term_categories.extend(gold_cat.cpu().tolist())
        self.pred_term_polarities.extend(pred_pol.cpu().tolist())
        self.gold_term_polarities.extend(gold_pol.cpu().tolist())
        self.num_terms.extend(n_terms.cpu().tolist())

    def compute(self, bio2id: Dict[str, int] = None) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            bio2id: BIO tag mapping

        Returns:
            Dictionary of metrics
        """
        return compute_all_metrics(
            pred_bio_tags=self.pred_bio_tags,
            gold_bio_tags=self.gold_bio_tags,
            pred_term_categories=self.pred_term_categories,
            gold_term_categories=self.gold_term_categories,
            pred_term_polarities=self.pred_term_polarities,
            gold_term_polarities=self.gold_term_polarities,
            num_terms=self.num_terms,
            bio2id=bio2id
        )
