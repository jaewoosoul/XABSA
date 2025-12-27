"""
Student model for XABSA task.

XLM-RoBERTa based multi-task model with:
- Token classification head (BIO tagging for ATE)
- Category classification head (per-term)
- Polarity classification head (per-term)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple


class XABSAModel(nn.Module):
    """
    Multi-task model for XABSA.

    Architecture:
    - Shared encoder: XLM-RoBERTa
    - Head 1: Token classifier (BIO tags for aspect term extraction)
    - Head 2: Category classifier (term-level)
    - Head 3: Polarity classifier (term-level)
    """

    def __init__(
        self,
        backbone: str = "xlm-roberta-base",
        num_ate_labels: int = 3,  # O, B, I
        num_category_labels: int = 13,
        num_polarity_labels: int = 3,
        dropout: float = 0.1,
        term_pooling: str = "mean",  # mean or start
        freeze_encoder: bool = False
    ):
        """
        Initialize model.

        Args:
            backbone: HuggingFace model name
            num_ate_labels: Number of BIO labels
            num_category_labels: Number of categories
            num_polarity_labels: Number of polarities
            dropout: Dropout rate
            term_pooling: Term pooling strategy (mean or start)
            freeze_encoder: Whether to freeze encoder
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_ate_labels = num_ate_labels
        self.num_category_labels = num_category_labels
        self.num_polarity_labels = num_polarity_labels
        self.term_pooling = term_pooling

        # Load backbone
        config = AutoConfig.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone, config=config)
        self.hidden_size = config.hidden_size

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Task-specific heads
        # Head 1: Token classification (BIO tagging)
        self.ate_classifier = nn.Linear(self.hidden_size, num_ate_labels)

        # Head 2: Category classification (term-level)
        self.category_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_category_labels)
        )

        # Head 3: Polarity classification (term-level)
        self.polarity_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_polarity_labels)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        term_masks: Optional[torch.Tensor] = None,
        bio_labels: Optional[torch.Tensor] = None,
        category_labels: Optional[torch.Tensor] = None,
        polarity_labels: Optional[torch.Tensor] = None,
        term_category_labels: Optional[torch.Tensor] = None,
        term_polarity_labels: Optional[torch.Tensor] = None,
        num_terms: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            term_masks: [batch_size, max_terms, seq_len] - masks for each term
            bio_labels: [batch_size, seq_len] - BIO labels
            category_labels: [batch_size] - category labels (simplified, first term)
            polarity_labels: [batch_size] - polarity labels (simplified, first term)
            term_category_labels: [batch_size, max_terms] - category labels per term
            term_polarity_labels: [batch_size, max_terms] - polarity labels per term
            num_terms: [batch_size] - number of terms in each sample

        Returns:
            Dictionary with logits and optionally loss
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)

        # Head 1: ATE (token classification)
        ate_logits = self.ate_classifier(sequence_output)  # [batch_size, seq_len, num_ate_labels]

        # Heads 2 & 3: Category and Polarity (term-level)
        # We need to pool term representations
        if term_masks is not None:
            # term_masks: [batch_size, max_terms, seq_len]
            batch_size, max_terms, seq_len = term_masks.shape

            # Pool term representations
            term_representations = self._pool_term_representations(
                sequence_output, term_masks
            )  # [batch_size, max_terms, hidden_size]

            # Classify each term
            category_logits = self.category_classifier(term_representations)  # [batch_size, max_terms, num_categories]
            polarity_logits = self.polarity_classifier(term_representations)  # [batch_size, max_terms, num_polarities]
        else:
            # Fallback: use [CLS] token representation
            cls_representation = sequence_output[:, 0, :]  # [batch_size, hidden_size]
            category_logits = self.category_classifier(cls_representation).unsqueeze(1)  # [batch_size, 1, num_categories]
            polarity_logits = self.polarity_classifier(cls_representation).unsqueeze(1)  # [batch_size, 1, num_polarities]

        # Prepare output
        output = {
            "ate_logits": ate_logits,
            "category_logits": category_logits,
            "polarity_logits": polarity_logits,
            "sequence_output": sequence_output
        }

        # Compute loss if labels provided
        if bio_labels is not None and term_category_labels is not None and term_polarity_labels is not None:
            loss = self.compute_loss(
                ate_logits=ate_logits,
                category_logits=category_logits,
                polarity_logits=polarity_logits,
                bio_labels=bio_labels,
                term_category_labels=term_category_labels,
                term_polarity_labels=term_polarity_labels,
                attention_mask=attention_mask
            )
            output["loss"] = loss

        return output

    def _pool_term_representations(
        self,
        sequence_output: torch.Tensor,
        term_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool term representations from sequence output.

        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            term_masks: [batch_size, max_terms, seq_len]

        Returns:
            term_representations: [batch_size, max_terms, hidden_size]
        """
        batch_size, seq_len, hidden_size = sequence_output.shape
        max_terms = term_masks.shape[1]

        # Expand dimensions for broadcasting
        # sequence_output: [batch_size, 1, seq_len, hidden_size]
        # term_masks: [batch_size, max_terms, seq_len, 1]
        sequence_expanded = sequence_output.unsqueeze(1).expand(batch_size, max_terms, seq_len, hidden_size)
        term_masks_expanded = term_masks.unsqueeze(-1).float()

        if self.term_pooling == "mean":
            # Mean pooling
            masked_output = sequence_expanded * term_masks_expanded
            term_sums = masked_output.sum(dim=2)  # [batch_size, max_terms, hidden_size]
            term_counts = term_masks_expanded.sum(dim=2).clamp(min=1)  # [batch_size, max_terms, 1]
            term_representations = term_sums / term_counts
        elif self.term_pooling == "start":
            # Use first token of each term
            # Find first token index for each term
            first_token_indices = term_masks.argmax(dim=2)  # [batch_size, max_terms]

            # Gather first tokens
            batch_indices = torch.arange(batch_size, device=sequence_output.device).unsqueeze(1).expand(batch_size, max_terms)
            term_representations = sequence_output[batch_indices, first_token_indices]  # [batch_size, max_terms, hidden_size]
        else:
            raise ValueError(f"Unknown term pooling strategy: {self.term_pooling}")

        return term_representations

    def compute_loss(
        self,
        ate_logits: torch.Tensor,
        category_logits: torch.Tensor,
        polarity_logits: torch.Tensor,
        bio_labels: torch.Tensor,
        term_category_labels: torch.Tensor,
        term_polarity_labels: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute multi-task loss.

        Args:
            ate_logits: [batch_size, seq_len, num_ate_labels]
            category_logits: [batch_size, max_terms, num_categories]
            polarity_logits: [batch_size, max_terms, num_polarities]
            bio_labels: [batch_size, seq_len]
            term_category_labels: [batch_size, max_terms]
            term_polarity_labels: [batch_size, max_terms]
            attention_mask: [batch_size, seq_len]
            loss_weights: Dictionary of loss weights

        Returns:
            Combined loss
        """
        if loss_weights is None:
            loss_weights = {"ate": 1.0, "category": 1.0, "polarity": 1.0}

        # ATE loss (token-level)
        # Flatten for CrossEntropyLoss
        ate_logits_flat = ate_logits.view(-1, self.num_ate_labels)
        bio_labels_flat = bio_labels.view(-1)

        # Only compute loss on non-padding tokens
        active_loss = attention_mask.view(-1) == 1
        active_logits = ate_logits_flat[active_loss]
        active_labels = bio_labels_flat[active_loss]

        # Compute class weights for BIO labels to handle class imbalance
        # Count occurrences of each class
        if active_labels.numel() > 0:
            unique_labels, counts = torch.unique(active_labels, return_counts=True)
            total = active_labels.numel()
            
            # Compute inverse frequency weights
            class_weights = torch.ones(self.num_ate_labels, device=active_logits.device)
            for label, count in zip(unique_labels, counts):
                if count > 0:
                    # Inverse frequency: more frequent = lower weight
                    class_weights[label] = total / (self.num_ate_labels * count.float())
            
            # Normalize weights
            class_weights = class_weights / class_weights.sum() * self.num_ate_labels
        else:
            class_weights = None

        # Use weighted loss to handle class imbalance (ATE only)
        if class_weights is not None:
            ate_loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
        else:
            ate_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        ate_loss = ate_loss_fct(active_logits, active_labels)

        # Category loss (term-level) - use separate loss function
        category_logits_flat = category_logits.view(-1, self.num_category_labels)
        category_labels_flat = term_category_labels.view(-1)
        category_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        category_loss = category_loss_fct(category_logits_flat, category_labels_flat)

        # Polarity loss (term-level) - use separate loss function
        polarity_logits_flat = polarity_logits.view(-1, self.num_polarity_labels)
        polarity_labels_flat = term_polarity_labels.view(-1)
        polarity_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        polarity_loss = polarity_loss_fct(polarity_logits_flat, polarity_labels_flat)

        # Combined loss
        total_loss = (
            loss_weights["ate"] * ate_loss +
            loss_weights["category"] * category_loss +
            loss_weights["polarity"] * polarity_loss
        )

        return total_loss

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        offset_mapping: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict triplets for inference.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            offset_mapping: [batch_size, seq_len, 2] - character offsets

        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        # Get predictions
        ate_preds = torch.argmax(outputs["ate_logits"], dim=-1)  # [batch_size, seq_len]
        category_preds = torch.argmax(outputs["category_logits"], dim=-1)  # [batch_size, max_terms]
        polarity_preds = torch.argmax(outputs["polarity_logits"], dim=-1)  # [batch_size, max_terms]

        return {
            "ate_predictions": ate_preds,
            "category_predictions": category_preds,
            "polarity_predictions": polarity_preds,
            **outputs
        }
