"""
Critic loss functions.

Train the critic to predict what the teacher would think.
The better the critic predicts, the better it has internalized
the teacher's judgment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticScoreLoss(nn.Module):
    """
    Loss for critic's score prediction.

    Trains the critic to predict the teacher's score.
    Uses smooth L1 loss for robustness to outliers.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        predicted_score: torch.Tensor,
        target_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted_score: Critic's predicted score [batch_size]
            target_score: Teacher's actual score [batch_size]

        Returns:
            Scalar loss
        """
        return F.smooth_l1_loss(predicted_score, target_score, beta=self.beta)


class CriticReasoningLoss(nn.Module):
    """
    Loss for critic's reasoning embedding.

    Trains the critic to produce embeddings that align with
    the teacher's reasoning. Uses cosine embedding loss.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        predicted_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted_embedding: Critic's reasoning embedding [batch_size, dim]
            target_embedding: Encoded teacher reasoning [batch_size, dim]

        Returns:
            Scalar loss (lower = more aligned)
        """
        # Normalize embeddings
        pred_norm = F.normalize(predicted_embedding, p=2, dim=-1)
        target_norm = F.normalize(target_embedding, p=2, dim=-1)

        # Cosine similarity
        cos_sim = (pred_norm * target_norm).sum(dim=-1)

        # Loss: 1 - cosine similarity (want similarity to be high)
        loss = 1.0 - cos_sim

        return loss.mean()


class CriticContrastiveLoss(nn.Module):
    """
    Contrastive loss for critic reasoning.

    Pulls critic's embedding toward good teacher explanations
    and pushes away from bad ones.
    """

    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        predicted_embedding: torch.Tensor,
        positive_embedding: torch.Tensor,
        negative_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            predicted_embedding: Critic's embedding [batch_size, dim]
            positive_embedding: Good teacher reasoning [batch_size, dim]
            negative_embedding: Bad/contrasting reasoning [batch_size, dim] (optional)

        Returns:
            Scalar loss
        """
        pred_norm = F.normalize(predicted_embedding, p=2, dim=-1)
        pos_norm = F.normalize(positive_embedding, p=2, dim=-1)

        # Positive similarity
        pos_sim = (pred_norm * pos_norm).sum(dim=-1) / self.temperature

        if negative_embedding is not None:
            neg_norm = F.normalize(negative_embedding, p=2, dim=-1)
            neg_sim = (pred_norm * neg_norm).sum(dim=-1) / self.temperature

            # InfoNCE-style loss
            logits = torch.stack([pos_sim, neg_sim], dim=-1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # Simple margin loss
            loss = F.relu(self.margin - pos_sim).mean()

        return loss


class CriticLoss(nn.Module):
    """
    Combined critic loss.

    Combines score prediction and reasoning alignment losses.
    """

    def __init__(
        self,
        score_weight: float = 1.0,
        reasoning_weight: float = 0.5,
        contrastive_weight: float = 0.3,
        score_beta: float = 1.0,
        reasoning_margin: float = 0.0,
        contrastive_temperature: float = 0.07,
    ):
        super().__init__()

        self.score_weight = score_weight
        self.reasoning_weight = reasoning_weight
        self.contrastive_weight = contrastive_weight

        self.score_loss = CriticScoreLoss(beta=score_beta)
        self.reasoning_loss = CriticReasoningLoss(margin=reasoning_margin)
        self.contrastive_loss = CriticContrastiveLoss(temperature=contrastive_temperature)

    def forward(
        self,
        predicted_score: torch.Tensor,
        target_score: torch.Tensor,
        predicted_embedding: torch.Tensor | None = None,
        target_embedding: torch.Tensor | None = None,
        negative_embedding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined critic loss.

        Returns dict with individual losses and total.
        """
        losses = {}

        # Score loss (always computed)
        losses["score"] = self.score_loss(predicted_score, target_score)

        # Reasoning loss (if embeddings provided)
        if predicted_embedding is not None and target_embedding is not None:
            losses["reasoning"] = self.reasoning_loss(predicted_embedding, target_embedding)

            # Contrastive loss (if negative provided)
            if negative_embedding is not None:
                losses["contrastive"] = self.contrastive_loss(
                    predicted_embedding, target_embedding, negative_embedding
                )

        # Total loss
        total = self.score_weight * losses["score"]

        if "reasoning" in losses:
            total = total + self.reasoning_weight * losses["reasoning"]

        if "contrastive" in losses:
            total = total + self.contrastive_weight * losses["contrastive"]

        losses["total"] = total

        return losses
