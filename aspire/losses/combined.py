"""
Combined ASPIRE loss.

Brings together critic and student losses into a unified training objective.
"""

import torch
import torch.nn as nn

from aspire.losses.critic import CriticLoss
from aspire.losses.student import StudentLoss


class AspireLoss(nn.Module):
    """
    Combined ASPIRE training loss.

    Orchestrates the training of both critic and student:
    1. Critic learns to predict teacher judgment
    2. Student learns from critic's predictions

    The key insight: the student doesn't learn directly from the teacher,
    but from the critic's internalization of the teacher's judgment.
    """

    def __init__(
        self,
        # Critic loss weights
        critic_score_weight: float = 1.0,
        critic_reasoning_weight: float = 0.5,
        critic_contrastive_weight: float = 0.3,
        # Student loss weights
        student_reward_weight: float = 1.0,
        student_contrastive_weight: float = 0.5,
        student_trajectory_weight: float = 0.3,
        student_coherence_weight: float = 0.2,
        student_kl_weight: float = 0.1,
        # Other settings
        target_score: float = 9.0,
        contrastive_margin: float = 0.5,
        contrastive_temperature: float = 0.07,
    ):
        super().__init__()

        self.critic_loss = CriticLoss(
            score_weight=critic_score_weight,
            reasoning_weight=critic_reasoning_weight,
            contrastive_weight=critic_contrastive_weight,
            contrastive_temperature=contrastive_temperature,
        )

        self.student_loss = StudentLoss(
            reward_weight=student_reward_weight,
            contrastive_weight=student_contrastive_weight,
            trajectory_weight=student_trajectory_weight,
            coherence_weight=student_coherence_weight,
            kl_weight=student_kl_weight,
            target_score=target_score,
            contrastive_margin=contrastive_margin,
        )

    def compute_critic_loss(
        self,
        predicted_score: torch.Tensor,
        target_score: torch.Tensor,
        predicted_embedding: torch.Tensor | None = None,
        target_embedding: torch.Tensor | None = None,
        negative_embedding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute critic loss.

        Use this to train the critic to predict teacher judgment.
        """
        return self.critic_loss(
            predicted_score=predicted_score,
            target_score=target_score,
            predicted_embedding=predicted_embedding,
            target_embedding=target_embedding,
            negative_embedding=negative_embedding,
        )

    def compute_student_loss(
        self,
        critic_score: torch.Tensor,
        student_embedding: torch.Tensor | None = None,
        teacher_embedding: torch.Tensor | None = None,
        turn_scores: list[torch.Tensor] | None = None,
        student_logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        reference_logits: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute student loss.

        Use this to train the student based on critic feedback.
        """
        return self.student_loss(
            critic_score=critic_score,
            student_embedding=student_embedding,
            teacher_embedding=teacher_embedding,
            turn_scores=turn_scores,
            student_logits=student_logits,
            labels=labels,
            reference_logits=reference_logits,
            attention_mask=attention_mask,
        )

    def forward(
        self,
        # Critic inputs
        critic_predicted_score: torch.Tensor,
        teacher_score: torch.Tensor,
        critic_predicted_embedding: torch.Tensor | None = None,
        teacher_reasoning_embedding: torch.Tensor | None = None,
        # Student inputs
        student_embedding: torch.Tensor | None = None,
        teacher_improved_embedding: torch.Tensor | None = None,
        turn_scores: list[torch.Tensor] | None = None,
        student_logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        reference_logits: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        # Control
        train_critic: bool = True,
        train_student: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Compute full ASPIRE loss.

        Returns dict with all loss components.
        """
        losses = {}

        # Critic loss
        if train_critic:
            critic_losses = self.compute_critic_loss(
                predicted_score=critic_predicted_score,
                target_score=teacher_score,
                predicted_embedding=critic_predicted_embedding,
                target_embedding=teacher_reasoning_embedding,
            )
            for k, v in critic_losses.items():
                losses[f"critic_{k}"] = v

        # Student loss
        if train_student:
            student_losses = self.compute_student_loss(
                critic_score=critic_predicted_score.detach(),  # Don't backprop through critic
                student_embedding=student_embedding,
                teacher_embedding=teacher_improved_embedding,
                turn_scores=turn_scores,
                student_logits=student_logits,
                labels=labels,
                reference_logits=reference_logits,
                attention_mask=attention_mask,
            )
            for k, v in student_losses.items():
                losses[f"student_{k}"] = v

        # Total loss
        total = torch.tensor(0.0, device=critic_predicted_score.device)
        if train_critic and "critic_total" in losses:
            total = total + losses["critic_total"]
        if train_student and "student_total" in losses:
            total = total + losses["student_total"]

        losses["total"] = total

        return losses
