"""
Student loss functions.

Train the student using signals from the critic (internalized teacher).
Multiple loss components capture different aspects of good responses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardLoss(nn.Module):
    """
    Reward-based loss from critic score.

    Higher critic score = lower loss. Encourages student to
    produce responses the critic judges favorably.
    """

    def __init__(self, target_score: float = 9.0, margin: float = 1.0):
        super().__init__()
        self.target_score = target_score
        self.margin = margin

    def forward(
        self,
        critic_score: torch.Tensor,
        response_logprobs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            critic_score: Critic's score for the response [batch_size]
            response_logprobs: Log probabilities of response tokens (for policy gradient)

        Returns:
            Scalar loss
        """
        # Normalize score to 0-1 range
        normalized_score = critic_score / 10.0

        # Hinge-style loss: penalize scores below target
        score_gap = (self.target_score / 10.0) - normalized_score
        loss = F.relu(score_gap + self.margin / 10.0)

        if response_logprobs is not None:
            # Policy gradient: weight log probs by reward
            reward = normalized_score - 0.5  # Center around 0.5
            loss = loss - (reward.detach() * response_logprobs.mean(dim=-1))

        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss between student and teacher responses.

    Pulls student embeddings toward teacher's improved response,
    pushes away from poor responses.
    """

    def __init__(self, margin: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        student_embedding: torch.Tensor,
        teacher_embedding: torch.Tensor,
        negative_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            student_embedding: Student response embedding [batch_size, dim]
            teacher_embedding: Teacher's improved response embedding [batch_size, dim]
            negative_embedding: Poor response embedding [batch_size, dim] (optional)

        Returns:
            Scalar loss
        """
        # Normalize
        student_norm = F.normalize(student_embedding, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_embedding, p=2, dim=-1)

        # Positive similarity (student <-> teacher improved)
        pos_sim = (student_norm * teacher_norm).sum(dim=-1)

        if negative_embedding is not None:
            negative_norm = F.normalize(negative_embedding, p=2, dim=-1)
            neg_sim = (student_norm * negative_norm).sum(dim=-1)

            # Triplet-style margin loss
            loss = F.relu(self.margin - pos_sim + neg_sim)
        else:
            # Pull toward teacher
            loss = 1.0 - pos_sim

        return loss.mean()


class TrajectoryLoss(nn.Module):
    """
    Trajectory improvement loss.

    Rewards improvement across dialogue turns. The student should
    get better (higher critic scores) as the dialogue progresses.
    """

    def __init__(self, improvement_bonus: float = 0.5):
        super().__init__()
        self.improvement_bonus = improvement_bonus

    def forward(
        self,
        turn_scores: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            turn_scores: List of critic scores for each turn [batch_size] each

        Returns:
            Scalar loss (negative = rewarding improvement)
        """
        if len(turn_scores) < 2:
            return torch.tensor(0.0, device=turn_scores[0].device)

        # Compute improvements between consecutive turns
        improvements = []
        for i in range(len(turn_scores) - 1):
            improvement = turn_scores[i + 1] - turn_scores[i]
            improvements.append(improvement)

        # Stack and compute mean improvement
        improvements = torch.stack(improvements, dim=-1)  # [batch_size, num_improvements]

        # Loss: negative of improvement (we want improvement to increase)
        # Bonus for consistent improvement
        mean_improvement = improvements.mean()

        # Penalize decline
        declines = F.relu(-improvements)
        decline_penalty = declines.mean()

        loss = -mean_improvement + decline_penalty

        return loss


class CoherenceLoss(nn.Module):
    """
    Response coherence loss.

    Encourages internally consistent responses. Uses perplexity
    or self-consistency measures.
    """

    def __init__(self, target_perplexity: float = 10.0):
        super().__init__()
        self.target_perplexity = target_perplexity

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]
            attention_mask: Mask for valid positions

        Returns:
            Scalar loss
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Cross entropy per token
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        token_losses = loss_fct(flat_logits, flat_labels)
        token_losses = token_losses.view(shift_labels.size())

        # Mask if provided
        if attention_mask is not None:
            mask = attention_mask[..., 1:].float()
            token_losses = token_losses * mask
            avg_loss = token_losses.sum() / mask.sum().clamp(min=1)
        else:
            avg_loss = token_losses.mean()

        # Perplexity
        perplexity = torch.exp(avg_loss)

        # Loss: penalize perplexity above target
        loss = F.relu(perplexity - self.target_perplexity)

        return loss


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss to keep student close to reference.

    Prevents the student from drifting too far from its
    original behavior during fine-tuning.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        student_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: Current model logits [batch_size, seq_len, vocab_size]
            reference_logits: Reference model logits [batch_size, seq_len, vocab_size]
            attention_mask: Mask for valid positions

        Returns:
            Scalar KL divergence loss
        """
        # Convert to log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)

        # KL divergence per position
        kl_div = F.kl_div(student_log_probs, reference_probs, reduction="none")
        kl_div = kl_div.sum(dim=-1)  # Sum over vocab

        # Mask if provided
        if attention_mask is not None:
            mask = attention_mask.float()
            kl_div = kl_div * mask
            avg_kl = kl_div.sum() / mask.sum().clamp(min=1)
        else:
            avg_kl = kl_div.mean()

        return self.beta * avg_kl


class StudentLoss(nn.Module):
    """
    Combined student loss.

    Combines multiple loss components to train the student.
    """

    def __init__(
        self,
        reward_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        trajectory_weight: float = 0.3,
        coherence_weight: float = 0.2,
        kl_weight: float = 0.1,
        target_score: float = 9.0,
        contrastive_margin: float = 0.5,
    ):
        super().__init__()

        self.reward_weight = reward_weight
        self.contrastive_weight = contrastive_weight
        self.trajectory_weight = trajectory_weight
        self.coherence_weight = coherence_weight
        self.kl_weight = kl_weight

        self.reward_loss = RewardLoss(target_score=target_score)
        self.contrastive_loss = ContrastiveLoss(margin=contrastive_margin)
        self.trajectory_loss = TrajectoryLoss()
        self.coherence_loss = CoherenceLoss()
        self.kl_loss = KLDivergenceLoss(beta=kl_weight)

    def forward(
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
        Compute combined student loss.

        Returns dict with individual losses and total.
        """
        losses = {}

        # Reward loss (always computed)
        losses["reward"] = self.reward_loss(critic_score)

        # Contrastive loss
        if student_embedding is not None and teacher_embedding is not None:
            losses["contrastive"] = self.contrastive_loss(student_embedding, teacher_embedding)

        # Trajectory loss
        if turn_scores is not None and len(turn_scores) >= 2:
            losses["trajectory"] = self.trajectory_loss(turn_scores)

        # Coherence loss
        if student_logits is not None and labels is not None:
            losses["coherence"] = self.coherence_loss(student_logits, labels, attention_mask)

        # KL divergence
        if student_logits is not None and reference_logits is not None:
            losses["kl"] = self.kl_loss(student_logits, reference_logits, attention_mask)

        # Total loss
        total = self.reward_weight * losses["reward"]

        if "contrastive" in losses:
            total = total + self.contrastive_weight * losses["contrastive"]

        if "trajectory" in losses:
            total = total + self.trajectory_weight * losses["trajectory"]

        if "coherence" in losses:
            total = total + self.coherence_weight * losses["coherence"]

        if "kl" in losses:
            total = total + losses["kl"]  # KL weight already applied

        losses["total"] = total

        return losses
