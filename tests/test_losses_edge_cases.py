"""
Edge case tests for ASPIRE loss functions.

Tests for boundary conditions, numerical stability, and edge cases:
- Zero/empty inputs
- Very large/small values
- Single element batches
- Gradient stability
- Mixed precision behavior

Windows compatibility notes:
- Use num_workers=0 in DataLoader tests
- Use if __name__ == "__main__": freeze_support() pattern
"""

from __future__ import annotations

from multiprocessing import freeze_support

import pytest
import torch
import torch.nn.functional as F

from aspire.losses.student import (
    RewardLoss,
    ContrastiveLoss,
    TrajectoryLoss,
    CoherenceLoss,
    KLDivergenceLoss,
    StudentLoss,
)
from aspire.losses.critic import (
    CriticLoss,
    CriticScoreLoss,
    CriticReasoningLoss,
    CriticContrastiveLoss,
)
from aspire.losses.combined import AspireLoss


# ============================================================================
# RewardLoss Edge Cases
# ============================================================================

class TestRewardLossEdgeCases:
    """Edge case tests for RewardLoss."""

    def test_reward_loss_perfect_score(self):
        """RewardLoss with score=10.0 (maximum)."""
        loss_fn = RewardLoss(target_score=9.0)
        perfect_score = torch.tensor([10.0, 10.0, 10.0])

        loss = loss_fn(perfect_score)

        # Perfect score should give minimal loss
        assert loss >= 0
        assert loss < 0.5  # Should be very low

    def test_reward_loss_zero_score(self):
        """RewardLoss with score=0.0 (minimum)."""
        loss_fn = RewardLoss(target_score=9.0)
        zero_score = torch.tensor([0.0, 0.0, 0.0])

        loss = loss_fn(zero_score)

        # Zero score should give high loss
        assert loss > 0

    def test_reward_loss_single_element(self):
        """RewardLoss with batch_size=1."""
        loss_fn = RewardLoss()
        single = torch.tensor([5.0])

        loss = loss_fn(single)

        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)

    def test_reward_loss_with_logprobs(self):
        """RewardLoss with response_logprobs (policy gradient mode)."""
        loss_fn = RewardLoss()
        scores = torch.tensor([7.0, 8.0])
        logprobs = torch.randn(2, 10)  # [batch, seq_len]

        loss = loss_fn(scores, response_logprobs=logprobs)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_reward_loss_gradient_flow(self):
        """Verify gradients flow through RewardLoss."""
        loss_fn = RewardLoss()
        scores = torch.tensor([5.0, 6.0], requires_grad=True)

        loss = loss_fn(scores)
        loss.backward()

        assert scores.grad is not None
        assert scores.grad.shape == scores.shape


# ============================================================================
# ContrastiveLoss Edge Cases
# ============================================================================

class TestContrastiveLossEdgeCases:
    """Edge case tests for ContrastiveLoss."""

    def test_contrastive_loss_identical_embeddings(self):
        """ContrastiveLoss with identical student/teacher."""
        loss_fn = ContrastiveLoss()
        embed = torch.randn(4, 256)

        loss = loss_fn(embed, embed.clone())

        # Identical embeddings = perfect alignment = low loss
        assert loss < 0.1

    def test_contrastive_loss_orthogonal_embeddings(self):
        """ContrastiveLoss with orthogonal embeddings."""
        loss_fn = ContrastiveLoss()
        # Create orthogonal vectors
        student = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        teacher = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

        loss = loss_fn(student, teacher)

        # Orthogonal = zero similarity = high loss
        assert loss > 0.5

    def test_contrastive_loss_opposite_embeddings(self):
        """ContrastiveLoss with opposite (negated) embeddings."""
        loss_fn = ContrastiveLoss()
        embed = torch.randn(4, 256)

        loss = loss_fn(embed, -embed)  # Negative = opposite direction

        # Opposite direction = maximum distance
        assert loss > 1.0

    def test_contrastive_loss_with_negative(self):
        """ContrastiveLoss with negative embedding (triplet mode)."""
        loss_fn = ContrastiveLoss(margin=0.5)
        student = torch.randn(4, 256)
        positive = student + torch.randn(4, 256) * 0.1  # Similar
        negative = torch.randn(4, 256)  # Random = different

        loss = loss_fn(student, positive, negative_embedding=negative)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_contrastive_loss_single_dimension(self):
        """ContrastiveLoss with 1D embeddings."""
        loss_fn = ContrastiveLoss()
        student = torch.tensor([[1.0], [0.5]])
        teacher = torch.tensor([[1.0], [0.5]])

        loss = loss_fn(student, teacher)

        assert not torch.isnan(loss)


# ============================================================================
# TrajectoryLoss Edge Cases
# ============================================================================

class TestTrajectoryLossEdgeCases:
    """Edge case tests for TrajectoryLoss."""

    def test_trajectory_loss_single_turn(self):
        """TrajectoryLoss with only one turn (no improvement possible)."""
        loss_fn = TrajectoryLoss()
        single = [torch.tensor([5.0, 6.0])]

        loss = loss_fn(single)

        # Single turn = zero loss (no trajectory)
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_trajectory_loss_constant_scores(self):
        """TrajectoryLoss with constant scores (no change)."""
        loss_fn = TrajectoryLoss()
        constant = [
            torch.tensor([5.0, 5.0]),
            torch.tensor([5.0, 5.0]),
            torch.tensor([5.0, 5.0]),
        ]

        loss = loss_fn(constant)

        # No change = zero improvement = zero loss
        assert abs(loss.item()) < 0.01

    def test_trajectory_loss_perfect_improvement(self):
        """TrajectoryLoss with monotonic improvement."""
        loss_fn = TrajectoryLoss()
        improving = [
            torch.tensor([3.0]),
            torch.tensor([5.0]),
            torch.tensor([7.0]),
            torch.tensor([9.0]),
        ]

        loss = loss_fn(improving)

        # Perfect improvement = negative loss (reward)
        assert loss < 0

    def test_trajectory_loss_decline(self):
        """TrajectoryLoss with declining scores."""
        loss_fn = TrajectoryLoss()
        declining = [
            torch.tensor([9.0]),
            torch.tensor([7.0]),
            torch.tensor([5.0]),
        ]

        loss = loss_fn(declining)

        # Decline = positive loss (penalty)
        assert loss > 0

    def test_trajectory_loss_mixed(self):
        """TrajectoryLoss with mixed improvement/decline."""
        loss_fn = TrajectoryLoss()
        mixed = [
            torch.tensor([5.0]),
            torch.tensor([7.0]),  # +2
            torch.tensor([6.0]),  # -1
            torch.tensor([8.0]),  # +2
        ]

        loss = loss_fn(mixed)

        # Net positive improvement, but with penalty for decline
        assert not torch.isnan(loss)


# ============================================================================
# CoherenceLoss Edge Cases
# ============================================================================

class TestCoherenceLossEdgeCases:
    """Edge case tests for CoherenceLoss."""

    def test_coherence_loss_perfect_prediction(self):
        """CoherenceLoss when model predicts correctly."""
        loss_fn = CoherenceLoss(target_perplexity=10.0)

        # Create logits where model predicts correctly
        batch_size, seq_len, vocab_size = 2, 10, 100
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Moderate confidence logits (not extreme values)
        logits = torch.zeros((batch_size, seq_len, vocab_size))
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s, labels[b, s]] = 5.0  # Higher but not extreme

        loss = loss_fn(logits, labels)

        # Should not be NaN and should be finite
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_coherence_loss_random_prediction(self):
        """CoherenceLoss with random logits."""
        loss_fn = CoherenceLoss(target_perplexity=10.0)

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = loss_fn(logits, labels)

        # Random = high perplexity = high loss
        assert not torch.isnan(loss)

    def test_coherence_loss_with_mask(self):
        """CoherenceLoss with attention mask."""
        loss_fn = CoherenceLoss()

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Mask out second half
        mask = torch.ones(batch_size, seq_len)
        mask[:, seq_len//2:] = 0

        loss = loss_fn(logits, labels, attention_mask=mask)

        assert not torch.isnan(loss)


# ============================================================================
# KLDivergenceLoss Edge Cases
# ============================================================================

class TestKLDivergenceLossEdgeCases:
    """Edge case tests for KLDivergenceLoss."""

    def test_kl_loss_identical_distributions(self):
        """KL divergence of identical distributions is zero."""
        loss_fn = KLDivergenceLoss(beta=1.0)

        logits = torch.randn(2, 10, 100)

        loss = loss_fn(logits, logits.clone())

        assert loss < 0.01  # Near zero

    def test_kl_loss_different_distributions(self):
        """KL divergence of different distributions is positive."""
        loss_fn = KLDivergenceLoss(beta=1.0)

        student = torch.randn(2, 10, 100)
        reference = torch.randn(2, 10, 100)

        loss = loss_fn(student, reference)

        assert loss > 0

    def test_kl_loss_beta_scaling(self):
        """KL loss scales with beta."""
        student = torch.randn(2, 10, 100)
        reference = torch.randn(2, 10, 100)

        loss_fn_1 = KLDivergenceLoss(beta=1.0)
        loss_fn_2 = KLDivergenceLoss(beta=0.5)

        loss_1 = loss_fn_1(student, reference)
        loss_2 = loss_fn_2(student, reference)

        # loss_2 should be ~half of loss_1
        assert abs(loss_2.item() - loss_1.item() * 0.5) < 0.1


# ============================================================================
# CriticLoss Edge Cases
# ============================================================================

class TestCriticLossEdgeCases:
    """Edge case tests for CriticLoss components."""

    def test_critic_score_loss_exact_match(self):
        """CriticScoreLoss is zero for exact match."""
        loss_fn = CriticScoreLoss()
        scores = torch.tensor([7.5, 8.0, 6.5])

        loss = loss_fn(scores, scores.clone())

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_critic_reasoning_loss_normalized(self):
        """CriticReasoningLoss handles unnormalized embeddings."""
        loss_fn = CriticReasoningLoss()

        # Large magnitude embeddings
        pred = torch.randn(4, 256) * 100
        target = torch.randn(4, 256) * 100

        loss = loss_fn(pred, target)

        # Should handle large values without overflow
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_critic_contrastive_temperature(self):
        """CriticContrastiveLoss temperature affects scaling."""
        pred = torch.randn(4, 256)
        target = pred + torch.randn(4, 256) * 0.1

        loss_low_temp = CriticContrastiveLoss(temperature=0.01)
        loss_high_temp = CriticContrastiveLoss(temperature=1.0)

        l1 = loss_low_temp(pred, target)
        l2 = loss_high_temp(pred, target)

        # Both should be valid
        assert not torch.isnan(l1)
        assert not torch.isnan(l2)


# ============================================================================
# AspireLoss Combined Edge Cases
# ============================================================================

class TestAspireLossEdgeCases:
    """Edge case tests for combined AspireLoss."""

    def test_aspire_loss_minimal_inputs(self):
        """AspireLoss with only required inputs."""
        loss_fn = AspireLoss()

        scores = torch.tensor([5.0])

        result = loss_fn.forward(
            critic_predicted_score=scores,
            teacher_score=scores.clone(),
            train_critic=True,
            train_student=True,
        )

        assert "total" in result
        assert not torch.isnan(result["total"])

    def test_aspire_loss_all_zeros(self):
        """AspireLoss with zero scores."""
        loss_fn = AspireLoss()

        zeros = torch.zeros(4)

        result = loss_fn.forward(
            critic_predicted_score=zeros,
            teacher_score=zeros.clone(),
            train_critic=True,
            train_student=True,
        )

        assert not torch.isnan(result["total"])

    def test_aspire_loss_large_batch(self):
        """AspireLoss with large batch size."""
        loss_fn = AspireLoss()

        large_batch = torch.randn(128) * 10  # Large batch, random scores
        large_embed = torch.randn(128, 768)

        result = loss_fn.forward(
            critic_predicted_score=large_batch.abs(),  # Positive scores
            teacher_score=large_batch.abs().clone(),
            critic_predicted_embedding=large_embed,
            teacher_reasoning_embedding=large_embed.clone(),
            train_critic=True,
            train_student=True,
        )

        assert not torch.isnan(result["total"])

    def test_aspire_loss_mixed_precision_bf16(self):
        """AspireLoss works with bfloat16 inputs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for bf16 test")

        loss_fn = AspireLoss()

        scores = torch.randn(4, dtype=torch.bfloat16, device="cuda")

        result = loss_fn.forward(
            critic_predicted_score=scores.abs() * 10,
            teacher_score=scores.abs().clone() * 10,
            train_critic=True,
            train_student=False,
        )

        assert not torch.isnan(result["total"])


# ============================================================================
# Windows Compatibility Entry Point
# ============================================================================

if __name__ == "__main__":
    freeze_support()
    pytest.main([__file__, "-v"])
