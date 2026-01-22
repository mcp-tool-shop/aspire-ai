"""
Edge case tests for ASPIRE critic architectures.

Tests for boundary conditions and edge cases:
- Empty/single token sequences
- Large batch sizes
- Gradient stability
- Pooling edge cases
- Device compatibility

Windows compatibility notes:
- Use num_workers=0 in DataLoader tests
- Use if __name__ == "__main__": freeze_support() pattern
"""

from __future__ import annotations

from multiprocessing import freeze_support

import pytest
import torch
import torch.nn as nn

from aspire.critic.base import BaseCritic, CriticOutput
from aspire.critic.head import CriticHead, MultiHeadCriticHead


# ============================================================================
# CriticHead Pooling Edge Cases
# ============================================================================

class TestCriticHeadPooling:
    """Edge case tests for CriticHead pooling strategies."""

    def test_mean_pooling_single_token(self):
        """Mean pooling with sequence length 1."""
        head = CriticHead(input_dim=256, hidden_dim=128, pooling="mean")
        hidden = torch.randn(2, 1, 256)  # Single token

        output = head(hidden_states=hidden)

        assert output.score.shape == (2,)
        assert not torch.isnan(output.score).any()

    def test_last_pooling_single_token(self):
        """Last pooling with sequence length 1."""
        head = CriticHead(input_dim=256, hidden_dim=128, pooling="last")
        hidden = torch.randn(2, 1, 256)

        output = head(hidden_states=hidden)

        assert output.score.shape == (2,)
        assert not torch.isnan(output.score).any()

    def test_attention_pooling_single_token(self):
        """Attention pooling with sequence length 1."""
        head = CriticHead(input_dim=256, hidden_dim=128, pooling="attention")
        hidden = torch.randn(2, 1, 256)

        output = head(hidden_states=hidden)

        assert output.score.shape == (2,)
        assert not torch.isnan(output.score).any()

    def test_mean_pooling_with_all_masked(self):
        """Mean pooling where attention mask is nearly all zeros."""
        head = CriticHead(input_dim=256, hidden_dim=128, pooling="mean")
        hidden = torch.randn(2, 10, 256)
        # Only first token is valid
        mask = torch.zeros(2, 10)
        mask[:, 0] = 1

        output = head(hidden_states=hidden, attention_mask=mask)

        assert not torch.isnan(output.score).any()

    def test_last_pooling_with_mask(self):
        """Last pooling correctly finds last valid token."""
        head = CriticHead(input_dim=256, hidden_dim=128, pooling="last")
        hidden = torch.randn(2, 10, 256)

        # Different sequence lengths per batch item (must be long type for indexing)
        mask = torch.zeros(2, 10, dtype=torch.long)
        mask[0, :5] = 1  # First item: 5 tokens
        mask[1, :8] = 1  # Second item: 8 tokens

        output = head(hidden_states=hidden, attention_mask=mask)

        assert output.score.shape == (2,)

    def test_attention_pooling_with_full_mask(self):
        """Attention pooling handles fully padded sequences gracefully."""
        head = CriticHead(input_dim=256, hidden_dim=128, pooling="attention")
        hidden = torch.randn(2, 10, 256)
        mask = torch.ones(2, 10)  # All valid

        output = head(hidden_states=hidden, attention_mask=mask)

        assert not torch.isnan(output.score).any()


# ============================================================================
# CriticHead Dimension Tests
# ============================================================================

class TestCriticHeadDimensions:
    """Tests for CriticHead with dimension scoring."""

    def test_dimension_scores_created(self):
        """Dimension heads are created when num_dimensions > 0."""
        head = CriticHead(input_dim=256, hidden_dim=128, num_dimensions=5)

        assert hasattr(head, "dimension_heads")
        assert len(head.dimension_heads) == 5

    def test_dimension_scores_output(self):
        """Forward returns dimension scores when configured."""
        head = CriticHead(input_dim=256, hidden_dim=128, num_dimensions=3)
        hidden = torch.randn(2, 10, 256)

        output = head(hidden_states=hidden)

        assert output.dimension_scores is not None
        assert len(output.dimension_scores) == 3
        assert "dim_0" in output.dimension_scores
        assert "dim_1" in output.dimension_scores
        assert "dim_2" in output.dimension_scores

    def test_dimension_scores_bounded(self):
        """Dimension scores are in [0, 10] range."""
        head = CriticHead(input_dim=256, hidden_dim=128, num_dimensions=3)
        hidden = torch.randn(4, 10, 256)

        output = head(hidden_states=hidden)

        for key, scores in output.dimension_scores.items():
            assert (scores >= 0).all()
            assert (scores <= 10).all()


# ============================================================================
# MultiHeadCriticHead Tests
# ============================================================================

class TestMultiHeadCriticHead:
    """Tests for MultiHeadCriticHead attention pooling."""

    def test_multihead_init(self):
        """MultiHeadCriticHead initializes with attention components."""
        head = MultiHeadCriticHead(input_dim=256, hidden_dim=128, num_heads=4)

        assert head.num_heads == 4
        assert hasattr(head, "multihead_attn")
        assert hasattr(head, "pool_query")

    def test_multihead_forward(self):
        """MultiHeadCriticHead forward pass works."""
        head = MultiHeadCriticHead(input_dim=256, hidden_dim=128, num_heads=4)
        hidden = torch.randn(2, 10, 256)

        output = head(hidden_states=hidden)

        assert output.score.shape == (2,)
        assert output.reasoning_embedding.shape == (2, 768)  # Default reasoning_dim

    def test_multihead_with_mask(self):
        """MultiHeadCriticHead handles attention mask."""
        head = MultiHeadCriticHead(input_dim=256, hidden_dim=128, num_heads=4)
        hidden = torch.randn(2, 10, 256)
        mask = torch.ones(2, 10)
        mask[:, 5:] = 0  # Mask out second half

        output = head(hidden_states=hidden, attention_mask=mask)

        assert not torch.isnan(output.score).any()

    def test_multihead_single_token(self):
        """MultiHeadCriticHead handles single token sequences."""
        head = MultiHeadCriticHead(input_dim=256, hidden_dim=128, num_heads=4)
        hidden = torch.randn(2, 1, 256)

        output = head(hidden_states=hidden)

        assert output.score.shape == (2,)


# ============================================================================
# CriticOutput Edge Cases
# ============================================================================

class TestCriticOutputEdgeCases:
    """Edge case tests for CriticOutput dataclass."""

    def test_critic_output_empty_dimension_scores(self):
        """CriticOutput handles empty dimension_scores dict."""
        score = torch.tensor([5.0])
        reasoning = torch.randn(1, 256)

        output = CriticOutput(
            score=score,
            reasoning_embedding=reasoning,
            dimension_scores={},  # Empty dict
        )

        result = output.to_dict()
        assert result["dimension_scores"] == {}

    def test_critic_output_single_element(self):
        """CriticOutput with batch_size=1."""
        score = torch.tensor([7.5])
        reasoning = torch.randn(1, 256)

        output = CriticOutput(score=score, reasoning_embedding=reasoning)
        result = output.to_dict()

        assert result["score"].shape == (1,)
        assert result["reasoning_embedding"].shape == (1, 256)

    def test_critic_output_large_batch(self):
        """CriticOutput with large batch."""
        batch_size = 128
        score = torch.randn(batch_size) * 10
        reasoning = torch.randn(batch_size, 768)

        output = CriticOutput(score=score, reasoning_embedding=reasoning)
        result = output.to_dict()

        assert result["score"].shape == (batch_size,)


# ============================================================================
# Gradient Flow Tests
# ============================================================================

class TestCriticGradientFlow:
    """Tests for gradient flow through critic components."""

    def test_critic_head_gradients(self):
        """Gradients flow through CriticHead."""
        head = CriticHead(input_dim=256, hidden_dim=128)
        hidden = torch.randn(2, 10, 256, requires_grad=True)

        output = head(hidden_states=hidden)
        loss = output.score.mean()
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape

    def test_multihead_critic_gradients(self):
        """Gradients flow through MultiHeadCriticHead."""
        head = MultiHeadCriticHead(input_dim=256, hidden_dim=128, num_heads=4)
        hidden = torch.randn(2, 10, 256, requires_grad=True)

        output = head(hidden_states=hidden)
        loss = output.score.mean() + output.reasoning_embedding.mean()
        loss.backward()

        assert hidden.grad is not None

    def test_dimension_head_gradients(self):
        """Gradients flow through dimension heads."""
        head = CriticHead(input_dim=256, hidden_dim=128, num_dimensions=3)
        hidden = torch.randn(2, 10, 256, requires_grad=True)

        output = head(hidden_states=hidden)

        # Loss from all outputs
        loss = output.score.mean()
        for scores in output.dimension_scores.values():
            loss = loss + scores.mean()

        loss.backward()

        assert hidden.grad is not None


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestCriticErrorHandling:
    """Tests for error handling in critic components."""

    def test_critic_head_requires_hidden_states(self):
        """CriticHead raises error without hidden_states."""
        head = CriticHead(input_dim=256, hidden_dim=128)

        with pytest.raises(ValueError, match="hidden_states"):
            head(input_ids=torch.randint(0, 100, (2, 10)))

    def test_critic_head_invalid_pooling(self):
        """CriticHead raises error for invalid pooling type."""
        head = CriticHead(input_dim=256, hidden_dim=128, pooling="mean")
        head.pooling = "invalid"  # Force invalid value

        hidden = torch.randn(2, 10, 256)

        with pytest.raises(ValueError, match="Unknown pooling"):
            head(hidden_states=hidden)


# ============================================================================
# Windows Compatibility Entry Point
# ============================================================================

if __name__ == "__main__":
    freeze_support()
    pytest.main([__file__, "-v"])
