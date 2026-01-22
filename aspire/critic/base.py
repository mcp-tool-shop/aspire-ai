"""
Base critic interface.

The critic is the heart of ASPIRE - it learns to predict teacher judgment
and becomes the internalized sense of quality that guides the student.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CriticOutput:
    """Output from the critic model."""

    # Predicted overall score (0-10)
    score: torch.Tensor

    # Predicted reasoning embedding (for alignment loss)
    reasoning_embedding: torch.Tensor | None = None

    # Per-dimension score predictions (optional)
    dimension_scores: dict[str, torch.Tensor] | None = None

    # Hidden states (for potential further processing)
    hidden_states: torch.Tensor | None = None

    # Attention weights (for interpretability)
    attentions: torch.Tensor | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, moving tensors to CPU."""
        return {
            "score": self.score.detach().cpu().numpy(),
            "reasoning_embedding": (
                self.reasoning_embedding.detach().cpu().numpy()
                if self.reasoning_embedding is not None
                else None
            ),
            "dimension_scores": (
                {k: v.detach().cpu().numpy() for k, v in self.dimension_scores.items()}
                if self.dimension_scores is not None
                else None
            ),
        }


class BaseCritic(nn.Module, ABC):
    """
    Abstract base class for critic models.

    The critic predicts what the teacher would think of a response.
    Different architectures trade off between efficiency and capability:

    - CriticHead: Lightweight MLP on student's hidden states (most efficient)
    - SeparateCritic: Independent model (most capable, most memory)
    - SharedEncoderCritic: Shared encoder with separate heads (balanced)

    The critic outputs:
    1. Score prediction (0-10 scale)
    2. Reasoning embedding (for alignment with teacher's reasoning)
    3. Optional dimension-specific scores
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        score_dim: int = 1,
        reasoning_dim: int = 768,
        num_dimensions: int = 0,  # If > 0, predict per-dimension scores
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.score_dim = score_dim
        self.reasoning_dim = reasoning_dim
        self.num_dimensions = num_dimensions

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> CriticOutput:
        """
        Forward pass through the critic.

        Can accept either:
        - input_ids + attention_mask: Full input, critic does its own encoding
        - hidden_states: Pre-computed hidden states from student model

        Returns:
            CriticOutput with score prediction and reasoning embedding
        """
        pass

    @abstractmethod
    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get list of trainable parameters for the optimizer."""
        pass

    def predict_score(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> float:
        """Convenience method to get score as float."""
        self.eval()
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                **kwargs,
            )
            return output.score.item()

    def save(self, path: str) -> None:
        """Save critic state."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "hidden_dim": self.hidden_dim,
                    "score_dim": self.score_dim,
                    "reasoning_dim": self.reasoning_dim,
                    "num_dimensions": self.num_dimensions,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, **kwargs) -> "BaseCritic":
        """Load critic from saved state."""
        checkpoint = torch.load(path)
        config = checkpoint["config"]
        config.update(kwargs)

        instance = cls(**config)
        instance.load_state_dict(checkpoint["state_dict"])
        return instance
