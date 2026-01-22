"""
Critic head - lightweight MLP on top of student's hidden states.

This is the most efficient architecture: the student model does the
heavy lifting of encoding, and the critic is just a small head that
learns to predict teacher judgment from those representations.

Memory efficient, fast, but limited to what the student can represent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from aspire.critic.base import BaseCritic, CriticOutput


class CriticHead(BaseCritic):
    """
    Lightweight critic head that sits on top of student's hidden states.

    Architecture:
    - Takes last hidden state from student model
    - Pools across sequence (mean or attention-weighted)
    - MLP layers to predict score and reasoning embedding

    Most efficient option: minimal additional parameters, shared compute
    with student forward pass.
    """

    def __init__(
        self,
        input_dim: int,  # Student's hidden dim
        hidden_dim: int = 512,
        num_layers: int = 2,
        reasoning_dim: int = 768,
        num_dimensions: int = 0,
        dropout: float = 0.1,
        pooling: str = "mean",  # "mean", "last", "attention"
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            score_dim=1,
            reasoning_dim=reasoning_dim,
            num_dimensions=num_dimensions,
        )

        self.input_dim = input_dim
        self.dropout = dropout
        self.pooling = pooling

        # Pooling attention (if using attention pooling)
        if pooling == "attention":
            self.pool_attention = nn.Linear(input_dim, 1)

        # MLP layers
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output heads
        self.score_head = nn.Linear(hidden_dim, 1)
        self.reasoning_head = nn.Linear(hidden_dim, reasoning_dim)

        # Optional: per-dimension score heads
        if num_dimensions > 0:
            self.dimension_heads = nn.ModuleList(
                [nn.Linear(hidden_dim, 1) for _ in range(num_dimensions)]
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pool hidden states across sequence dimension."""

        if self.pooling == "last":
            # Use last token (works well for causal LMs)
            if attention_mask is not None:
                # Find last non-padded position
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.size(0)
                return hidden_states[
                    torch.arange(batch_size, device=hidden_states.device), seq_lengths
                ]
            else:
                return hidden_states[:, -1, :]

        elif self.pooling == "mean":
            # Mean pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                return hidden_states.mean(dim=1)

        elif self.pooling == "attention":
            # Attention-weighted pooling
            attn_weights = self.pool_attention(hidden_states).squeeze(-1)  # [B, S]

            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=-1)  # [B, S]
            return torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # [B, H]

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> CriticOutput:
        """
        Forward pass.

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] from student model
            attention_mask: [batch_size, seq_len] attention mask

        Returns:
            CriticOutput with score and reasoning embedding
        """
        if hidden_states is None:
            raise ValueError("CriticHead requires hidden_states from student model")

        # Pool across sequence
        pooled = self._pool(hidden_states, attention_mask)  # [B, input_dim]

        # MLP
        features = self.mlp(pooled)  # [B, hidden_dim]

        # Score prediction (sigmoid to bound 0-10)
        score = torch.sigmoid(self.score_head(features)) * 10.0  # [B, 1]

        # Reasoning embedding
        reasoning = self.reasoning_head(features)  # [B, reasoning_dim]
        reasoning = F.normalize(reasoning, p=2, dim=-1)  # L2 normalize

        # Optional dimension scores
        dimension_scores = None
        if self.num_dimensions > 0:
            dimension_scores = {}
            for i, head in enumerate(self.dimension_heads):
                dim_score = torch.sigmoid(head(features)) * 10.0
                dimension_scores[f"dim_{i}"] = dim_score

        return CriticOutput(
            score=score.squeeze(-1),
            reasoning_embedding=reasoning,
            dimension_scores=dimension_scores,
            hidden_states=features,
        )

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get all trainable parameters."""
        return list(self.parameters())


class MultiHeadCriticHead(CriticHead):
    """
    Critic head with multiple attention heads for richer pooling.

    Uses multi-head attention to capture different aspects of the response,
    then combines them for final prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        reasoning_dim: int = 768,
        num_dimensions: int = 0,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        # Initialize with attention pooling
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            reasoning_dim=reasoning_dim,
            num_dimensions=num_dimensions,
            dropout=dropout,
            pooling="attention",  # Will be overridden
        )

        self.num_heads = num_heads

        # Multi-head attention pooling
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable query for pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, input_dim))

    def _pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Multi-head attention pooling."""

        batch_size = hidden_states.size(0)

        # Expand query for batch
        query = self.pool_query.expand(batch_size, -1, -1)

        # Create key padding mask if needed
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # True = ignore

        # Attention pooling
        pooled, _ = self.multihead_attn(
            query=query,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
        )

        return pooled.squeeze(1)  # [B, input_dim]
