"""
Shared encoder critic - shares encoder with student, separate heads.

A balanced architecture: shares the heavy encoder computation with
the student model, but has separate heads for critic-specific predictions.
Good trade-off between efficiency and capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from aspire.critic.base import BaseCritic, CriticOutput


class SharedEncoderCritic(BaseCritic):
    """
    Critic that shares encoder with student but has separate heads.

    Architecture:
    - Uses student's encoder (shared weights)
    - Adds separate pooling and prediction heads
    - Can optionally add adapter layers for critic-specific features

    Balanced trade-off: efficient computation sharing, but critic can
    develop some independent representations.
    """

    def __init__(
        self,
        student_model: nn.Module,
        hidden_dim: int = 512,
        reasoning_dim: int = 768,
        num_dimensions: int = 0,
        dropout: float = 0.1,
        use_adapters: bool = False,
        adapter_dim: int = 64,
    ):
        # Get student hidden size
        if hasattr(student_model.config, "hidden_size"):
            student_hidden_size = student_model.config.hidden_size
        else:
            # Try to infer from model
            student_hidden_size = student_model.get_input_embeddings().embedding_dim

        super().__init__(
            hidden_dim=hidden_dim,
            score_dim=1,
            reasoning_dim=reasoning_dim,
            num_dimensions=num_dimensions,
        )

        self.student_model = student_model
        self.use_adapters = use_adapters
        self.student_hidden_size = student_hidden_size

        # Optional: Adapter layers for critic-specific processing
        if use_adapters:
            self.adapter = nn.Sequential(
                nn.Linear(student_hidden_size, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, student_hidden_size),
                nn.Dropout(dropout),
            )
            # Initialize adapter to be close to identity at start
            nn.init.zeros_(self.adapter[0].weight)
            nn.init.zeros_(self.adapter[2].weight)

        # Attention pooling
        self.pool_attention = nn.Sequential(
            nn.Linear(student_hidden_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Projection and heads
        self.projection = nn.Sequential(
            nn.Linear(student_hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.reasoning_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, reasoning_dim),
        )

        if num_dimensions > 0:
            self.dimension_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim // 2, 1),
                    )
                    for _ in range(num_dimensions)
                ]
            )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        use_student_forward: bool = True,
        **kwargs,
    ) -> CriticOutput:
        """
        Forward pass.

        Args:
            input_ids: Token IDs for encoding
            attention_mask: Attention mask
            hidden_states: Pre-computed hidden states (skips encoding)
            use_student_forward: Whether to run student forward (or use provided hidden_states)
        """

        # Get hidden states from student if not provided
        if hidden_states is None:
            if input_ids is None:
                raise ValueError("Need either input_ids or hidden_states")

            with torch.no_grad():  # Don't backprop through student during critic forward
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]  # Last layer

        # Apply adapter if enabled
        if self.use_adapters:
            hidden_states = hidden_states + self.adapter(hidden_states)

        # Attention-weighted pooling
        attn_weights = self.pool_attention(hidden_states).squeeze(-1)  # [B, S]

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)  # [B, S]
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # [B, H]

        # Project
        features = self.projection(pooled)

        # Predictions
        score = torch.sigmoid(self.score_head(features)) * 10.0

        reasoning = self.reasoning_head(features)
        reasoning = F.normalize(reasoning, p=2, dim=-1)

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
            attentions=attn_weights,
        )

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get trainable parameters (excludes student model)."""
        params = []

        if self.use_adapters:
            params.extend(self.adapter.parameters())

        params.extend(self.pool_attention.parameters())
        params.extend(self.projection.parameters())
        params.extend(self.score_head.parameters())
        params.extend(self.reasoning_head.parameters())

        if self.num_dimensions > 0:
            for head in self.dimension_heads:
                params.extend(head.parameters())

        return params

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get attention weights for interpretability."""
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            return output.attentions
