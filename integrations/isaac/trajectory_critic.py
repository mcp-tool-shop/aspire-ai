"""
Trajectory Critic - Learns to predict teacher judgments of robot motion.

The critic observes state-action trajectories and predicts:
1. Quality score (what would the teacher rate this?)
2. Reasoning (why would the teacher give this score?)
3. Improvement (what action would be better?)

After training, the critic enables self-refinement without teacher API calls.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CriticArchitecture, CriticConfig


@dataclass
class CriticOutput:
    """Output from the trajectory critic."""

    # Predicted teacher score (0-10)
    score: torch.Tensor  # (batch,)

    # Predicted reasoning embedding (for distillation)
    reasoning_embedding: torch.Tensor | None = None  # (batch, hidden_dim)

    # Suggested action improvement
    action_improvement: torch.Tensor | None = None  # (batch, seq, action_dim)

    # Per-timestep scores (where are the problems?)
    timestep_scores: torch.Tensor | None = None  # (batch, seq)

    # Dimension-specific scores
    dimension_scores: dict[str, torch.Tensor] | None = None

    # Attention weights (for interpretability)
    attention_weights: torch.Tensor | None = None  # (batch, heads, seq, seq)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TrajectoryEncoder(nn.Module):
    """
    Encodes state-action trajectories into a representation.

    Supports multiple architectures:
    - Transformer: Best for long trajectories, captures global dependencies
    - LSTM: Good for sequential processing
    - TCN: Efficient temporal convolutions
    - MLP: Simple baseline for short trajectories
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 256,
        architecture: CriticArchitecture = CriticArchitecture.TRANSFORMER,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture

        input_dim = state_dim + action_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if architecture == CriticArchitecture.TRANSFORMER:
            self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            self._needs_mask = True

        elif architecture == CriticArchitecture.LSTM:
            self.encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,  # Bidirectional doubles this
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True,
            )
            self._needs_mask = False

        elif architecture == CriticArchitecture.TCN:
            # Temporal Convolutional Network
            layers = []
            for i in range(num_layers):
                dilation = 2 ** i
                layers.append(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    )
                )
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            self.encoder = nn.Sequential(*layers)
            self._needs_mask = False

        elif architecture == CriticArchitecture.MLP:
            # Flatten and process with MLP
            self.encoder = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(hidden_dim * max_seq_len, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self._needs_mask = False
            self.max_seq_len = max_seq_len

        # Output layer norm
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Encode a trajectory.

        Args:
            states: (batch, seq, state_dim)
            actions: (batch, seq, action_dim)
            mask: (batch, seq) - True for valid timesteps

        Returns:
            sequence_encoding: (batch, seq, hidden_dim) or (batch, hidden_dim) for MLP
            attention_weights: (batch, heads, seq, seq) for transformer, else None
        """
        batch, seq, _ = states.shape

        # Concatenate state and action
        x = torch.cat([states, actions], dim=-1)  # (batch, seq, state+action)
        x = self.input_proj(x)  # (batch, seq, hidden)

        attention_weights = None

        if self.architecture == CriticArchitecture.TRANSFORMER:
            x = self.pos_encoding(x)

            # Create attention mask if needed
            if mask is not None and self._needs_mask:
                # Convert to attention mask format
                attn_mask = ~mask  # Transformer uses True for masked positions
            else:
                attn_mask = None

            x = self.encoder(x, src_key_padding_mask=attn_mask)

        elif self.architecture == CriticArchitecture.LSTM:
            if mask is not None:
                # Pack padded sequence
                lengths = mask.sum(dim=1).cpu()
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                x, _ = self.encoder(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            else:
                x, _ = self.encoder(x)

        elif self.architecture == CriticArchitecture.TCN:
            # TCN expects (batch, channels, seq)
            x = x.transpose(1, 2)
            x = self.encoder(x)
            x = x.transpose(1, 2)

        elif self.architecture == CriticArchitecture.MLP:
            # Pad to max_seq_len
            if seq < self.max_seq_len:
                padding = torch.zeros(
                    batch, self.max_seq_len - seq, self.hidden_dim,
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=1)
            elif seq > self.max_seq_len:
                x = x[:, :self.max_seq_len]

            x = self.encoder(x)  # (batch, hidden)
            x = x.unsqueeze(1)   # (batch, 1, hidden) for consistency

        x = self.output_norm(x)
        return x, attention_weights


class MotionCriticHead(nn.Module):
    """
    Prediction heads for the trajectory critic.

    Predicts:
    - Overall quality score
    - Per-dimension scores (safety, efficiency, smoothness)
    - Per-timestep scores (where are problems?)
    - Action improvements (what would be better?)
    - Reasoning embedding (for distillation from teacher)
    """

    DIMENSIONS = ["safety", "efficiency", "smoothness", "goal_achievement"]

    def __init__(
        self,
        hidden_dim: int = 256,
        action_dim: int = 0,
        predict_score: bool = True,
        predict_dimensions: bool = True,
        predict_timesteps: bool = True,
        predict_improvement: bool = True,
        predict_reasoning: bool = True,
        reasoning_dim: int = 256,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Score prediction (0-10)
        if predict_score:
            self.score_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),  # Output in [0, 1], scale to [0, 10]
            )
        else:
            self.score_head = None

        # Per-dimension scores
        if predict_dimensions:
            self.dimension_heads = nn.ModuleDict({
                dim: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid(),
                )
                for dim in self.DIMENSIONS
            })
        else:
            self.dimension_heads = None

        # Per-timestep scores
        if predict_timesteps:
            self.timestep_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.timestep_head = None

        # Action improvement prediction
        if predict_improvement and action_dim > 0:
            self.improvement_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),  # Bounded action correction
            )
        else:
            self.improvement_head = None

        # Reasoning embedding (for distillation)
        if predict_reasoning:
            self.reasoning_head = nn.Sequential(
                nn.Linear(hidden_dim, reasoning_dim),
                nn.LayerNorm(reasoning_dim),
            )
        else:
            self.reasoning_head = None

    def forward(
        self,
        encoding: torch.Tensor,
        pool: Literal["mean", "cls", "last"] = "mean",
    ) -> CriticOutput:
        """
        Predict from trajectory encoding.

        Args:
            encoding: (batch, seq, hidden) or (batch, hidden)
            pool: How to pool sequence for global predictions

        Returns:
            CriticOutput with all predictions
        """
        # Handle both sequence and pooled inputs
        if encoding.dim() == 3:
            batch, seq, hidden = encoding.shape

            # Pool for global predictions
            if pool == "mean":
                pooled = encoding.mean(dim=1)
            elif pool == "cls":
                pooled = encoding[:, 0]
            elif pool == "last":
                pooled = encoding[:, -1]
            else:
                raise ValueError(f"Unknown pool: {pool}")

            has_sequence = True
        else:
            pooled = encoding
            has_sequence = False

        # Score prediction
        if self.score_head is not None:
            score = self.score_head(pooled).squeeze(-1) * 10.0  # Scale to [0, 10]
        else:
            score = None

        # Dimension scores
        if self.dimension_heads is not None:
            dimension_scores = {
                dim: head(pooled).squeeze(-1) * 10.0
                for dim, head in self.dimension_heads.items()
            }
        else:
            dimension_scores = None

        # Timestep scores
        if self.timestep_head is not None and has_sequence:
            timestep_scores = self.timestep_head(encoding).squeeze(-1) * 10.0
        else:
            timestep_scores = None

        # Action improvement
        if self.improvement_head is not None and has_sequence:
            action_improvement = self.improvement_head(encoding)
        else:
            action_improvement = None

        # Reasoning embedding
        if self.reasoning_head is not None:
            reasoning_embedding = self.reasoning_head(pooled)
        else:
            reasoning_embedding = None

        return CriticOutput(
            score=score,
            dimension_scores=dimension_scores,
            timestep_scores=timestep_scores,
            action_improvement=action_improvement,
            reasoning_embedding=reasoning_embedding,
        )


class TrajectoryCritic(nn.Module):
    """
    Complete trajectory critic model.

    Combines encoder and prediction heads to evaluate robot motion
    and predict what the teacher would think.
    """

    def __init__(self, config: CriticConfig | None = None):
        super().__init__()

        if config is None:
            config = CriticConfig()

        self.config = config

        # Build encoder
        self.encoder = TrajectoryEncoder(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_seq_len=config.max_trajectory_len,
            architecture=config.architecture,
            dropout=config.dropout,
        )

        # Build prediction heads
        self.heads = MotionCriticHead(
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            predict_score=config.predict_score,
            predict_dimensions=True,
            predict_timesteps=True,
            predict_improvement=config.predict_improvement,
            predict_reasoning=config.predict_reasoning,
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> CriticOutput:
        """
        Evaluate a trajectory.

        Args:
            states: (batch, seq, state_dim)
            actions: (batch, seq, action_dim)
            mask: (batch, seq) - True for valid timesteps

        Returns:
            CriticOutput with score, reasoning, and improvement suggestions
        """
        # Encode trajectory
        encoding, attention = self.encoder(states, actions, mask)

        # Predict
        output = self.heads(encoding)
        output.attention_weights = attention

        return output

    def get_action_improvement(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get suggested action improvements.

        Returns the delta to add to actions for better motion.
        """
        output = self.forward(states, actions, mask)
        if output.action_improvement is None:
            return torch.zeros_like(actions)
        return output.action_improvement

    def score_trajectory(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get just the quality score for a trajectory.
        """
        output = self.forward(states, actions, mask)
        return output.score


class TrajectoryCriticLoss(nn.Module):
    """
    Loss function for training the trajectory critic.

    Trains the critic to predict teacher judgments:
    - Score prediction loss (MSE)
    - Dimension score losses
    - Reasoning distillation (cosine similarity)
    - Timestep-level losses
    """

    def __init__(
        self,
        score_weight: float = 1.0,
        dimension_weight: float = 0.5,
        reasoning_weight: float = 0.3,
        timestep_weight: float = 0.2,
    ):
        super().__init__()

        self.score_weight = score_weight
        self.dimension_weight = dimension_weight
        self.reasoning_weight = reasoning_weight
        self.timestep_weight = timestep_weight

        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        critic_output: CriticOutput,
        teacher_score: torch.Tensor,
        teacher_dimensions: dict[str, torch.Tensor] | None = None,
        teacher_reasoning_embedding: torch.Tensor | None = None,
        teacher_timestep_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute critic training loss.

        Args:
            critic_output: Predictions from critic
            teacher_score: Ground truth overall score (batch,)
            teacher_dimensions: Ground truth dimension scores
            teacher_reasoning_embedding: Teacher's reasoning embedding
            teacher_timestep_scores: Per-timestep ground truth

        Returns:
            Dictionary with individual losses and total
        """
        losses = {}

        # Score prediction loss
        if critic_output.score is not None:
            losses["score"] = self.mse(critic_output.score, teacher_score)
        else:
            losses["score"] = torch.tensor(0.0)

        # Dimension score losses
        if (
            critic_output.dimension_scores is not None
            and teacher_dimensions is not None
        ):
            dim_losses = []
            for dim, pred in critic_output.dimension_scores.items():
                if dim in teacher_dimensions:
                    dim_losses.append(self.mse(pred, teacher_dimensions[dim]))

            if dim_losses:
                losses["dimensions"] = torch.stack(dim_losses).mean()
            else:
                losses["dimensions"] = torch.tensor(0.0)
        else:
            losses["dimensions"] = torch.tensor(0.0)

        # Reasoning distillation
        if (
            critic_output.reasoning_embedding is not None
            and teacher_reasoning_embedding is not None
        ):
            # Cosine similarity loss (1 - similarity)
            similarity = self.cosine(
                critic_output.reasoning_embedding,
                teacher_reasoning_embedding,
            )
            losses["reasoning"] = (1.0 - similarity).mean()
        else:
            losses["reasoning"] = torch.tensor(0.0)

        # Timestep-level loss
        if (
            critic_output.timestep_scores is not None
            and teacher_timestep_scores is not None
        ):
            losses["timesteps"] = self.mse(
                critic_output.timestep_scores,
                teacher_timestep_scores,
            )
        else:
            losses["timesteps"] = torch.tensor(0.0)

        # Total weighted loss
        losses["total"] = (
            self.score_weight * losses["score"]
            + self.dimension_weight * losses["dimensions"]
            + self.reasoning_weight * losses["reasoning"]
            + self.timestep_weight * losses["timesteps"]
        )

        return losses
