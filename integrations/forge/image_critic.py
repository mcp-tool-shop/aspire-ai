"""
Image Critic Model for ASPIRE.

The critic learns to predict what the vision teacher would think about
generated images. After training, it provides aesthetic guidance without
requiring API calls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ImageCriticOutput:
    """Output from the image critic."""

    # Overall aesthetic score (0-10)
    overall_score: torch.Tensor

    # Per-dimension scores
    aesthetic_score: torch.Tensor | None = None
    composition_score: torch.Tensor | None = None
    color_score: torch.Tensor | None = None
    lighting_score: torch.Tensor | None = None
    style_score: torch.Tensor | None = None
    prompt_adherence_score: torch.Tensor | None = None

    # Reasoning embedding (for alignment with teacher)
    reasoning_embedding: torch.Tensor | None = None

    # Hidden features
    features: torch.Tensor | None = None


class CLIPImageCritic(nn.Module):
    """
    Critic that uses CLIP image embeddings to predict aesthetic quality.

    Lightweight architecture: CLIP encoder is frozen, only the prediction
    heads are trained. Fast and memory efficient.
    """

    def __init__(
        self,
        clip_model,  # CLIP vision encoder from the SD model
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_dimensions: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.clip_model = clip_model
        self.clip_dim = clip_model.config.hidden_size if hasattr(clip_model, 'config') else 768

        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # MLP projection
        layers = []
        current_dim = self.clip_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Score prediction heads
        self.overall_head = nn.Linear(hidden_dim, 1)

        # Per-dimension heads
        self.dimension_heads = nn.ModuleDict({
            "aesthetic": nn.Linear(hidden_dim, 1),
            "composition": nn.Linear(hidden_dim, 1),
            "color": nn.Linear(hidden_dim, 1),
            "lighting": nn.Linear(hidden_dim, 1),
            "style": nn.Linear(hidden_dim, 1),
            "prompt_adherence": nn.Linear(hidden_dim, 1),
        })

        # Reasoning embedding head (for teacher alignment)
        self.reasoning_head = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        images: torch.Tensor,  # [B, C, H, W] normalized images
        return_features: bool = False,
    ) -> ImageCriticOutput:
        """
        Forward pass.

        Args:
            images: Batch of images [B, 3, 224, 224] (CLIP input size)
            return_features: Whether to return intermediate features

        Returns:
            ImageCriticOutput with scores and optional features
        """
        # Get CLIP embeddings
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(images)  # [B, clip_dim]

        # MLP projection
        features = self.mlp(clip_features)  # [B, hidden_dim]

        # Overall score (sigmoid * 10 to get 0-10 range)
        overall = torch.sigmoid(self.overall_head(features)) * 10.0

        # Per-dimension scores
        dim_scores = {}
        for name, head in self.dimension_heads.items():
            dim_scores[name] = torch.sigmoid(head(features)) * 10.0

        # Reasoning embedding (L2 normalized)
        reasoning = self.reasoning_head(features)
        reasoning = F.normalize(reasoning, p=2, dim=-1)

        return ImageCriticOutput(
            overall_score=overall.squeeze(-1),
            aesthetic_score=dim_scores["aesthetic"].squeeze(-1),
            composition_score=dim_scores["composition"].squeeze(-1),
            color_score=dim_scores["color"].squeeze(-1),
            lighting_score=dim_scores["lighting"].squeeze(-1),
            style_score=dim_scores["style"].squeeze(-1),
            prompt_adherence_score=dim_scores["prompt_adherence"].squeeze(-1),
            reasoning_embedding=reasoning,
            features=features if return_features else None,
        )

    def predict_score(self, images: torch.Tensor) -> float:
        """Convenience method to get overall score."""
        self.eval()
        with torch.no_grad():
            output = self.forward(images)
            return output.overall_score.mean().item()

    def get_trainable_parameters(self):
        """Get parameters that should be trained (excludes frozen CLIP)."""
        params = []
        params.extend(self.mlp.parameters())
        params.extend(self.overall_head.parameters())
        for head in self.dimension_heads.values():
            params.extend(head.parameters())
        params.extend(self.reasoning_head.parameters())
        return params


class LatentCritic(nn.Module):
    """
    Critic that operates on diffusion latents directly.

    More efficient as it doesn't require decoding to image space.
    Can be integrated into the sampling loop for guidance.
    """

    def __init__(
        self,
        latent_channels: int = 4,  # SD latent channels
        latent_size: int = 64,     # Typical latent size for 512x512
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Convolutional encoder for latents
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # /2
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # /4
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, hidden_dim, 3, stride=2, padding=1),  # /8
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # Global pool
        )

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Score heads
        self.overall_head = nn.Linear(hidden_dim, 1)
        self.reasoning_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        latents: torch.Tensor,  # [B, 4, H, W] diffusion latents
        timestep: torch.Tensor | None = None,  # Optional timestep embedding
    ) -> ImageCriticOutput:
        """
        Forward pass on latents.

        Args:
            latents: Diffusion latents
            timestep: Optional timestep for conditioning

        Returns:
            ImageCriticOutput
        """
        # Encode latents
        features = self.conv_encoder(latents)  # [B, hidden_dim, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, hidden_dim]

        # MLP
        features = self.mlp(features)

        # Scores
        overall = torch.sigmoid(self.overall_head(features)) * 10.0

        # Reasoning
        reasoning = F.normalize(self.reasoning_head(features), p=2, dim=-1)

        return ImageCriticOutput(
            overall_score=overall.squeeze(-1),
            reasoning_embedding=reasoning,
            features=features,
        )

    def get_guidance_scale(
        self,
        latents: torch.Tensor,
        target_score: float = 8.0,
    ) -> torch.Tensor:
        """
        Compute guidance scale based on current quality prediction.

        Higher guidance when quality is low, lower when quality is good.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(latents)
            current_score = output.overall_score

            # Scale: more guidance needed when score is low
            guidance = (target_score - current_score) / target_score
            guidance = torch.clamp(guidance, 0.0, 1.0)

            return guidance


class ImageCriticLoss(nn.Module):
    """Loss function for training the image critic."""

    def __init__(
        self,
        score_weight: float = 1.0,
        dimension_weight: float = 0.5,
        reasoning_weight: float = 0.3,
    ):
        super().__init__()
        self.score_weight = score_weight
        self.dimension_weight = dimension_weight
        self.reasoning_weight = reasoning_weight

    def forward(
        self,
        predicted: ImageCriticOutput,
        target_overall: torch.Tensor,
        target_dimensions: dict[str, torch.Tensor] | None = None,
        target_reasoning: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute critic loss.

        Args:
            predicted: Critic output
            target_overall: Teacher's overall score
            target_dimensions: Teacher's per-dimension scores
            target_reasoning: Teacher's reasoning embedding

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # Overall score loss
        losses["overall"] = F.smooth_l1_loss(
            predicted.overall_score, target_overall
        )

        # Per-dimension losses
        if target_dimensions and predicted.aesthetic_score is not None:
            dim_losses = []
            for name, target in target_dimensions.items():
                pred = getattr(predicted, f"{name}_score", None)
                if pred is not None:
                    dim_losses.append(F.smooth_l1_loss(pred, target))

            if dim_losses:
                losses["dimensions"] = torch.stack(dim_losses).mean()

        # Reasoning alignment loss
        if target_reasoning is not None and predicted.reasoning_embedding is not None:
            # Cosine similarity loss
            cos_sim = F.cosine_similarity(
                predicted.reasoning_embedding, target_reasoning, dim=-1
            )
            losses["reasoning"] = 1.0 - cos_sim.mean()

        # Total loss
        total = self.score_weight * losses["overall"]

        if "dimensions" in losses:
            total = total + self.dimension_weight * losses["dimensions"]

        if "reasoning" in losses:
            total = total + self.reasoning_weight * losses["reasoning"]

        losses["total"] = total

        return losses
