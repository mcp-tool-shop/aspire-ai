"""
Separate critic model - independent model for evaluation.

This is the most capable architecture: the critic has its own encoder
and can develop independent representations. More memory intensive but
potentially more powerful for complex judgment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from aspire.critic.base import BaseCritic, CriticOutput


class SeparateCritic(BaseCritic):
    """
    Separate critic with its own encoder model.

    Architecture:
    - Independent encoder (can be different architecture from student)
    - Full sequence processing
    - MLP heads for score and reasoning prediction

    Most capable but most memory intensive. Good for when you need
    the critic to understand things the student might miss.
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/deberta-v3-base",
        hidden_dim: int = 512,
        reasoning_dim: int = 768,
        num_dimensions: int = 0,
        dropout: float = 0.1,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        freeze_encoder: bool = False,
        device: str = "cuda",
    ):
        # Get encoder hidden size first
        encoder_config = AutoModel.from_pretrained(model_name_or_path).config
        encoder_hidden_size = encoder_config.hidden_size

        super().__init__(
            hidden_dim=hidden_dim,
            score_dim=1,
            reasoning_dim=reasoning_dim,
            num_dimensions=num_dimensions,
        )

        self.model_name_or_path = model_name_or_path
        self.freeze_encoder = freeze_encoder

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load encoder
        if quantization_config:
            self.encoder = AutoModel.from_pretrained(
                model_name_or_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            self.encoder = AutoModel.from_pretrained(model_name_or_path)
            self.encoder = self.encoder.to(device)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection from encoder to hidden dim
        self.projection = nn.Sequential(
            nn.Linear(encoder_hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output heads
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

        # Optional dimension heads
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
        text: str | list[str] | None = None,
        **kwargs,
    ) -> CriticOutput:
        """
        Forward pass.

        Can accept either:
        - input_ids + attention_mask: Pre-tokenized input
        - text: Raw text to tokenize
        - hidden_states: Not used (we have our own encoder)
        """

        # Tokenize if text provided
        if text is not None:
            if isinstance(text, str):
                text = [text]

            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.encoder.device)
            attention_mask = encoded["attention_mask"].to(self.encoder.device)

        if input_ids is None:
            raise ValueError("SeparateCritic requires input_ids or text")

        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use CLS token or mean pooling
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Mean pooling
            last_hidden = outputs.last_hidden_state
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = last_hidden.mean(dim=1)

        # Project
        features = self.projection(pooled)

        # Score prediction
        score = torch.sigmoid(self.score_head(features)) * 10.0

        # Reasoning embedding
        reasoning = self.reasoning_head(features)
        reasoning = F.normalize(reasoning, p=2, dim=-1)

        # Dimension scores
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
        """Get trainable parameters."""
        params = []

        # Encoder params (if not frozen)
        if not self.freeze_encoder:
            params.extend(self.encoder.parameters())

        # Head params
        params.extend(self.projection.parameters())
        params.extend(self.score_head.parameters())
        params.extend(self.reasoning_head.parameters())

        if self.num_dimensions > 0:
            for head in self.dimension_heads:
                params.extend(head.parameters())

        return params

    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        """Convenience method to encode text to features."""
        self.eval()
        with torch.no_grad():
            output = self.forward(text=text)
            return output.hidden_states
