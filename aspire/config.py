"""
Configuration for ASPIRE training.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class StudentConfig(BaseModel):
    """Configuration for the student model being fine-tuned."""

    model_name_or_path: str = "microsoft/Phi-3-mini-4k-instruct"
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    use_gradient_checkpointing: bool = True
    max_length: int = 2048

    # LoRA config
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class CriticConfig(BaseModel):
    """Configuration for the critic model (internalized teacher judgment)."""

    # Critic architecture type
    architecture: Literal["head", "separate", "shared_encoder"] = "head"

    # If "head": adds MLP head on top of student's hidden states
    head_hidden_dim: int = 512
    head_num_layers: int = 2

    # If "separate": uses a separate small model
    separate_model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    separate_load_in_4bit: bool = True

    # Output dimensions
    score_output_dim: int = 1  # Single score prediction
    reasoning_embedding_dim: int = 768  # For reasoning alignment


class TeacherConfig(BaseModel):
    """Configuration for teacher model(s)."""

    # Default teacher
    default_teacher: str = "claude"

    # Teacher-specific settings
    claude_model: str = "claude-sonnet-4-20250514"
    openai_model: str = "gpt-4o"
    local_model_path: str | None = None

    # Generation settings
    max_tokens: int = 1024
    temperature: float = 0.7

    # Dialogue settings
    max_dialogue_turns: int = 3
    challenge_types: list[str] = Field(
        default_factory=lambda: [
            "probe_reasoning",
            "edge_case",
            "devils_advocate",
            "socratic",
            "clarification",
        ]
    )


class CurriculumConfig(BaseModel):
    """Configuration for curriculum-based learning."""

    # Curriculum stages
    stages: list[str] = Field(
        default_factory=lambda: [
            "foundation",
            "reasoning",
            "nuance",
            "adversarial",
            "transfer",
        ]
    )

    # When to advance stages (based on critic accuracy)
    stage_advancement_threshold: float = 0.8
    min_steps_per_stage: int = 500

    # Curriculum data paths
    data_dir: Path = Path("data/curriculum")


class LossConfig(BaseModel):
    """Configuration for loss functions."""

    # Critic loss weights
    critic_score_weight: float = 1.0
    critic_reasoning_weight: float = 0.5

    # Student loss weights
    student_reward_weight: float = 1.0
    student_contrastive_weight: float = 0.5
    student_trajectory_weight: float = 0.3
    student_coherence_weight: float = 0.2

    # Contrastive loss settings
    contrastive_margin: float = 0.5
    contrastive_temperature: float = 0.07


class TrainingConfig(BaseModel):
    """Configuration for the training loop."""

    # Basic training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    critic_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Optimization
    optimizer: Literal["adamw", "adamw_8bit", "paged_adamw_8bit"] = "paged_adamw_8bit"
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    max_grad_norm: float = 1.0

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10

    # Output
    output_dir: Path = Path("outputs")

    # Windows compatibility - CRITICAL
    dataloader_num_workers: int = 0  # Must be 0 on Windows


class AspireConfig(BaseSettings):
    """Main configuration for ASPIRE training."""

    # Component configs
    student: StudentConfig = Field(default_factory=StudentConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # Experiment tracking
    experiment_name: str = "aspire-run"
    use_wandb: bool = True
    wandb_project: str = "aspire-ai"

    # Seeds
    seed: int = 42

    # Environment
    device: str = "cuda"

    model_config = {"env_prefix": "ASPIRE_", "env_nested_delimiter": "__"}

    @classmethod
    def from_yaml(cls, path: Path) -> "AspireConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
