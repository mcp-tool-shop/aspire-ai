"""
Configuration for ASPIRE Isaac Gym integration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class TaskType(str, Enum):
    """Supported robotics task types."""
    REACHING = "reaching"
    MANIPULATION = "manipulation"
    LOCOMOTION = "locomotion"
    NAVIGATION = "navigation"
    ASSEMBLY = "assembly"


class CriticArchitecture(str, Enum):
    """Trajectory critic architectures."""
    TRANSFORMER = "transformer"  # Attention over trajectory
    LSTM = "lstm"                # Sequential processing
    TCN = "tcn"                  # Temporal convolutional
    MLP = "mlp"                  # Simple feedforward (for short horizons)


@dataclass
class TeacherConfig:
    """Configuration for motion teachers."""

    # Which personas to use
    personas: list[str] = field(default_factory=lambda: [
        "safety_inspector",
        "efficiency_expert",
        "grace_coach",
    ])

    # How to combine multiple teachers
    strategy: Literal["vote", "rotate", "debate"] = "vote"

    # Use physics oracle (ground truth from simulator)?
    use_physics_oracle: bool = True

    # Use VLM for visual understanding of scenes?
    use_vision_teacher: bool = False
    vision_model: str = "claude-3-5-sonnet-20241022"

    # Weights for different evaluation dimensions
    safety_weight: float = 2.0      # Safety is paramount
    efficiency_weight: float = 1.0
    smoothness_weight: float = 0.5
    goal_achievement_weight: float = 1.5


@dataclass
class CriticConfig:
    """Configuration for trajectory critic."""

    architecture: CriticArchitecture = CriticArchitecture.TRANSFORMER

    # Model dimensions
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8  # For transformer

    # Input processing
    state_dim: int = 0      # Set from environment
    action_dim: int = 0     # Set from environment
    max_trajectory_len: int = 256

    # Output heads
    predict_score: bool = True
    predict_reasoning: bool = True
    predict_improvement: bool = True  # Suggest better action

    # Training
    dropout: float = 0.1
    use_layer_norm: bool = True


@dataclass
class TrainingConfig:
    """Training configuration for embodied ASPIRE."""

    # Environment
    env_name: str = "FrankaCubeStack-v0"
    num_envs: int = 512  # Parallel environments (GPU-accelerated)

    # Episodes
    max_episode_length: int = 256
    episodes_per_epoch: int = 100
    epochs: int = 100

    # Trajectory collection
    trajectory_batch_size: int = 64
    critic_update_frequency: int = 1  # Update critic every N episodes
    policy_update_frequency: int = 1

    # Learning rates
    critic_lr: float = 3e-4
    policy_lr: float = 1e-4

    # Loss weights
    critic_score_weight: float = 1.0
    critic_reasoning_weight: float = 0.5
    policy_reward_weight: float = 1.0
    policy_improvement_weight: float = 0.3

    # Regularization
    kl_coefficient: float = 0.01  # Prevent policy collapse
    entropy_bonus: float = 0.01  # Encourage exploration

    # Checkpointing
    save_frequency: int = 10
    checkpoint_dir: str = "checkpoints/isaac"

    # Logging
    log_frequency: int = 1
    use_wandb: bool = False
    wandb_project: str = "aspire-isaac"


@dataclass
class IsaacAspireConfig:
    """Complete configuration for ASPIRE Isaac training."""

    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Hardware
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        if self.training.num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        if self.teacher.safety_weight < 1.0:
            import warnings
            warnings.warn(
                "Safety weight < 1.0 is not recommended for real robot deployment"
            )
