"""
Loss functions for ASPIRE training.

Two types of losses:
1. Critic losses - train the critic to predict teacher judgment
2. Student losses - train the student using critic feedback
"""

from aspire.losses.critic import (
    CriticLoss,
    CriticScoreLoss,
    CriticReasoningLoss,
)
from aspire.losses.student import (
    StudentLoss,
    RewardLoss,
    ContrastiveLoss,
    TrajectoryLoss,
    CoherenceLoss,
)
from aspire.losses.combined import AspireLoss

__all__ = [
    # Critic losses
    "CriticLoss",
    "CriticScoreLoss",
    "CriticReasoningLoss",
    # Student losses
    "StudentLoss",
    "RewardLoss",
    "ContrastiveLoss",
    "TrajectoryLoss",
    "CoherenceLoss",
    # Combined
    "AspireLoss",
]
