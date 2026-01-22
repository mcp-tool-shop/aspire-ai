"""
Critic models for ASPIRE training.

The critic learns to predict what the teacher would think - becoming
the internalized judgment that guides the student.
"""

from aspire.critic.base import BaseCritic, CriticOutput
from aspire.critic.head import CriticHead
from aspire.critic.separate import SeparateCritic
from aspire.critic.shared import SharedEncoderCritic

__all__ = [
    "BaseCritic",
    "CriticOutput",
    "CriticHead",
    "SeparateCritic",
    "SharedEncoderCritic",
]
