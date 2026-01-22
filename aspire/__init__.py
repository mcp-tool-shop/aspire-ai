"""
ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine

A fine-tuning paradigm that mirrors human learning through internalized critics.
"""

__version__ = "0.1.0"

from aspire.config import AspireConfig
from aspire.trainer import AspireTrainer

__all__ = ["AspireConfig", "AspireTrainer", "__version__"]
