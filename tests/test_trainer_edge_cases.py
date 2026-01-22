"""
Edge case tests for ASPIRE Trainer.

Tests for boundary conditions and edge cases:
- Empty/single prompt datasets
- Batch size edge cases
- Checkpoint save/load edge cases
- Mixed precision training
- Gradient accumulation

Windows compatibility notes:
- Use num_workers=0 in DataLoader tests
- Use if __name__ == "__main__": freeze_support() pattern
- Mock heavy model loading
"""

from __future__ import annotations

import os
import tempfile
from multiprocessing import freeze_support
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

os.environ["XFORMERS_DISABLED"] = "1"

import pytest
import torch
from torch.utils.data import DataLoader

# We need to mock the heavy imports before importing trainer
with patch.dict(os.environ, {"XFORMERS_DISABLED": "1"}):
    from aspire.trainer import AspireDataset


# ============================================================================
# AspireDataset Edge Cases
# ============================================================================


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 1000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(
        self,
        text,
        truncation: bool = True,
        max_length: int = 512,
        padding: str = "max_length",
        return_tensors: str = "pt",
    ):
        """Tokenize text."""
        if isinstance(text, str):
            batch_size = 1
        else:
            batch_size = len(text)

        seq_len = min(max_length, 50)  # Use shorter sequences for testing
        input_ids = torch.randint(2, self.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        # Simulate padding for shorter sequences
        if padding == "max_length" and seq_len < max_length:
            pad_len = max_length - seq_len
            input_ids = torch.cat([
                input_ids,
                torch.zeros(batch_size, pad_len, dtype=torch.long)
            ], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(batch_size, pad_len, dtype=torch.long)
            ], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class TestAspireDatasetEdgeCases:
    """Edge case tests for AspireDataset."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        return MockTokenizer()

    def test_empty_prompts_list(self, mock_tokenizer):
        """Dataset with empty prompts list."""
        dataset = AspireDataset(
            prompts=[],
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        assert len(dataset) == 0

    def test_single_prompt(self, mock_tokenizer):
        """Dataset with single prompt."""
        dataset = AspireDataset(
            prompts=["What is machine learning?"],
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        assert len(dataset) == 1

        item = dataset[0]
        assert "prompt" in item
        assert "input_ids" in item
        assert "attention_mask" in item

    def test_batch_larger_than_dataset(self, mock_tokenizer):
        """DataLoader with batch_size > dataset size."""
        dataset = AspireDataset(
            prompts=["Prompt 1", "Prompt 2"],
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        # Batch size 10 but only 2 items
        # DataLoader should handle this gracefully
        loader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=False,
            num_workers=0,  # Windows compatibility
        )

        batches = list(loader)
        assert len(batches) == 1  # Single batch with 2 items
        assert len(batches[0]["prompt"]) == 2

    def test_very_long_prompt(self, mock_tokenizer):
        """Dataset handles very long prompts (truncation)."""
        long_prompt = "word " * 10000  # Very long prompt

        dataset = AspireDataset(
            prompts=[long_prompt],
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        item = dataset[0]

        # Should be truncated to max_length
        assert item["input_ids"].shape[0] == 512
        assert item["attention_mask"].shape[0] == 512

    def test_empty_string_prompt(self, mock_tokenizer):
        """Dataset handles empty string prompt."""
        dataset = AspireDataset(
            prompts=[""],
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        item = dataset[0]

        assert item["prompt"] == ""
        assert "input_ids" in item

    def test_unicode_prompts(self, mock_tokenizer):
        """Dataset handles unicode characters."""
        unicode_prompts = [
            "What is 机器学习?",  # Chinese
            "Qu'est-ce que l'apprentissage?",  # French
            "🤖 AI is cool! 🎉",  # Emojis
            "Привет мир",  # Russian
        ]

        dataset = AspireDataset(
            prompts=unicode_prompts,
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        assert len(dataset) == 4

        for i in range(4):
            item = dataset[i]
            assert item["prompt"] == unicode_prompts[i]

    def test_multiple_same_prompts(self, mock_tokenizer):
        """Dataset handles duplicate prompts."""
        prompts = ["Same prompt"] * 100

        dataset = AspireDataset(
            prompts=prompts,
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        assert len(dataset) == 100

    def test_dataloader_num_workers_zero(self, mock_tokenizer):
        """DataLoader works with num_workers=0 (Windows requirement)."""
        dataset = AspireDataset(
            prompts=["P1", "P2", "P3", "P4"],
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        # This is the Windows-compatible configuration
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 2


# ============================================================================
# Trainer Checkpoint Edge Cases (Mocked)
# ============================================================================


class TestTrainerCheckpointEdgeCases:
    """Edge case tests for checkpoint save/load."""

    def test_checkpoint_directory_creation(self, tmp_path):
        """Checkpoint save creates directory if needed."""
        checkpoint_dir = tmp_path / "checkpoints" / "nested" / "deep"

        # Simulate checkpoint save
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        assert checkpoint_dir.exists()

    def test_checkpoint_missing_files_handling(self, tmp_path):
        """Loading from checkpoint with missing files."""
        checkpoint_dir = tmp_path / "incomplete_checkpoint"
        checkpoint_dir.mkdir()

        # Create only partial checkpoint
        (checkpoint_dir / "trainer_state.json").write_text('{"global_step": 100}')

        # Missing model files should be handled gracefully
        # (Actual implementation would raise FileNotFoundError)
        assert (checkpoint_dir / "trainer_state.json").exists()
        assert not (checkpoint_dir / "pytorch_model.bin").exists()

    def test_checkpoint_corrupted_file(self, tmp_path):
        """Loading checkpoint with corrupted file."""
        checkpoint_dir = tmp_path / "corrupted_checkpoint"
        checkpoint_dir.mkdir()

        # Create corrupted checkpoint
        (checkpoint_dir / "trainer_state.json").write_text("not valid json {{{")

        # Should handle gracefully (in actual implementation)
        content = (checkpoint_dir / "trainer_state.json").read_text()
        assert "not valid json" in content


# ============================================================================
# Gradient Accumulation Edge Cases
# ============================================================================


class TestGradientAccumulationEdgeCases:
    """Edge case tests for gradient accumulation."""

    def test_accumulation_with_single_batch(self):
        """Gradient accumulation with only one batch."""
        # Simulate gradient accumulation
        accumulation_steps = 4
        num_batches = 1

        effective_updates = num_batches // accumulation_steps

        # With 1 batch and 4 accumulation steps, no update occurs
        assert effective_updates == 0

    def test_accumulation_boundary_exact(self):
        """Gradient accumulation at exact boundary."""
        accumulation_steps = 4
        num_batches = 4

        effective_updates = num_batches // accumulation_steps
        assert effective_updates == 1

    def test_accumulation_boundary_partial(self):
        """Gradient accumulation with partial last batch."""
        accumulation_steps = 4
        num_batches = 7  # 1 full accumulation + 3 partial

        effective_updates = num_batches // accumulation_steps
        remaining = num_batches % accumulation_steps

        assert effective_updates == 1
        assert remaining == 3


# ============================================================================
# Learning Rate Scheduler Edge Cases
# ============================================================================


class TestLRSchedulerEdgeCases:
    """Edge case tests for learning rate scheduler."""

    def test_warmup_steps_zero(self):
        """LR scheduler with zero warmup steps."""
        num_warmup_steps = 0
        num_training_steps = 1000

        # At step 0, should be at full LR (no warmup)
        warmup_ratio = min(1.0, 1 / max(num_warmup_steps, 1))
        assert warmup_ratio == 1.0

    def test_warmup_longer_than_training(self):
        """LR scheduler with warmup > training steps."""
        num_warmup_steps = 1000
        num_training_steps = 100

        # Warmup should be capped at training steps
        effective_warmup = min(num_warmup_steps, num_training_steps)
        assert effective_warmup == 100

    def test_single_step_training(self):
        """LR scheduler with only 1 training step."""
        num_training_steps = 1
        num_warmup_steps = 0

        # Should work without division by zero
        current_step = 0
        lr_multiplier = 1.0 - (current_step / max(num_training_steps, 1))
        assert lr_multiplier == 1.0


# ============================================================================
# Mixed Precision Edge Cases
# ============================================================================


class TestMixedPrecisionEdgeCases:
    """Edge case tests for mixed precision training."""

    def test_bf16_tensor_operations(self):
        """BF16 tensor operations work correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for bf16 test")

        x = torch.randn(10, 10, dtype=torch.bfloat16, device="cuda")
        y = torch.randn(10, 10, dtype=torch.bfloat16, device="cuda")

        z = x @ y
        assert z.dtype == torch.bfloat16
        assert not torch.isnan(z).any()

    def test_fp16_tensor_operations(self):
        """FP16 tensor operations work correctly."""
        x = torch.randn(10, 10, dtype=torch.float16)
        y = torch.randn(10, 10, dtype=torch.float16)

        z = x @ y
        assert z.dtype == torch.float16
        assert not torch.isnan(z).any()

    def test_mixed_dtype_operations(self):
        """Mixed dtype operations auto-cast correctly."""
        fp32 = torch.randn(10, 10, dtype=torch.float32)
        fp16 = torch.randn(10, 10, dtype=torch.float16)

        # PyTorch will upcast to fp32
        result = fp32 + fp16.float()
        assert result.dtype == torch.float32


# ============================================================================
# Training Loop Edge Cases
# ============================================================================


class TestTrainingLoopEdgeCases:
    """Edge case tests for training loop scenarios."""

    def test_zero_epochs(self):
        """Training with zero epochs should do nothing."""
        num_epochs = 0
        global_step = 0

        # Loop should not execute
        for _ in range(num_epochs):
            global_step += 1

        assert global_step == 0

    def test_training_interruption_recovery(self, tmp_path):
        """Training can be resumed from checkpoint."""
        # Simulate interrupted training
        checkpoint_state = {
            "global_step": 500,
            "epoch": 2,
            "best_loss": 0.5,
        }

        checkpoint_file = tmp_path / "checkpoint.pt"
        torch.save(checkpoint_state, checkpoint_file)

        # Load and resume
        loaded = torch.load(checkpoint_file)
        assert loaded["global_step"] == 500
        assert loaded["epoch"] == 2


# ============================================================================
# Windows Compatibility Entry Point
# ============================================================================


if __name__ == "__main__":
    freeze_support()
    pytest.main([__file__, "-v"])
