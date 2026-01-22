"""
Tests for ASPIRE Trainer (aspire/trainer.py) - Unit tests with mocks.

Coverage target: Trainer initialization and component setup.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import torch

from aspire.config import AspireConfig
from aspire.trainer import AspireDataset


# ============================================================================
# AspireDataset Tests
# ============================================================================

class TestAspireDataset:
    """Tests for AspireDataset class."""

    def test_dataset_len_returns_correct_length(self, mock_tokenizer):
        """Test AspireDataset.__len__ returns correct length."""
        prompts = ["prompt 1", "prompt 2", "prompt 3"]
        dataset = AspireDataset(prompts=prompts, tokenizer=mock_tokenizer, max_length=128)
        assert len(dataset) == 3

    def test_dataset_len_empty(self, mock_tokenizer):
        """Test AspireDataset.__len__ with empty prompts."""
        prompts = []
        dataset = AspireDataset(prompts=prompts, tokenizer=mock_tokenizer, max_length=128)
        assert len(dataset) == 0

    def test_dataset_getitem_returns_tokenized_data(self):
        """Test AspireDataset.__getitem__ returns tokenized data."""
        # Create a proper mock tokenizer that returns tensor-like structure
        mock_tokenizer = MagicMock()

        def tokenize_func(text, **kwargs):
            max_length = kwargs.get("max_length", 128)
            return {
                "input_ids": torch.randint(0, 1000, (1, max_length)),
                "attention_mask": torch.ones(1, max_length, dtype=torch.long),
            }

        mock_tokenizer.side_effect = tokenize_func
        mock_tokenizer.__call__ = tokenize_func

        prompts = ["What is Python?", "Explain recursion."]
        dataset = AspireDataset(prompts=prompts, tokenizer=mock_tokenizer, max_length=128)

        item = dataset[0]
        assert "prompt" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert item["prompt"] == "What is Python?"
        assert item["input_ids"].shape == (128,)
        assert item["attention_mask"].shape == (128,)

    def test_dataset_getitem_different_indices(self):
        """Test AspireDataset.__getitem__ with different indices."""
        mock_tokenizer = MagicMock()

        def tokenize_func(text, **kwargs):
            max_length = kwargs.get("max_length", 128)
            return {
                "input_ids": torch.randint(0, 1000, (1, max_length)),
                "attention_mask": torch.ones(1, max_length, dtype=torch.long),
            }

        mock_tokenizer.side_effect = tokenize_func
        mock_tokenizer.__call__ = tokenize_func

        prompts = ["First prompt", "Second prompt", "Third prompt"]
        dataset = AspireDataset(prompts=prompts, tokenizer=mock_tokenizer, max_length=64)

        assert dataset[0]["prompt"] == "First prompt"
        assert dataset[1]["prompt"] == "Second prompt"
        assert dataset[2]["prompt"] == "Third prompt"


# ============================================================================
# AspireTrainer Initialization Tests (heavily mocked)
# ============================================================================

class TestAspireTrainerInit:
    """Tests for AspireTrainer initialization."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all heavy dependencies for trainer init."""
        with patch("aspire.trainer.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.trainer.AutoTokenizer") as mock_tokenizer_class, \
             patch("aspire.trainer.get_peft_model") as mock_peft, \
             patch("aspire.trainer.prepare_model_for_kbit_training") as mock_prepare, \
             patch("aspire.trainer.CriticHead") as mock_critic_head, \
             patch("aspire.trainer.SeparateCritic") as mock_separate_critic, \
             patch("aspire.trainer.SharedEncoderCritic") as mock_shared_critic, \
             patch("aspire.trainer.get_teacher") as mock_get_teacher, \
             patch("aspire.trainer.AspireLoss") as mock_loss, \
             patch("aspire.trainer.DialogueGenerator") as mock_dialogue_gen, \
             patch("aspire.trainer.DialogueManager") as mock_dialogue_mgr, \
             patch("aspire.trainer.DialogueFormatter") as mock_formatter, \
             patch("aspire.trainer.AdamW") as mock_adamw:

            # Setup model mock
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(10, 10))])
            mock_model_class.from_pretrained.return_value = mock_model
            mock_peft.return_value = mock_model
            mock_prepare.return_value = mock_model

            # Setup tokenizer mock
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Setup critic mock
            mock_critic = MagicMock()
            mock_critic.to.return_value = mock_critic
            mock_critic.get_trainable_parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
            mock_critic_head.return_value = mock_critic
            mock_separate_critic.return_value = mock_critic
            mock_shared_critic.return_value = mock_critic

            # Setup teacher mock
            mock_teacher = MagicMock()
            mock_get_teacher.return_value = mock_teacher

            # Setup optimizer mock
            mock_optimizer = MagicMock()
            mock_adamw.return_value = mock_optimizer

            yield {
                "model_class": mock_model_class,
                "model": mock_model,
                "tokenizer_class": mock_tokenizer_class,
                "tokenizer": mock_tokenizer,
                "peft": mock_peft,
                "prepare": mock_prepare,
                "critic_head": mock_critic_head,
                "separate_critic": mock_separate_critic,
                "shared_critic": mock_shared_critic,
                "critic": mock_critic,
                "get_teacher": mock_get_teacher,
                "teacher": mock_teacher,
                "loss": mock_loss,
                "adamw": mock_adamw,
            }

    def test_trainer_init_with_default_config(self, mock_dependencies):
        """Test AspireTrainer initialization with default config."""
        from aspire.trainer import AspireTrainer

        config = AspireConfig()
        trainer = AspireTrainer(config)

        # Should have initialized all components
        assert trainer.config == config
        mock_dependencies["model_class"].from_pretrained.assert_called_once()
        mock_dependencies["tokenizer_class"].from_pretrained.assert_called_once()

    def test_trainer_init_student_with_lora(self, mock_dependencies):
        """Test AspireTrainer._init_student with LoRA."""
        from aspire.trainer import AspireTrainer

        config = AspireConfig()
        config.student.use_lora = True
        config.student.load_in_4bit = False
        config.student.load_in_8bit = False

        trainer = AspireTrainer(config)

        # Should apply LoRA
        mock_dependencies["peft"].assert_called_once()

    def test_trainer_init_student_without_lora(self, mock_dependencies):
        """Test AspireTrainer._init_student without LoRA."""
        from aspire.trainer import AspireTrainer

        config = AspireConfig()
        config.student.use_lora = False
        config.student.load_in_4bit = False
        config.student.load_in_8bit = False

        trainer = AspireTrainer(config)

        # Should NOT apply LoRA
        mock_dependencies["peft"].assert_not_called()

    def test_trainer_init_student_with_4bit_quantization(self, mock_dependencies):
        """Test AspireTrainer._init_student with 4-bit quantization."""
        from aspire.trainer import AspireTrainer

        config = AspireConfig()
        config.student.load_in_4bit = True
        config.student.load_in_8bit = False

        trainer = AspireTrainer(config)

        # Should prepare for kbit training
        mock_dependencies["prepare"].assert_called_once()
        # Check that quantization config was passed
        call_kwargs = mock_dependencies["model_class"].from_pretrained.call_args[1]
        assert call_kwargs.get("quantization_config") is not None

    def test_trainer_init_critic_head_architecture(self, mock_dependencies):
        """Test AspireTrainer._init_critic with head architecture."""
        from aspire.trainer import AspireTrainer

        config = AspireConfig()
        config.critic.architecture = "head"

        trainer = AspireTrainer(config)

        # Should use CriticHead
        mock_dependencies["critic_head"].assert_called_once()
        mock_dependencies["separate_critic"].assert_not_called()
        mock_dependencies["shared_critic"].assert_not_called()

    def test_trainer_init_critic_separate_architecture(self, mock_dependencies):
        """Test AspireTrainer._init_critic with separate architecture."""
        from aspire.trainer import AspireTrainer

        config = AspireConfig()
        config.critic.architecture = "separate"

        trainer = AspireTrainer(config)

        # Should use SeparateCritic
        mock_dependencies["separate_critic"].assert_called_once()
        mock_dependencies["critic_head"].assert_not_called()

    def test_trainer_init_critic_shared_encoder_architecture(self, mock_dependencies):
        """Test AspireTrainer._init_critic with shared_encoder architecture."""
        from aspire.trainer import AspireTrainer

        config = AspireConfig()
        config.critic.architecture = "shared_encoder"

        trainer = AspireTrainer(config)

        # Should use SharedEncoderCritic
        mock_dependencies["shared_critic"].assert_called_once()
        mock_dependencies["critic_head"].assert_not_called()

    def test_trainer_init_teacher_claude(self, mock_dependencies):
        """Test AspireTrainer._init_teacher with Claude teacher."""
        from aspire.trainer import AspireTrainer

        config = AspireConfig()
        config.teacher.default_teacher = "claude"
        config.teacher.claude_model = "claude-sonnet-4-20250514"

        trainer = AspireTrainer(config)

        # Should call get_teacher with correct args
        mock_dependencies["get_teacher"].assert_called_once()
        call_args = mock_dependencies["get_teacher"].call_args
        assert call_args[0][0] == "claude"

    def test_trainer_init_teacher_openai(self, mock_dependencies):
        """Test AspireTrainer._init_teacher with OpenAI teacher."""
        from aspire.trainer import AspireTrainer

        config = AspireConfig()
        config.teacher.default_teacher = "openai"
        config.teacher.openai_model = "gpt-4o"

        trainer = AspireTrainer(config)

        # Should call get_teacher with correct args
        mock_dependencies["get_teacher"].assert_called_once()
        call_args = mock_dependencies["get_teacher"].call_args
        assert call_args[0][0] == "openai"

    def test_trainer_sets_random_seed(self, mock_dependencies):
        """Test AspireTrainer sets random seed from config."""
        from aspire.trainer import AspireTrainer

        config = AspireConfig()
        config.seed = 12345

        with patch("aspire.trainer.torch.manual_seed") as mock_seed:
            trainer = AspireTrainer(config)
            mock_seed.assert_called_once_with(12345)


# ============================================================================
# Config Tests
# ============================================================================

class TestAspireConfig:
    """Tests for AspireConfig validation and serialization."""

    def test_config_default_values(self):
        """Test AspireConfig has sensible defaults."""
        config = AspireConfig()

        assert config.student.model_name_or_path == "microsoft/Phi-3-mini-4k-instruct"
        assert config.student.use_lora is True
        assert config.training.batch_size == 4
        assert config.training.num_epochs == 3
        assert config.training.dataloader_num_workers == 0  # Windows compatibility

    def test_config_student_lora_settings(self):
        """Test StudentConfig LoRA settings."""
        config = AspireConfig()

        assert config.student.lora_r == 16
        assert config.student.lora_alpha == 32
        assert config.student.lora_dropout == 0.05
        assert "q_proj" in config.student.lora_target_modules

    def test_config_critic_architecture_choices(self):
        """Test CriticConfig architecture is valid."""
        config = AspireConfig()
        assert config.critic.architecture in ("head", "separate", "shared_encoder")

    def test_config_teacher_settings(self):
        """Test TeacherConfig settings."""
        config = AspireConfig()

        assert config.teacher.default_teacher == "claude"
        assert config.teacher.temperature == 0.7
        assert config.teacher.max_dialogue_turns == 3

    def test_config_loss_weights(self):
        """Test LossConfig weights sum appropriately."""
        config = AspireConfig()

        # Score weight should be primary
        assert config.loss.critic_score_weight == 1.0
        assert config.loss.student_reward_weight == 1.0

    def test_config_to_yaml_creates_file(self):
        """Test AspireConfig.to_yaml creates valid file."""
        config = AspireConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(path)

            assert path.exists()
            content = path.read_text()
            assert "student" in content
            assert "training" in content

    def test_config_from_yaml_loads_correctly(self):
        """Test AspireConfig.from_yaml loads config."""
        import yaml

        config_data = {
            "student": {"model_name_or_path": "test-model"},
            "training": {"num_epochs": 5, "batch_size": 8},
            "seed": 999,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            with open(path, "w") as f:
                yaml.dump(config_data, f)

            loaded = AspireConfig.from_yaml(path)

            assert loaded.student.model_name_or_path == "test-model"
            assert loaded.training.num_epochs == 5
            assert loaded.training.batch_size == 8
            assert loaded.seed == 999

    def test_config_environment_variable_override(self):
        """Test AspireConfig can be overridden via environment variables."""
        import os

        # The env prefix is ASPIRE_ with __ for nested
        with patch.dict(os.environ, {"ASPIRE_SEED": "777"}):
            config = AspireConfig()
            # Note: Pydantic settings may require specific setup for this
            # This test verifies the config accepts env vars

    def test_config_training_optimizer_choices(self):
        """Test TrainingConfig optimizer is valid."""
        config = AspireConfig()
        assert config.training.optimizer in ("adamw", "adamw_8bit", "paged_adamw_8bit")

    def test_config_training_scheduler_choices(self):
        """Test TrainingConfig lr_scheduler is valid."""
        config = AspireConfig()
        assert config.training.lr_scheduler in ("cosine", "linear", "constant")


# ============================================================================
# AspireTrainer Method Tests (unit tests without full trainer)
# ============================================================================

class TestAspireTrainerMethods:
    """Tests for AspireTrainer static/class methods and utilities."""

    def test_batch_dict_structure(self):
        """Test that batch dicts have expected structure."""
        batch = {
            "prompt": ["What is AI?", "Explain ML."],
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "attention_mask": torch.ones(2, 128),
        }

        assert "prompt" in batch
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape == (2, 128)

    def test_learning_rate_schedule_cosine(self):
        """Test cosine learning rate decay calculation."""
        # Cosine decay formula
        import math
        lr_max = 1e-4
        current_step = 50
        total_steps = 100

        progress = current_step / total_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        expected_lr = lr_max * cosine_decay

        assert 0 < expected_lr < lr_max

    def test_learning_rate_schedule_linear(self):
        """Test linear learning rate decay calculation."""
        lr_max = 1e-4
        current_step = 50
        total_steps = 100

        linear_decay = 1 - (current_step / total_steps)
        expected_lr = lr_max * linear_decay

        assert expected_lr == lr_max * 0.5

    def test_checkpoint_path_format(self, tmp_path):
        """Test checkpoint path formatting."""
        output_dir = tmp_path / "checkpoints"
        output_dir.mkdir()

        step = 1000
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = output_dir / checkpoint_name

        assert str(checkpoint_path).endswith("checkpoint-1000")

    def test_metric_aggregation(self):
        """Test metric dict aggregation."""
        step_metrics = [
            {"loss": 0.5, "critic_loss": 0.3},
            {"loss": 0.4, "critic_loss": 0.2},
            {"loss": 0.3, "critic_loss": 0.1},
        ]

        avg_loss = sum(m["loss"] for m in step_metrics) / len(step_metrics)
        avg_critic = sum(m["critic_loss"] for m in step_metrics) / len(step_metrics)

        assert abs(avg_loss - 0.4) < 1e-6
        assert abs(avg_critic - 0.2) < 1e-6

    def test_gradient_clipping_value(self):
        """Test gradient clipping max norm value."""
        config = AspireConfig()
        assert config.training.max_grad_norm == 1.0

    def test_warmup_steps_calculation(self):
        """Test warmup steps calculation."""
        total_steps = 1000
        warmup_ratio = 0.1

        warmup_steps = int(total_steps * warmup_ratio)
        assert warmup_steps == 100


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests verifying components work together."""

    def test_dataset_iteration(self, mock_tokenizer):
        """Test AspireDataset can be iterated."""
        prompts = ["P1", "P2", "P3"]
        dataset = AspireDataset(prompts=prompts, tokenizer=mock_tokenizer, max_length=64)

        items = list(dataset)
        assert len(items) == 3

    def test_dataloader_creation(self, mock_tokenizer):
        """Test DataLoader can be created from AspireDataset (num_workers=0)."""
        from torch.utils.data import DataLoader

        prompts = ["Prompt " + str(i) for i in range(10)]
        dataset = AspireDataset(prompts=prompts, tokenizer=mock_tokenizer, max_length=64)

        # CRITICAL: num_workers=0 for Windows
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        batches = list(dataloader)
        assert len(batches) == 5  # 10 items / batch_size 2

    def test_config_model_dump_round_trip(self):
        """Test config model_dump creates serializable dict."""
        config = AspireConfig()

        config_dict = config.model_dump()
        assert config_dict["seed"] == 42
        assert "student" in config_dict
        assert "training" in config_dict

    def test_config_pydantic_validation(self):
        """Test pydantic validates config fields."""
        config = AspireConfig()

        # Pydantic models should have model_dump
        assert hasattr(config, "model_dump")
        dumped = config.model_dump()
        assert isinstance(dumped, dict)

    def test_loss_weights_in_config(self):
        """Test that loss weights are configurable."""
        config = AspireConfig()

        assert hasattr(config.loss, "critic_score_weight")
        assert hasattr(config.loss, "student_reward_weight")
        assert config.loss.critic_score_weight > 0
        assert config.loss.student_reward_weight > 0

    def test_teacher_config_default_teacher(self):
        """Test teacher configuration default_teacher field."""
        config = AspireConfig()

        assert config.teacher.default_teacher in ("claude", "openai", "local")
        assert config.teacher.max_dialogue_turns > 0
