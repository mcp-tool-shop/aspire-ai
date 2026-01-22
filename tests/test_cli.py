"""
Tests for ASPIRE CLI (aspire/cli.py).

Coverage target: CLI commands and options.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import tempfile

import pytest
from typer.testing import CliRunner

from aspire.cli import app
from aspire import __version__


runner = CliRunner()


# ============================================================================
# Version Tests
# ============================================================================

def test_version_flag_long():
    """Test aspire --version outputs version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_flag_short():
    """Test aspire -V outputs version."""
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


# ============================================================================
# Help Tests
# ============================================================================

def test_no_args_shows_help():
    """Test aspire with no args shows help."""
    result = runner.invoke(app, [])
    # Exit code 2 is expected for "no_args_is_help=True" behavior
    assert result.exit_code in (0, 2)
    assert "ASPIRE" in result.stdout or "Usage" in result.stdout


# ============================================================================
# Doctor Command Tests
# ============================================================================

def test_doctor_checks_python_version():
    """Test aspire doctor checks Python version."""
    result = runner.invoke(app, ["doctor"])
    # Should contain Python version check
    assert "Python" in result.stdout
    # Python version should be displayed
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert py_version in result.stdout


def test_doctor_checks_pytorch():
    """Test aspire doctor checks PyTorch."""
    result = runner.invoke(app, ["doctor"])
    # Should check for PyTorch
    assert "PyTorch" in result.stdout


def test_doctor_checks_transformers():
    """Test aspire doctor checks transformers."""
    result = runner.invoke(app, ["doctor"])
    # Should check for transformers
    assert "Transformers" in result.stdout


def test_doctor_with_anthropic_key_set():
    """Test aspire doctor with ANTHROPIC_API_KEY set."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123456789012"}):
        result = runner.invoke(app, ["doctor"])
        assert "ANTHROPIC_API_KEY" in result.stdout
        # Should show masked key
        assert "sk-ant-t" in result.stdout or "set" in result.stdout.lower()


def test_doctor_with_anthropic_key_missing():
    """Test aspire doctor with ANTHROPIC_API_KEY missing."""
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    with patch.dict(os.environ, env, clear=True):
        result = runner.invoke(app, ["doctor"])
        assert "ANTHROPIC_API_KEY" in result.stdout
        # Should indicate not set or warn
        assert "not set" in result.stdout.lower() or "WARN" in result.stdout


def test_doctor_with_openai_key_set():
    """Test aspire doctor with OPENAI_API_KEY set."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test1234567890123456789"}):
        result = runner.invoke(app, ["doctor"])
        assert "OPENAI_API_KEY" in result.stdout


def test_doctor_disk_space_check():
    """Test aspire doctor disk space check."""
    result = runner.invoke(app, ["doctor"])
    # Should check storage/disk space
    assert "Storage" in result.stdout or "disk" in result.stdout.lower() or "Free" in result.stdout


# ============================================================================
# Init Command Tests
# ============================================================================

def test_init_creates_config_file():
    """Test aspire init creates config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test-config.yaml"
        result = runner.invoke(app, ["init", "--output", str(output_path)])
        assert result.exit_code == 0
        assert output_path.exists()
        # Verify it's a valid YAML file (can be read as text)
        content = output_path.read_text()
        assert len(content) > 0
        assert "student" in content or "training" in content


def test_init_custom_output_path():
    """Test aspire init --output custom.yaml uses custom path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = Path(tmpdir) / "my-custom-config.yaml"
        result = runner.invoke(app, ["init", "--output", str(custom_path)])
        assert result.exit_code == 0
        assert custom_path.exists()
        assert "my-custom-config.yaml" in str(custom_path)


# ============================================================================
# Teachers Command Tests
# ============================================================================

def test_teachers_lists_all_teachers():
    """Test aspire teachers lists all teachers."""
    result = runner.invoke(app, ["teachers"])
    assert result.exit_code == 0
    # Should list known teachers
    assert "claude" in result.stdout.lower()
    assert "openai" in result.stdout.lower() or "gpt" in result.stdout.lower()
    assert "socratic" in result.stdout.lower()


# ============================================================================
# Train Command Tests (with mocks)
# ============================================================================

def test_train_with_demo_prompts():
    """Test aspire train with demo prompts (mock trainer)."""
    # Setup mocks
    mock_config = MagicMock()
    mock_config.training = MagicMock()
    mock_config.teacher = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = {"loss": [1.0, 0.5]}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch at the aspire.config and aspire.trainer module level
        with patch("aspire.config.AspireConfig", return_value=mock_config) as mock_config_class, \
             patch("aspire.trainer.AspireTrainer", return_value=mock_trainer) as mock_trainer_class:
            mock_config_class.from_yaml.return_value = mock_config
            result = runner.invoke(app, ["train", "--output", tmpdir])
            # Should attempt to train (may show warning about demo prompts)
            assert "demo" in result.stdout.lower() or mock_trainer.train.called


def test_train_with_config_file():
    """Test aspire train --config loads config file."""
    # Setup mocks
    mock_config = MagicMock()
    mock_config.training = MagicMock()
    mock_config.teacher = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = {"loss": [1.0, 0.5]}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a config file
        config_path = Path(tmpdir) / "config.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump({"training": {"num_epochs": 1}}, f)

        # Patch at the aspire.config and aspire.trainer module level
        with patch("aspire.config.AspireConfig", return_value=mock_config) as mock_config_class, \
             patch("aspire.trainer.AspireTrainer", return_value=mock_trainer):
            mock_config_class.from_yaml.return_value = mock_config
            result = runner.invoke(app, ["train", "--config", str(config_path), "--output", tmpdir])
            # Should load config from file
            mock_config_class.from_yaml.assert_called_once()


# ============================================================================
# Evaluate Command Tests (with mocks)
# ============================================================================

def test_evaluate_with_mock_checkpoint():
    """Test aspire evaluate with mock checkpoint."""
    # Setup mocks
    mock_config = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer._evaluate = AsyncMock(return_value={"avg_score": 7.5, "min_score": 5.0, "max_score": 9.0})

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock checkpoint structure
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        checkpoint_dir.mkdir()
        config_path = checkpoint_dir / "config.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump({"training": {"num_epochs": 1}}, f)

        # Create prompts file
        prompts_path = Path(tmpdir) / "prompts.json"
        with open(prompts_path, "w") as f:
            json.dump(["Test prompt 1", "Test prompt 2"], f)

        # Patch at the aspire.config and aspire.trainer module level
        with patch("aspire.config.AspireConfig") as mock_config_class, \
             patch("aspire.trainer.AspireTrainer", return_value=mock_trainer):
            mock_config_class.from_yaml.return_value = mock_config
            result = runner.invoke(app, ["evaluate", str(checkpoint_dir), "--prompts", str(prompts_path)])
            # Should attempt evaluation
            mock_trainer.load_checkpoint.assert_called_once()
