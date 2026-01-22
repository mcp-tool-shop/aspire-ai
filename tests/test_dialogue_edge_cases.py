"""
Edge case tests for ASPIRE Dialogue components.

Tests for boundary conditions and edge cases:
- Empty/very long prompts
- Unicode content
- Cache corruption/recovery
- Concurrent access
- Zero turns
- Max length truncation

Windows compatibility notes:
- Use num_workers=0 in DataLoader tests
- Use if __name__ == "__main__": freeze_support() pattern
- Mock heavy model loading
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from multiprocessing import freeze_support
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["XFORMERS_DISABLED"] = "1"

import pytest
import torch

from aspire.dialogue.generator import DialogueGenerator, GeneratedDialogue
from aspire.dialogue.formatter import DialogueFormatter, FormattedDialogue
from aspire.dialogue.manager import DialogueManager
from aspire.teachers.base import (
    DialogueHistory,
    DialogueTurn,
    TeacherChallenge,
    TeacherEvaluation,
    ChallengeType,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_tokenizer():
    """Mock HuggingFace tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    def mock_call(text, **kwargs):
        batch_size = 1 if isinstance(text, str) else len(text)
        max_length = kwargs.get("max_length", 512)
        return MagicMock(
            input_ids=torch.randint(0, 1000, (batch_size, max_length)),
            attention_mask=torch.ones(batch_size, max_length, dtype=torch.long),
        )

    tokenizer.side_effect = mock_call
    tokenizer.__call__ = mock_call
    tokenizer.decode = MagicMock(return_value="decoded response")
    return tokenizer


@pytest.fixture
def mock_student_model():
    """Mock student model."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.hidden_size = 768

    def mock_generate(**kwargs):
        input_length = kwargs.get("input_ids", torch.zeros(1, 10)).shape[1]
        return torch.randint(0, 1000, (1, input_length + 50))

    model.generate = MagicMock(side_effect=mock_generate)
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_teacher():
    """Mock teacher."""
    teacher = MagicMock()
    teacher.name = "MockTeacher"

    async def mock_challenge(**kwargs):
        return TeacherChallenge(
            challenge_type=ChallengeType.PROBE_REASONING,
            content="Why do you think that?",
        )

    async def mock_evaluate(**kwargs):
        return TeacherEvaluation(
            overall_score=7.5,
            dimension_scores=[],
            reasoning="Good response",
            improved_response="Improved version",
        )

    teacher.challenge = AsyncMock(side_effect=mock_challenge)
    teacher.evaluate = AsyncMock(side_effect=mock_evaluate)
    return teacher


# ============================================================================
# DialogueGenerator Edge Cases
# ============================================================================


class TestDialogueGeneratorEdgeCases:
    """Edge case tests for DialogueGenerator."""

    def test_generator_empty_prompt(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """Generator handles empty string prompt."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=1,
            device="cpu",
        )

        # Format student input with empty prompt
        formatted = generator._format_student_input("")

        # Should still produce some format
        assert isinstance(formatted, str)
        assert "Task:" in formatted

    def test_generator_very_long_prompt(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """Generator handles very long prompts."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=1,
            device="cpu",
        )

        long_prompt = "word " * 10000

        formatted = generator._format_student_input(long_prompt)

        # Should still format (truncation happens at tokenizer)
        assert isinstance(formatted, str)

    def test_generator_unicode_content(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """Generator handles unicode characters."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=1,
            device="cpu",
        )

        unicode_prompts = [
            "Explain 机器学习 (machine learning)",
            "Comment fonctionne l'IA?",
            "🤖 What is AI? 🧠",
            "Привет! Explain neural networks.",
        ]

        for prompt in unicode_prompts:
            formatted = generator._format_student_input(prompt)
            assert isinstance(formatted, str)
            assert "Task:" in formatted

    @pytest.mark.asyncio
    async def test_generator_zero_turns(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """Generator with max_turns=0 returns minimal dialogue."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=0,  # No dialogue turns
            device="cpu",
        )

        dialogue = await generator.generate_dialogue(
            "Test prompt",
            initial_response="Initial response",
        )

        # Should have 0 turns
        assert dialogue.history.num_turns == 0
        # But should still have initial response and final evaluation
        assert dialogue.initial_response == "Initial response"
        assert dialogue.final_evaluation is not None

    def test_generator_format_with_history(
        self, mock_student_model, mock_tokenizer, mock_teacher
    ):
        """Generator formats input with dialogue history."""
        generator = DialogueGenerator(
            student_model=mock_student_model,
            student_tokenizer=mock_tokenizer,
            teacher=mock_teacher,
            max_turns=3,
            device="cpu",
        )

        # Create history
        history = DialogueHistory(
            prompt="What is AI?",
            initial_response="AI is artificial intelligence.",
        )
        challenge = TeacherChallenge(ChallengeType.PROBE_REASONING, "Why?")
        turn = DialogueTurn(turn_number=1, challenge=challenge, student_response="Because...")
        history.add_turn(turn)

        formatted = generator._format_student_input(
            "What is AI?",
            challenge="New challenge",
            history=history,
        )

        # Should include history
        assert "Your initial response:" in formatted
        assert "Because..." in formatted


# ============================================================================
# DialogueFormatter Edge Cases
# ============================================================================


class TestDialogueFormatterEdgeCases:
    """Edge case tests for DialogueFormatter."""

    def test_formatter_all_format_types(self):
        """Test all supported format types."""
        format_types = ["standard", "chat", "instruction"]

        # Create minimal dialogue
        history = DialogueHistory(prompt="Test", initial_response="Response")
        eval = TeacherEvaluation(overall_score=7.0, dimension_scores=[], reasoning="OK")
        history.final_evaluation = eval

        dialogue = GeneratedDialogue(
            prompt="Test",
            initial_response="Response",
            history=history,
            final_evaluation=eval,
            turn_evaluations=[],
            metadata={},
        )

        for fmt_type in format_types:
            formatter = DialogueFormatter(format_type=fmt_type)
            formatted = formatter.format_dialogue(dialogue)

            assert isinstance(formatted, FormattedDialogue)
            assert formatted.input_text is not None
            assert formatted.target_text is not None

    def test_formatter_no_improved_response(self):
        """Formatter handles dialogue without improved response."""
        formatter = DialogueFormatter(format_type="standard")

        history = DialogueHistory(prompt="Test", initial_response="Response")
        eval = TeacherEvaluation(
            overall_score=7.0,
            dimension_scores=[],
            reasoning="OK",
            improved_response=None,  # No improved response
        )
        history.final_evaluation = eval

        dialogue = GeneratedDialogue(
            prompt="Test",
            initial_response="Response",
            history=history,
            final_evaluation=eval,
            turn_evaluations=[],
            metadata={},
        )

        formatted = formatter.format_dialogue(dialogue, use_improved_as_target=True)

        # Should fall back to initial_response
        assert formatted.target_text == "Response"

    def test_formatter_with_reasoning(self):
        """Formatter includes reasoning when configured and turns exist."""
        formatter = DialogueFormatter(format_type="instruction", include_reasoning=True)

        history = DialogueHistory(prompt="Test", initial_response="Initial Response")

        # Add a turn with evaluation (required for feedback to appear)
        turn_eval = TeacherEvaluation(
            overall_score=6.0,
            dimension_scores=[],
            reasoning="Good start, but could be more detailed",
        )
        turn = DialogueTurn(
            turn_number=1,
            challenge=TeacherChallenge(ChallengeType.PROBE_REASONING, "Can you elaborate?"),
            student_response="Here is more detail...",
            evaluation=turn_eval,
        )
        history.add_turn(turn)

        final_eval = TeacherEvaluation(
            overall_score=7.0,
            dimension_scores=[],
            reasoning="Good explanation with clear logic",
            strengths=["Clear", "Concise"],
            weaknesses=["Could add examples"],
        )
        history.final_evaluation = final_eval

        dialogue = GeneratedDialogue(
            prompt="Test",
            initial_response="Initial Response",
            history=history,
            final_evaluation=final_eval,
            turn_evaluations=[turn_eval],
            metadata={"num_turns": 1},
        )

        formatted = formatter.format_dialogue(dialogue)

        # Should include feedback section when turns exist and include_reasoning=True
        assert "### Feedback" in formatted.input_text
        assert "Good start" in formatted.input_text

    def test_formatter_many_turns(self):
        """Formatter handles dialogue with many turns."""
        formatter = DialogueFormatter(format_type="chat")

        history = DialogueHistory(prompt="Test", initial_response="Response")

        # Add 10 turns
        for i in range(10):
            challenge = TeacherChallenge(ChallengeType.PROBE_REASONING, f"Challenge {i}")
            turn = DialogueTurn(
                turn_number=i + 1,
                challenge=challenge,
                student_response=f"Response to challenge {i}",
            )
            history.add_turn(turn)

        eval = TeacherEvaluation(overall_score=7.0, dimension_scores=[], reasoning="OK")
        history.final_evaluation = eval

        dialogue = GeneratedDialogue(
            prompt="Test",
            initial_response="Response",
            history=history,
            final_evaluation=eval,
            turn_evaluations=[None] * 10,
            metadata={"num_turns": 10},
        )

        formatted = formatter.format_dialogue(dialogue)

        # Should include all turns
        for i in range(10):
            assert f"Challenge {i}" in formatted.full_conversation


# ============================================================================
# DialogueManager Cache Edge Cases
# ============================================================================


class TestDialogueManagerCacheEdgeCases:
    """Edge case tests for DialogueManager caching."""

    @pytest.fixture
    def mock_generator(self, mock_student_model, mock_tokenizer, mock_teacher):
        """Create mock generator."""
        generator = MagicMock()
        generator.teacher = mock_teacher

        async def mock_generate(prompt, **kwargs):
            history = DialogueHistory(prompt=prompt, initial_response="Generated")
            eval = TeacherEvaluation(overall_score=7.0, dimension_scores=[], reasoning="OK")
            history.final_evaluation = eval
            return GeneratedDialogue(
                prompt=prompt,
                initial_response="Generated",
                history=history,
                final_evaluation=eval,
                turn_evaluations=[],
                metadata={"num_turns": 0},
            )

        generator.generate_dialogue = AsyncMock(side_effect=mock_generate)
        return generator

    def test_cache_corruption_recovery(self, mock_generator, tmp_path):
        """Manager handles corrupted cache files."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "cache",
        )

        # Create corrupted cache file
        cache_key = manager._get_cache_key("test prompt")
        corrupt_path = manager.cache_dir / f"{cache_key}.json"
        corrupt_path.write_text("{ invalid json {{")

        # Load should return None and not crash
        loaded = manager._load_from_cache("test prompt")
        assert loaded is None

    def test_cache_empty_directory(self, mock_generator, tmp_path):
        """Manager handles empty cache directory."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "empty_cache",
        )

        # Iterate empty cache
        cached = list(manager.iterate_cached())
        assert len(cached) == 0

        # Stats for empty cache
        stats = manager.cache_stats()
        assert stats["count"] == 0
        assert stats["size_bytes"] == 0

    @pytest.mark.asyncio
    async def test_cache_concurrent_writes(self, mock_generator, tmp_path):
        """Manager handles concurrent cache writes."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "concurrent_cache",
        )

        # Simulate concurrent writes (same prompt)
        async def write_and_read(suffix):
            prompt = f"concurrent prompt {suffix}"
            dialogue = await manager.get_dialogue(prompt)
            return dialogue

        # Run concurrently
        results = await asyncio.gather(*[write_and_read(i) for i in range(5)])

        # All should complete without error
        assert len(results) == 5
        for result in results:
            assert result is not None

    def test_cache_unicode_prompts(self, mock_generator, tmp_path):
        """Manager handles unicode prompts in cache keys."""
        manager = DialogueManager(
            generator=mock_generator,
            cache_dir=tmp_path / "unicode_cache",
        )

        unicode_prompts = [
            "What is 机器学习?",
            "Qu'est-ce que l'IA?",
            "🤖 AI rocks! 🎉",
        ]

        for prompt in unicode_prompts:
            key = manager._get_cache_key(prompt)
            # Should be valid hex hash
            assert all(c in "0123456789abcdef" for c in key)


# ============================================================================
# DialogueHistory Edge Cases
# ============================================================================


class TestDialogueHistoryEdgeCases:
    """Edge case tests for DialogueHistory."""

    def test_history_json_serialization(self):
        """DialogueHistory can be serialized to JSON."""
        history = DialogueHistory(
            prompt="Test prompt",
            initial_response="Test response",
        )

        # Add a turn
        challenge = TeacherChallenge(ChallengeType.PROBE_REASONING, "Why?")
        eval = TeacherEvaluation(overall_score=7.0, dimension_scores=[], reasoning="OK")
        turn = DialogueTurn(turn_number=1, challenge=challenge, student_response="Because", evaluation=eval)
        history.add_turn(turn)

        # Serialize
        data = {
            "prompt": history.prompt,
            "initial_response": history.initial_response,
            "num_turns": history.num_turns,
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "challenge_content": t.challenge.content,
                    "student_response": t.student_response,
                }
                for t in history.turns
            ],
        }

        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        assert parsed["prompt"] == "Test prompt"
        assert parsed["num_turns"] == 1

    def test_history_empty_turns(self):
        """DialogueHistory with no turns."""
        history = DialogueHistory(
            prompt="Test",
            initial_response="Response",
        )

        assert history.num_turns == 0
        assert history.turns == []
        assert history.get_trajectory_scores() == []

    def test_history_mixed_evaluations(self):
        """DialogueHistory with some turns evaluated, some not."""
        history = DialogueHistory(prompt="Test", initial_response="Response")

        # Turn 1: evaluated
        t1 = DialogueTurn(
            turn_number=1,
            challenge=TeacherChallenge(ChallengeType.PROBE_REASONING, "Q1"),
            student_response="A1",
            evaluation=TeacherEvaluation(overall_score=6.0, dimension_scores=[], reasoning="OK"),
        )
        history.add_turn(t1)

        # Turn 2: not evaluated
        t2 = DialogueTurn(
            turn_number=2,
            challenge=TeacherChallenge(ChallengeType.EDGE_CASE, "Q2"),
            student_response="A2",
            evaluation=None,
        )
        history.add_turn(t2)

        # Turn 3: evaluated
        t3 = DialogueTurn(
            turn_number=3,
            challenge=TeacherChallenge(ChallengeType.STEELMAN, "Q3"),
            student_response="A3",
            evaluation=TeacherEvaluation(overall_score=8.0, dimension_scores=[], reasoning="Good"),
        )
        history.add_turn(t3)

        # Trajectory scores should only include evaluated turns
        scores = history.get_trajectory_scores()
        assert scores == [6.0, 8.0]


# ============================================================================
# Windows Compatibility Entry Point
# ============================================================================


if __name__ == "__main__":
    freeze_support()
    pytest.main([__file__, "-v"])
