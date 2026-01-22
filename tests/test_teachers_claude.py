"""
Tests for Claude Teacher (aspire/teachers/claude.py).

Coverage target: Claude teacher initialization, challenge generation, and evaluation.
All tests use mocks to avoid real API calls.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aspire.teachers.base import (
    ChallengeType,
    DialogueHistory,
    EvaluationDimension,
    TeacherChallenge,
    TeacherEvaluation,
)
from aspire.teachers.claude import ClaudeTeacher, ClaudeTeacherError


# ============================================================================
# Initialization Tests
# ============================================================================

class TestClaudeTeacherInit:
    """Tests for ClaudeTeacher initialization."""

    def test_init_with_api_key_direct(self):
        """Test ClaudeTeacher initialization with API key passed directly."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic") as mock_client:
            teacher = ClaudeTeacher(api_key="sk-ant-test-key-12345")

            assert teacher.model == "claude-sonnet-4-20250514"
            assert teacher.name == "Claude Teacher"
            mock_client.assert_called_once_with(api_key="sk-ant-test-key-12345")

    def test_init_with_api_key_from_env(self):
        """Test ClaudeTeacher initialization with API key from environment."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env-key"}):
            with patch("aspire.teachers.claude.anthropic.AsyncAnthropic") as mock_client:
                teacher = ClaudeTeacher()

                mock_client.assert_called_once_with(api_key="sk-ant-env-key")

    def test_init_raises_error_without_api_key(self):
        """Test ClaudeTeacher raises ClaudeTeacherError without API key."""
        # Clear environment variable
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ClaudeTeacherError) as exc_info:
                ClaudeTeacher()

            assert "ANTHROPIC_API_KEY not found" in str(exc_info.value)
            assert "https://console.anthropic.com" in str(exc_info.value)

    def test_init_with_custom_model(self):
        """Test ClaudeTeacher initialization with custom model."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic"):
            teacher = ClaudeTeacher(
                api_key="test-key",
                model="claude-opus-4-20250514"
            )

            assert teacher.model == "claude-opus-4-20250514"

    def test_init_with_custom_name_and_description(self):
        """Test ClaudeTeacher initialization with custom name and description."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic"):
            teacher = ClaudeTeacher(
                api_key="test-key",
                name="Custom Claude",
                description="A custom Claude teacher"
            )

            assert teacher.name == "Custom Claude"
            assert teacher.description == "A custom Claude teacher"

    def test_init_with_custom_temperature_and_tokens(self):
        """Test ClaudeTeacher initialization with custom temperature and max_tokens."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic"):
            teacher = ClaudeTeacher(
                api_key="test-key",
                temperature=0.5,
                max_tokens=2048
            )

            assert teacher.temperature == 0.5
            assert teacher.max_tokens == 2048


# ============================================================================
# Challenge Tests
# ============================================================================

class TestClaudeTeacherChallenge:
    """Tests for ClaudeTeacher.challenge method."""

    @pytest.fixture
    def mock_teacher(self):
        """Create a mocked ClaudeTeacher."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            teacher = ClaudeTeacher(api_key="test-key")
            teacher.client = mock_client
            yield teacher, mock_client

    @pytest.mark.asyncio
    async def test_challenge_returns_teacher_challenge(
        self, mock_teacher, mock_claude_response_challenge
    ):
        """Test ClaudeTeacher.challenge returns TeacherChallenge (mock API)."""
        teacher, mock_client = mock_teacher

        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(mock_claude_response_challenge))]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await teacher.challenge(
            prompt="Explain recursion",
            student_response="Recursion is a function calling itself."
        )

        assert isinstance(result, TeacherChallenge)
        assert result.content == mock_claude_response_challenge["challenge"]
        assert result.difficulty == mock_claude_response_challenge["difficulty"]

    @pytest.mark.asyncio
    async def test_challenge_with_different_challenge_types(self, mock_teacher):
        """Test ClaudeTeacher.challenge with different challenge types (mock API)."""
        teacher, mock_client = mock_teacher

        challenge_types = [
            ChallengeType.PROBE_REASONING,
            ChallengeType.EDGE_CASE,
            ChallengeType.DEVILS_ADVOCATE,
            ChallengeType.SOCRATIC,
        ]

        for challenge_type in challenge_types:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text=json.dumps({
                "challenge": f"A {challenge_type.value} challenge",
                "context": "Testing",
                "difficulty": 0.5
            }))]
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            result = await teacher.challenge(
                prompt="Test prompt",
                student_response="Test response",
                challenge_type=challenge_type
            )

            assert result.challenge_type == challenge_type

    @pytest.mark.asyncio
    async def test_challenge_builds_history_context(
        self, mock_teacher, sample_dialogue_history
    ):
        """Test ClaudeTeacher.challenge builds history context."""
        teacher, mock_client = mock_teacher

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "challenge": "Follow-up question",
            "context": "Based on history",
            "difficulty": 0.7
        }))]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await teacher.challenge(
            prompt=sample_dialogue_history.prompt,
            student_response="Updated response",
            dialogue_history=sample_dialogue_history
        )

        # Verify API was called with history context
        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        assert len(messages) > 0
        message_content = messages[0]["content"]
        assert "Previous dialogue" in message_content

    @pytest.mark.asyncio
    async def test_challenge_handles_json_in_code_block(self, mock_teacher):
        """Test ClaudeTeacher.challenge handles JSON in markdown code blocks."""
        teacher, mock_client = mock_teacher

        # Response with JSON in code block
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""Here's my challenge:
```json
{
    "challenge": "What about edge cases?",
    "context": "Testing limits",
    "difficulty": 0.8
}
```
""")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await teacher.challenge(
            prompt="Test",
            student_response="Test response"
        )

        assert result.content == "What about edge cases?"
        assert result.difficulty == 0.8

    @pytest.mark.asyncio
    async def test_challenge_handles_json_parse_error_gracefully(self, mock_teacher):
        """Test ClaudeTeacher handles JSON parse errors gracefully."""
        teacher, mock_client = mock_teacher

        # Invalid JSON response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is not valid JSON, but a plain challenge question.")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await teacher.challenge(
            prompt="Test",
            student_response="Test response"
        )

        # Should fall back to using raw response as challenge
        assert isinstance(result, TeacherChallenge)
        assert "This is not valid JSON" in result.content
        assert result.difficulty == 0.5  # Default difficulty


# ============================================================================
# Evaluation Tests
# ============================================================================

class TestClaudeTeacherEvaluate:
    """Tests for ClaudeTeacher.evaluate method."""

    @pytest.fixture
    def mock_teacher(self):
        """Create a mocked ClaudeTeacher."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            teacher = ClaudeTeacher(api_key="test-key")
            teacher.client = mock_client
            yield teacher, mock_client

    @pytest.mark.asyncio
    async def test_evaluate_returns_teacher_evaluation(
        self, mock_teacher, mock_claude_response_evaluate
    ):
        """Test ClaudeTeacher.evaluate returns TeacherEvaluation (mock API)."""
        teacher, mock_client = mock_teacher

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(mock_claude_response_evaluate))]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await teacher.evaluate(
            prompt="Explain recursion",
            student_response="Recursion is a function calling itself."
        )

        assert isinstance(result, TeacherEvaluation)
        assert result.overall_score == mock_claude_response_evaluate["overall_score"]
        assert result.reasoning == mock_claude_response_evaluate["reasoning"]
        assert len(result.dimension_scores) > 0

    @pytest.mark.asyncio
    async def test_evaluate_with_generate_improved_true(
        self, mock_teacher, mock_claude_response_evaluate
    ):
        """Test ClaudeTeacher.evaluate with generate_improved=True (mock API)."""
        teacher, mock_client = mock_teacher

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(mock_claude_response_evaluate))]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response",
            generate_improved=True
        )

        assert result.improved_response == mock_claude_response_evaluate["improved_response"]

    @pytest.mark.asyncio
    async def test_evaluate_with_generate_improved_false(self, mock_teacher):
        """Test ClaudeTeacher.evaluate with generate_improved=False (mock API)."""
        teacher, mock_client = mock_teacher

        response_data = {
            "overall_score": 7.0,
            "dimension_scores": [],
            "reasoning": "Good response",
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
            # No improved_response
        }
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(response_data))]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response",
            generate_improved=False
        )

        assert result.improved_response is None

    @pytest.mark.asyncio
    async def test_evaluate_parses_dimension_scores(self, mock_teacher):
        """Test ClaudeTeacher.evaluate correctly parses dimension scores."""
        teacher, mock_client = mock_teacher

        response_data = {
            "overall_score": 8.0,
            "dimension_scores": [
                {"dimension": "correctness", "score": 9.0, "explanation": "Very accurate"},
                {"dimension": "reasoning", "score": 8.0, "explanation": "Good logic"},
                {"dimension": "clarity", "score": 7.5, "explanation": "Clear"},
            ],
            "reasoning": "Well done",
            "strengths": ["Accurate"],
            "weaknesses": [],
            "suggestions": []
        }
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(response_data))]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response"
        )

        assert len(result.dimension_scores) == 3
        assert result.dimension_scores[0].dimension == EvaluationDimension.CORRECTNESS
        assert result.dimension_scores[0].score == 9.0

    @pytest.mark.asyncio
    async def test_evaluate_handles_json_parse_error(self, mock_teacher):
        """Test ClaudeTeacher.evaluate handles JSON parse errors."""
        teacher, mock_client = mock_teacher

        # Invalid JSON
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is not valid JSON evaluation.")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response"
        )

        # Should return fallback evaluation
        assert isinstance(result, TeacherEvaluation)
        assert result.overall_score == 5.0
        assert "This is not valid JSON" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_with_dialogue_history(
        self, mock_teacher, sample_dialogue_history, mock_claude_response_evaluate
    ):
        """Test ClaudeTeacher.evaluate builds dialogue history context."""
        teacher, mock_client = mock_teacher

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(mock_claude_response_evaluate))]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await teacher.evaluate(
            prompt=sample_dialogue_history.prompt,
            student_response="Final response",
            dialogue_history=sample_dialogue_history
        )

        # Verify API was called with history context
        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        message_content = messages[0]["content"]
        assert "Dialogue history" in message_content


# ============================================================================
# Challenge Description Tests
# ============================================================================

class TestClaudeTeacherChallengeDescriptions:
    """Tests for ClaudeTeacher._get_challenge_description."""

    def test_get_challenge_description_for_all_types(self):
        """Test ClaudeTeacher._get_challenge_description for all challenge types."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic"):
            teacher = ClaudeTeacher(api_key="test-key")

            for challenge_type in ChallengeType:
                description = teacher._get_challenge_description(challenge_type)
                assert isinstance(description, str)
                assert len(description) > 0

    def test_challenge_description_probe_reasoning(self):
        """Test _get_challenge_description for PROBE_REASONING."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic"):
            teacher = ClaudeTeacher(api_key="test-key")

            desc = teacher._get_challenge_description(ChallengeType.PROBE_REASONING)
            assert "why" in desc.lower() or "how" in desc.lower()

    def test_challenge_description_edge_case(self):
        """Test _get_challenge_description for EDGE_CASE."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic"):
            teacher = ClaudeTeacher(api_key="test-key")

            desc = teacher._get_challenge_description(ChallengeType.EDGE_CASE)
            assert "edge" in desc.lower() or "limit" in desc.lower()

    def test_challenge_description_devils_advocate(self):
        """Test _get_challenge_description for DEVILS_ADVOCATE."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic"):
            teacher = ClaudeTeacher(api_key="test-key")

            desc = teacher._get_challenge_description(ChallengeType.DEVILS_ADVOCATE)
            assert "opposite" in desc.lower() or "argue" in desc.lower()

    def test_challenge_description_socratic(self):
        """Test _get_challenge_description for SOCRATIC."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic"):
            teacher = ClaudeTeacher(api_key="test-key")

            desc = teacher._get_challenge_description(ChallengeType.SOCRATIC)
            assert "assumption" in desc.lower() or "question" in desc.lower()


# ============================================================================
# System Prompt Tests
# ============================================================================

class TestClaudeTeacherSystemPrompt:
    """Tests for ClaudeTeacher system prompt."""

    def test_system_prompt_includes_name(self):
        """Test system prompt includes teacher name."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic"):
            teacher = ClaudeTeacher(api_key="test-key", name="Test Claude")

            prompt = teacher.get_system_prompt()
            assert "Test Claude" in prompt

    def test_system_prompt_includes_description(self):
        """Test system prompt includes teacher description."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic"):
            teacher = ClaudeTeacher(
                api_key="test-key",
                description="A very special teacher"
            )

            prompt = teacher.get_system_prompt()
            assert "very special teacher" in prompt

    def test_system_prompt_includes_aspire_context(self):
        """Test system prompt includes ASPIRE context."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic"):
            teacher = ClaudeTeacher(api_key="test-key")

            prompt = teacher.get_system_prompt()
            assert "ASPIRE" in prompt


# ============================================================================
# Repr Tests
# ============================================================================

class TestClaudeTeacherRepr:
    """Tests for ClaudeTeacher string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        with patch("aspire.teachers.claude.anthropic.AsyncAnthropic"):
            teacher = ClaudeTeacher(api_key="test-key", name="My Claude")

            repr_str = repr(teacher)
            assert "ClaudeTeacher" in repr_str
            assert "My Claude" in repr_str
