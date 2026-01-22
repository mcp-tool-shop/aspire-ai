"""
Tests for OpenAI Teacher (aspire/teachers/openai.py).

Coverage target: OpenAI teacher initialization, challenge generation, and evaluation.
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
from aspire.teachers.openai import OpenAITeacher, OpenAITeacherError


# ============================================================================
# Initialization Tests
# ============================================================================

class TestOpenAITeacherInit:
    """Tests for OpenAITeacher initialization."""

    def test_init_with_api_key_direct(self):
        """Test OpenAITeacher initialization with API key passed directly."""
        with patch("aspire.teachers.openai.AsyncOpenAI") as mock_client:
            teacher = OpenAITeacher(api_key="sk-test-key-12345")

            assert teacher.model == "gpt-4o"
            assert teacher.name == "GPT-4 Teacher"
            mock_client.assert_called_once_with(api_key="sk-test-key-12345")

    def test_init_with_api_key_from_env(self):
        """Test OpenAITeacher initialization with API key from environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-key"}):
            with patch("aspire.teachers.openai.AsyncOpenAI") as mock_client:
                teacher = OpenAITeacher()

                mock_client.assert_called_once_with(api_key="sk-env-key")

    def test_init_raises_error_without_api_key(self):
        """Test OpenAITeacher raises OpenAITeacherError without API key."""
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(OpenAITeacherError) as exc_info:
                OpenAITeacher()

            assert "OPENAI_API_KEY not found" in str(exc_info.value)
            assert "https://platform.openai.com" in str(exc_info.value)

    def test_init_with_custom_model(self):
        """Test OpenAITeacher initialization with custom model."""
        with patch("aspire.teachers.openai.AsyncOpenAI"):
            teacher = OpenAITeacher(
                api_key="test-key",
                model="gpt-4-turbo"
            )

            assert teacher.model == "gpt-4-turbo"

    def test_init_with_custom_name_and_description(self):
        """Test OpenAITeacher initialization with custom name and description."""
        with patch("aspire.teachers.openai.AsyncOpenAI"):
            teacher = OpenAITeacher(
                api_key="test-key",
                name="Custom GPT",
                description="A custom GPT teacher"
            )

            assert teacher.name == "Custom GPT"
            assert teacher.description == "A custom GPT teacher"

    def test_init_with_custom_temperature_and_tokens(self):
        """Test OpenAITeacher initialization with custom temperature and max_tokens."""
        with patch("aspire.teachers.openai.AsyncOpenAI"):
            teacher = OpenAITeacher(
                api_key="test-key",
                temperature=0.3,
                max_tokens=4096
            )

            assert teacher.temperature == 0.3
            assert teacher.max_tokens == 4096


# ============================================================================
# Challenge Tests
# ============================================================================

class TestOpenAITeacherChallenge:
    """Tests for OpenAITeacher.challenge method."""

    @pytest.fixture
    def mock_teacher(self):
        """Create a mocked OpenAITeacher."""
        with patch("aspire.teachers.openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            teacher = OpenAITeacher(api_key="test-key")
            teacher.client = mock_client
            yield teacher, mock_client

    @pytest.mark.asyncio
    async def test_challenge_returns_teacher_challenge(
        self, mock_teacher, mock_openai_response_challenge
    ):
        """Test OpenAITeacher.challenge returns TeacherChallenge (mock API)."""
        teacher, mock_client = mock_teacher

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(mock_openai_response_challenge)))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await teacher.challenge(
            prompt="Explain recursion",
            student_response="Recursion is a function calling itself."
        )

        assert isinstance(result, TeacherChallenge)
        assert result.content == mock_openai_response_challenge["challenge"]
        assert result.difficulty == mock_openai_response_challenge["difficulty"]

    @pytest.mark.asyncio
    async def test_challenge_with_different_challenge_types(self, mock_teacher):
        """Test OpenAITeacher.challenge with different challenge types (mock API)."""
        teacher, mock_client = mock_teacher

        challenge_types = [
            ChallengeType.PROBE_REASONING,
            ChallengeType.EDGE_CASE,
            ChallengeType.CLARIFICATION,
            ChallengeType.EXTENSION,
        ]

        for challenge_type in challenge_types:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content=json.dumps({
                    "challenge": f"A {challenge_type.value} challenge",
                    "context": "Testing",
                    "difficulty": 0.6
                })))
            ]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await teacher.challenge(
                prompt="Test prompt",
                student_response="Test response",
                challenge_type=challenge_type
            )

            assert result.challenge_type == challenge_type

    @pytest.mark.asyncio
    async def test_challenge_with_dialogue_history(
        self, mock_teacher, sample_dialogue_history
    ):
        """Test OpenAITeacher.challenge with dialogue history."""
        teacher, mock_client = mock_teacher

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "challenge": "Follow-up question",
                "context": "Based on history",
                "difficulty": 0.7
            })))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await teacher.challenge(
            prompt=sample_dialogue_history.prompt,
            student_response="Updated response",
            dialogue_history=sample_dialogue_history
        )

        # Verify API was called
        assert mock_client.chat.completions.create.called
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))

        # Find user message with history
        user_message = [m for m in messages if m["role"] == "user"][0]
        assert "Previous dialogue" in user_message["content"]

    @pytest.mark.asyncio
    async def test_challenge_handles_json_parse_error_gracefully(self, mock_teacher):
        """Test OpenAITeacher handles JSON parse errors gracefully."""
        teacher, mock_client = mock_teacher

        # Invalid JSON response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="This is not valid JSON challenge."))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await teacher.challenge(
            prompt="Test",
            student_response="Test response"
        )

        # Should fall back to using raw response as challenge
        assert isinstance(result, TeacherChallenge)
        assert "This is not valid JSON" in result.content
        assert result.difficulty == 0.5

    @pytest.mark.asyncio
    async def test_challenge_uses_json_response_format(self, mock_teacher):
        """Test OpenAITeacher.challenge requests JSON response format."""
        teacher, mock_client = mock_teacher

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "challenge": "Test",
                "context": "Test",
                "difficulty": 0.5
            })))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await teacher.challenge(
            prompt="Test",
            student_response="Response"
        )

        # Verify response_format was set
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs.get("response_format") == {"type": "json_object"}


# ============================================================================
# Evaluation Tests
# ============================================================================

class TestOpenAITeacherEvaluate:
    """Tests for OpenAITeacher.evaluate method."""

    @pytest.fixture
    def mock_teacher(self):
        """Create a mocked OpenAITeacher."""
        with patch("aspire.teachers.openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            teacher = OpenAITeacher(api_key="test-key")
            teacher.client = mock_client
            yield teacher, mock_client

    @pytest.mark.asyncio
    async def test_evaluate_returns_teacher_evaluation(
        self, mock_teacher, mock_openai_response_evaluate
    ):
        """Test OpenAITeacher.evaluate returns TeacherEvaluation (mock API)."""
        teacher, mock_client = mock_teacher

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(mock_openai_response_evaluate)))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await teacher.evaluate(
            prompt="Explain recursion",
            student_response="Recursion is a function calling itself."
        )

        assert isinstance(result, TeacherEvaluation)
        assert result.overall_score == mock_openai_response_evaluate["overall_score"]
        assert result.reasoning == mock_openai_response_evaluate["reasoning"]

    @pytest.mark.asyncio
    async def test_evaluate_with_generate_improved_true(
        self, mock_teacher, mock_openai_response_evaluate
    ):
        """Test OpenAITeacher.evaluate with generate_improved=True (mock API)."""
        teacher, mock_client = mock_teacher

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(mock_openai_response_evaluate)))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response",
            generate_improved=True
        )

        assert result.improved_response == mock_openai_response_evaluate["improved_response"]

    @pytest.mark.asyncio
    async def test_evaluate_parses_dimension_scores(self, mock_teacher):
        """Test OpenAITeacher.evaluate correctly parses dimension scores."""
        teacher, mock_client = mock_teacher

        response_data = {
            "overall_score": 7.5,
            "dimension_scores": [
                {"dimension": "correctness", "score": 8.0, "explanation": "Accurate"},
                {"dimension": "reasoning", "score": 7.0, "explanation": "Good logic"},
            ],
            "reasoning": "Good job",
            "strengths": ["Clear"],
            "weaknesses": [],
            "suggestions": []
        }
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(response_data)))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response"
        )

        assert len(result.dimension_scores) == 2
        assert result.dimension_scores[0].dimension == EvaluationDimension.CORRECTNESS
        assert result.dimension_scores[0].score == 8.0

    @pytest.mark.asyncio
    async def test_evaluate_handles_json_parse_error(self, mock_teacher):
        """Test OpenAITeacher.evaluate handles JSON parse errors."""
        teacher, mock_client = mock_teacher

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Not valid JSON evaluation."))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response"
        )

        assert isinstance(result, TeacherEvaluation)
        assert result.overall_score == 5.0
        assert "Not valid JSON" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_skips_invalid_dimensions(self, mock_teacher):
        """Test OpenAITeacher.evaluate skips invalid dimension names."""
        teacher, mock_client = mock_teacher

        response_data = {
            "overall_score": 7.0,
            "dimension_scores": [
                {"dimension": "correctness", "score": 8.0, "explanation": "Good"},
                {"dimension": "invalid_dimension", "score": 5.0, "explanation": "Skip"},
                {"dimension": "clarity", "score": 7.0, "explanation": "Clear"},
            ],
            "reasoning": "Overall good",
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(response_data)))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response"
        )

        # Only valid dimensions should be included
        assert len(result.dimension_scores) == 2
        dimension_names = [ds.dimension for ds in result.dimension_scores]
        assert EvaluationDimension.CORRECTNESS in dimension_names
        assert EvaluationDimension.CLARITY in dimension_names

    @pytest.mark.asyncio
    async def test_evaluate_uses_lower_temperature(self, mock_teacher):
        """Test OpenAITeacher.evaluate uses lower temperature for consistency."""
        teacher, mock_client = mock_teacher

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "overall_score": 7.0,
                "dimension_scores": [],
                "reasoning": "Test",
                "strengths": [],
                "weaknesses": [],
                "suggestions": []
            })))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await teacher.evaluate(
            prompt="Test",
            student_response="Response"
        )

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs.get("temperature") == 0.3


# ============================================================================
# System Prompt Tests
# ============================================================================

class TestOpenAITeacherSystemPrompt:
    """Tests for OpenAITeacher system prompt."""

    def test_system_prompt_includes_name(self):
        """Test system prompt includes teacher name."""
        with patch("aspire.teachers.openai.AsyncOpenAI"):
            teacher = OpenAITeacher(api_key="test-key", name="Test GPT")

            prompt = teacher.get_system_prompt()
            assert "Test GPT" in prompt

    def test_system_prompt_includes_aspire_context(self):
        """Test system prompt includes ASPIRE context."""
        with patch("aspire.teachers.openai.AsyncOpenAI"):
            teacher = OpenAITeacher(api_key="test-key")

            prompt = teacher.get_system_prompt()
            assert "ASPIRE" in prompt


# ============================================================================
# Repr Tests
# ============================================================================

class TestOpenAITeacherRepr:
    """Tests for OpenAITeacher string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        with patch("aspire.teachers.openai.AsyncOpenAI"):
            teacher = OpenAITeacher(api_key="test-key", name="My GPT")

            repr_str = repr(teacher)
            assert "OpenAITeacher" in repr_str
            assert "My GPT" in repr_str
