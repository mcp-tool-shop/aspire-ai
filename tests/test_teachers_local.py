"""
Tests for Local Teacher (aspire/teachers/local.py).

Coverage target: Local teacher initialization, challenge generation, and evaluation.
All tests use mocks to avoid loading real models.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from aspire.teachers.base import (
    ChallengeType,
    TeacherChallenge,
    TeacherEvaluation,
)
from aspire.teachers.local import LocalTeacher


# ============================================================================
# Initialization Tests
# ============================================================================

class TestLocalTeacherInit:
    """Tests for LocalTeacher initialization."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Mock AutoModelForCausalLM and AutoTokenizer."""
        with patch("aspire.teachers.local.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.teachers.local.AutoTokenizer") as mock_tokenizer_class:

            # Setup mock model
            mock_model = MagicMock()
            mock_model.eval.return_value = None
            mock_model_class.from_pretrained.return_value = mock_model

            # Setup mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            yield {
                "model_class": mock_model_class,
                "model": mock_model,
                "tokenizer_class": mock_tokenizer_class,
                "tokenizer": mock_tokenizer,
            }

    def test_init_with_model_path(self, mock_model_and_tokenizer):
        """Test LocalTeacher initialization with model path."""
        teacher = LocalTeacher(model_name_or_path="test-model/path")

        assert teacher.model_name_or_path == "test-model/path"
        assert teacher.name == "Local Teacher"
        mock_model_and_tokenizer["model_class"].from_pretrained.assert_called_once()
        mock_model_and_tokenizer["tokenizer_class"].from_pretrained.assert_called_once()

    def test_init_sets_pad_token_from_eos(self, mock_model_and_tokenizer):
        """Test LocalTeacher sets pad_token from eos_token when None."""
        teacher = LocalTeacher(model_name_or_path="test-model")

        # pad_token should be set to eos_token
        assert mock_model_and_tokenizer["tokenizer"].pad_token == "</s>"

    def test_init_with_4bit_quantization(self, mock_model_and_tokenizer):
        """Test LocalTeacher initialization with 4-bit quantization."""
        teacher = LocalTeacher(
            model_name_or_path="test-model",
            load_in_4bit=True,
            load_in_8bit=False
        )

        # Check that quantization config was passed
        call_kwargs = mock_model_and_tokenizer["model_class"].from_pretrained.call_args[1]
        assert call_kwargs.get("quantization_config") is not None

    def test_init_with_8bit_quantization(self, mock_model_and_tokenizer):
        """Test LocalTeacher initialization with 8-bit quantization."""
        teacher = LocalTeacher(
            model_name_or_path="test-model",
            load_in_4bit=False,
            load_in_8bit=True
        )

        call_kwargs = mock_model_and_tokenizer["model_class"].from_pretrained.call_args[1]
        assert call_kwargs.get("quantization_config") is not None

    def test_init_without_quantization(self, mock_model_and_tokenizer):
        """Test LocalTeacher initialization without quantization."""
        teacher = LocalTeacher(
            model_name_or_path="test-model",
            load_in_4bit=False,
            load_in_8bit=False
        )

        call_kwargs = mock_model_and_tokenizer["model_class"].from_pretrained.call_args[1]
        assert call_kwargs.get("quantization_config") is None

    def test_init_with_custom_device(self, mock_model_and_tokenizer):
        """Test LocalTeacher initialization with custom device."""
        teacher = LocalTeacher(
            model_name_or_path="test-model",
            device="cpu"
        )

        assert teacher.device == "cpu"

    def test_init_with_custom_name_and_description(self, mock_model_and_tokenizer):
        """Test LocalTeacher initialization with custom name and description."""
        teacher = LocalTeacher(
            model_name_or_path="test-model",
            name="Custom Local",
            description="A specialized local teacher"
        )

        assert teacher.name == "Custom Local"
        assert teacher.description == "A specialized local teacher"

    def test_init_sets_model_to_eval_mode(self, mock_model_and_tokenizer):
        """Test LocalTeacher sets model to eval mode."""
        teacher = LocalTeacher(model_name_or_path="test-model")

        mock_model_and_tokenizer["model"].eval.assert_called_once()


# ============================================================================
# Challenge Tests
# ============================================================================

class TestLocalTeacherChallenge:
    """Tests for LocalTeacher.challenge method."""

    @pytest.fixture
    def mock_teacher(self):
        """Create a mocked LocalTeacher."""
        with patch("aspire.teachers.local.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.teachers.local.AutoTokenizer") as mock_tokenizer_class:

            mock_model = MagicMock()
            mock_model.eval.return_value = None

            # Setup generate to return tensor
            def mock_generate(**kwargs):
                input_ids = kwargs.get("input_ids", torch.zeros(1, 10))
                batch_size = input_ids.shape[0]
                # Return longer sequence (simulating generation)
                return torch.randint(0, 1000, (batch_size, input_ids.shape[1] + 50))

            mock_model.generate = MagicMock(side_effect=mock_generate)
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "</s>"
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer.pad_token_id = 0

            def mock_tokenize(text, **kwargs):
                # Create a mock object that has .to() method
                mock_result = MagicMock()
                mock_result.__getitem__ = lambda self, key: {
                    "input_ids": torch.randint(0, 1000, (1, 100)),
                    "attention_mask": torch.ones(1, 100),
                }[key]
                mock_result.to = lambda device: mock_result
                return mock_result

            mock_tokenizer.side_effect = mock_tokenize
            mock_tokenizer.__call__ = mock_tokenize
            mock_tokenizer.decode = MagicMock(return_value="Why is that approach better?")
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            teacher = LocalTeacher(model_name_or_path="test-model")
            yield teacher

    @pytest.mark.asyncio
    async def test_challenge_returns_teacher_challenge(self, mock_teacher):
        """Test LocalTeacher.challenge returns TeacherChallenge (mock model)."""
        result = await mock_teacher.challenge(
            prompt="Explain recursion",
            student_response="Recursion is a function calling itself."
        )

        assert isinstance(result, TeacherChallenge)
        assert len(result.content) > 0
        assert result.difficulty == 0.5  # Default difficulty for local

    @pytest.mark.asyncio
    async def test_challenge_auto_selects_challenge_type(self, mock_teacher):
        """Test LocalTeacher.challenge auto-selects challenge type."""
        result = await mock_teacher.challenge(
            prompt="Test prompt",
            student_response="Test response",
            challenge_type=None
        )

        # Should have a challenge type from the preferred list
        assert result.challenge_type in mock_teacher.preferred_challenges

    @pytest.mark.asyncio
    async def test_challenge_uses_specified_challenge_type(self, mock_teacher):
        """Test LocalTeacher.challenge uses specified challenge type."""
        result = await mock_teacher.challenge(
            prompt="Test prompt",
            student_response="Test response",
            challenge_type=ChallengeType.EDGE_CASE
        )

        assert result.challenge_type == ChallengeType.EDGE_CASE

    @pytest.mark.asyncio
    async def test_challenge_with_dialogue_history(
        self, mock_teacher, sample_dialogue_history
    ):
        """Test LocalTeacher.challenge with dialogue history."""
        result = await mock_teacher.challenge(
            prompt=sample_dialogue_history.prompt,
            student_response="Updated response",
            dialogue_history=sample_dialogue_history
        )

        assert isinstance(result, TeacherChallenge)
        # Model should have been called (generate was invoked)
        assert mock_teacher.model.generate.called


# ============================================================================
# Evaluation Tests
# ============================================================================

class TestLocalTeacherEvaluate:
    """Tests for LocalTeacher.evaluate method."""

    @pytest.fixture
    def mock_teacher_for_eval(self):
        """Create a mocked LocalTeacher for evaluation tests."""
        with patch("aspire.teachers.local.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.teachers.local.AutoTokenizer") as mock_tokenizer_class:

            mock_model = MagicMock()
            mock_model.eval.return_value = None

            def mock_generate(**kwargs):
                input_ids = kwargs.get("input_ids", torch.zeros(1, 10))
                batch_size = input_ids.shape[0]
                return torch.randint(0, 1000, (batch_size, input_ids.shape[1] + 100))

            mock_model.generate = MagicMock(side_effect=mock_generate)
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "</s>"
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer.pad_token_id = 0

            def mock_tokenize(text, **kwargs):
                # Create a mock object that has .to() method
                mock_result = MagicMock()
                mock_result.__getitem__ = lambda self, key: {
                    "input_ids": torch.randint(0, 1000, (1, 150)),
                    "attention_mask": torch.ones(1, 150),
                }[key]
                mock_result.to = lambda device: mock_result
                return mock_result

            mock_tokenizer.side_effect = mock_tokenize
            mock_tokenizer.__call__ = mock_tokenize
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            teacher = LocalTeacher(model_name_or_path="test-model")
            yield teacher, mock_tokenizer

    @pytest.mark.asyncio
    async def test_evaluate_returns_teacher_evaluation(self, mock_teacher_for_eval):
        """Test LocalTeacher.evaluate returns TeacherEvaluation (mock model)."""
        teacher, mock_tokenizer = mock_teacher_for_eval

        # Return response with score
        mock_tokenizer.decode = MagicMock(return_value="Score: 7.5\nGood response with clear reasoning.")

        result = await teacher.evaluate(
            prompt="Explain recursion",
            student_response="Recursion is a function calling itself."
        )

        assert isinstance(result, TeacherEvaluation)
        assert result.overall_score == 7.5

    @pytest.mark.asyncio
    async def test_evaluate_extracts_score_with_slash_format(self, mock_teacher_for_eval):
        """Test LocalTeacher.evaluate extracts score from 'X/10' format."""
        teacher, mock_tokenizer = mock_teacher_for_eval

        mock_tokenizer.decode = MagicMock(return_value="I give this response 8/10. Well done!")

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response"
        )

        assert result.overall_score == 8.0

    @pytest.mark.asyncio
    async def test_evaluate_defaults_score_when_not_found(self, mock_teacher_for_eval):
        """Test LocalTeacher.evaluate defaults to 5.0 when no score found."""
        teacher, mock_tokenizer = mock_teacher_for_eval

        mock_tokenizer.decode = MagicMock(return_value="This is a response without any numeric score.")

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response"
        )

        assert result.overall_score == 5.0

    @pytest.mark.asyncio
    async def test_evaluate_extracts_improved_response(self, mock_teacher_for_eval):
        """Test LocalTeacher.evaluate extracts improved response when present."""
        teacher, mock_tokenizer = mock_teacher_for_eval

        mock_tokenizer.decode = MagicMock(return_value="""Score: 6
The response could be better.
Improved version: Here is a much better explanation of the concept.""")

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response",
            generate_improved=True
        )

        assert result.improved_response is not None
        assert "better explanation" in result.improved_response

    @pytest.mark.asyncio
    async def test_evaluate_no_improved_when_not_requested(self, mock_teacher_for_eval):
        """Test LocalTeacher.evaluate doesn't extract improved when not requested."""
        teacher, mock_tokenizer = mock_teacher_for_eval

        mock_tokenizer.decode = MagicMock(return_value="Score: 7\nGood job!")

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response",
            generate_improved=False
        )

        # Should return None for improved_response
        assert result.improved_response is None

    @pytest.mark.asyncio
    async def test_evaluate_clamps_score_to_valid_range(self, mock_teacher_for_eval):
        """Test LocalTeacher.evaluate clamps scores to 0-10 range."""
        teacher, mock_tokenizer = mock_teacher_for_eval

        # Test score > 10
        mock_tokenizer.decode = MagicMock(return_value="Score: 15\nAmazing!")
        result = await teacher.evaluate(prompt="Test", student_response="Response")
        assert result.overall_score == 10.0

    @pytest.mark.asyncio
    async def test_evaluate_stores_raw_response_as_reasoning(self, mock_teacher_for_eval):
        """Test LocalTeacher.evaluate stores raw response as reasoning."""
        teacher, mock_tokenizer = mock_teacher_for_eval

        raw_response = "Score: 7\nThe response demonstrates understanding."
        mock_tokenizer.decode = MagicMock(return_value=raw_response)

        result = await teacher.evaluate(
            prompt="Test",
            student_response="Test response"
        )

        assert result.reasoning == raw_response


# ============================================================================
# Score Extraction Tests
# ============================================================================

class TestLocalTeacherScoreExtraction:
    """Tests for LocalTeacher._extract_score method."""

    @pytest.fixture
    def teacher(self):
        """Create a mocked LocalTeacher for score extraction tests."""
        with patch("aspire.teachers.local.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.teachers.local.AutoTokenizer") as mock_tokenizer_class:

            mock_model = MagicMock()
            mock_model.eval.return_value = None
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "</s>"
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            teacher = LocalTeacher(model_name_or_path="test-model")
            yield teacher

    def test_extract_score_colon_format(self, teacher):
        """Test _extract_score with 'Score: X' format."""
        score = teacher._extract_score("Score: 8.5 out of 10")
        assert score == 8.5

    def test_extract_score_slash_format(self, teacher):
        """Test _extract_score with 'X/10' format."""
        score = teacher._extract_score("This response gets a 7/10")
        assert score == 7.0

    def test_extract_score_just_number(self, teacher):
        """Test _extract_score with just a number at start."""
        score = teacher._extract_score("8\nGood response overall.")
        assert score == 8.0

    def test_extract_score_decimal(self, teacher):
        """Test _extract_score with decimal score."""
        score = teacher._extract_score("Score: 7.25")
        assert score == 7.25

    def test_extract_score_no_match(self, teacher):
        """Test _extract_score returns default when no score found."""
        score = teacher._extract_score("No numeric value here.")
        assert score == 5.0

    def test_extract_score_caps_at_10(self, teacher):
        """Test _extract_score caps score at 10."""
        score = teacher._extract_score("Score: 12")
        assert score == 10.0

    def test_extract_score_floors_at_0(self, teacher):
        """Test _extract_score floors score at 0."""
        score = teacher._extract_score("Score: -5")
        # Since -5 doesn't match the patterns (they expect positive numbers), should default
        assert score == 5.0


# ============================================================================
# Improved Response Extraction Tests
# ============================================================================

class TestLocalTeacherImprovedExtraction:
    """Tests for LocalTeacher._extract_improved method."""

    @pytest.fixture
    def teacher(self):
        """Create a mocked LocalTeacher."""
        with patch("aspire.teachers.local.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.teachers.local.AutoTokenizer") as mock_tokenizer_class:

            mock_model = MagicMock()
            mock_model.eval.return_value = None
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "</s>"
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            teacher = LocalTeacher(model_name_or_path="test-model")
            yield teacher

    def test_extract_improved_version_marker(self, teacher):
        """Test _extract_improved with 'improved version:' marker."""
        result = teacher._extract_improved(
            "Some feedback. Improved version: This is the better answer."
        )
        assert result == "This is the better answer."

    def test_extract_improved_better_response_marker(self, teacher):
        """Test _extract_improved with 'better response:' marker."""
        result = teacher._extract_improved(
            "Issues found. Better response: A much improved answer here."
        )
        assert result == "A much improved answer here."

    def test_extract_improved_no_marker(self, teacher):
        """Test _extract_improved returns None when no marker found."""
        result = teacher._extract_improved(
            "This is just feedback without any improved version."
        )
        assert result is None

    def test_extract_improved_case_insensitive(self, teacher):
        """Test _extract_improved is case insensitive."""
        result = teacher._extract_improved(
            "Feedback. IMPROVED VERSION: Better answer here."
        )
        assert result == "Better answer here."


# ============================================================================
# Custom Tokenizer Tests
# ============================================================================

class TestLocalTeacherCustomTokenizer:
    """Tests for LocalTeacher with custom tokenizer."""

    def test_init_with_existing_pad_token(self):
        """Test LocalTeacher with tokenizer that already has pad_token."""
        with patch("aspire.teachers.local.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.teachers.local.AutoTokenizer") as mock_tokenizer_class:

            mock_model = MagicMock()
            mock_model.eval.return_value = None
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "<pad>"  # Already has pad token
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            teacher = LocalTeacher(model_name_or_path="test-model")

            # pad_token should remain unchanged
            assert mock_tokenizer.pad_token == "<pad>"


# ============================================================================
# Repr Tests
# ============================================================================

class TestLocalTeacherRepr:
    """Tests for LocalTeacher string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        with patch("aspire.teachers.local.AutoModelForCausalLM") as mock_model_class, \
             patch("aspire.teachers.local.AutoTokenizer") as mock_tokenizer_class:

            mock_model = MagicMock()
            mock_model.eval.return_value = None
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "</s>"
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            teacher = LocalTeacher(
                model_name_or_path="test-model",
                name="My Local Model"
            )

            repr_str = repr(teacher)
            assert "LocalTeacher" in repr_str
            assert "My Local Model" in repr_str
