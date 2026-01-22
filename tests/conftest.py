"""
Pytest configuration and shared fixtures for ASPIRE tests.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Environment Setup
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure environment for testing."""
    # Disable xformers (not available on all systems)
    os.environ["XFORMERS_DISABLED"] = "1"

    # Use CPU for tests by default (faster, more portable)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    yield


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get appropriate device for tests."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def cpu_device():
    """Force CPU device."""
    return torch.device("cpu")


# ============================================================================
# Mock Fixtures - API Clients
# ============================================================================

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic API client."""
    with patch("anthropic.AsyncAnthropic") as mock_class:
        mock_client = AsyncMock()
        mock_class.return_value = mock_client

        # Mock message creation
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is a helpful response.")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        yield mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI API client."""
    with patch("openai.AsyncOpenAI") as mock_class:
        mock_client = AsyncMock()
        mock_class.return_value = mock_client

        # Mock chat completion
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="This is a helpful response."))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        yield mock_client


# ============================================================================
# Mock Fixtures - Models
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
        if isinstance(text, str):
            text = [text]

        batch_size = len(text)
        max_length = kwargs.get("max_length", 512)

        return {
            "input_ids": torch.randint(0, 1000, (batch_size, max_length)),
            "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
        }

    tokenizer.side_effect = mock_call
    tokenizer.__call__ = mock_call
    tokenizer.decode = MagicMock(return_value="decoded text")

    return tokenizer


@pytest.fixture
def mock_model():
    """Mock HuggingFace model."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.hidden_size = 768
    model.config.vocab_size = 32000

    # Mock forward pass
    def mock_forward(**kwargs):
        batch_size = kwargs.get("input_ids", torch.zeros(1, 10)).shape[0]
        seq_len = kwargs.get("input_ids", torch.zeros(1, 10)).shape[1]

        output = MagicMock()
        output.logits = torch.randn(batch_size, seq_len, 32000)
        output.loss = torch.tensor(1.5)
        output.hidden_states = (torch.randn(batch_size, seq_len, 768),)
        return output

    model.side_effect = mock_forward
    model.__call__ = mock_forward
    model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(10, 10))]))
    model.to = MagicMock(return_value=model)
    model.train = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)

    return model


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_dialogue():
    """Sample dialogue for testing."""
    return {
        "context": "You are a helpful AI assistant.",
        "prompt": "What is the capital of France?",
        "response": "The capital of France is Paris.",
    }


@pytest.fixture
def sample_dialogues():
    """Multiple sample dialogues."""
    return [
        {
            "context": "You are a helpful AI assistant.",
            "prompt": "What is 2+2?",
            "response": "2+2 equals 4.",
        },
        {
            "context": "You are a coding assistant.",
            "prompt": "How do I print hello world in Python?",
            "response": "Use: print('Hello, World!')",
        },
        {
            "context": "You are a friendly assistant.",
            "prompt": "Tell me a joke.",
            "response": "Why did the scarecrow win an award? He was outstanding in his field!",
        },
    ]


@pytest.fixture
def sample_code_good():
    """Sample good Python code."""
    return '''
def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer.

    Args:
        n: Non-negative integer

    Returns:
        Factorial of n

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''


@pytest.fixture
def sample_code_bad():
    """Sample bad Python code with issues."""
    return '''
def f(x):
    y = eval(x)
    data = []
    for i in range(len(y)):
        if y[i] != None:
            data.append(y[i])
    return data
'''


@pytest.fixture
def sample_code_security_issues():
    """Sample code with security vulnerabilities."""
    return '''
import os
import pickle

def process_user_input(user_data):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_data}"

    # Command injection
    os.system(f"echo {user_data}")

    # Insecure deserialization
    obj = pickle.loads(user_data)

    # Hardcoded password
    password = "admin123"

    return obj
'''


# ============================================================================
# Tensor Fixtures
# ============================================================================

@pytest.fixture
def random_embeddings():
    """Random embedding tensors for testing."""
    batch_size = 4
    seq_len = 128
    hidden_dim = 768

    return torch.randn(batch_size, seq_len, hidden_dim)


@pytest.fixture
def random_scores():
    """Random score tensors."""
    batch_size = 4
    return torch.rand(batch_size) * 10  # Scores 0-10


@pytest.fixture
def random_logits():
    """Random logit tensors."""
    batch_size = 4
    seq_len = 128
    vocab_size = 32000

    return torch.randn(batch_size, seq_len, vocab_size)


# ============================================================================
# Utility Functions
# ============================================================================

def assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5):
    """Assert two tensors are approximately equal."""
    assert torch.allclose(a, b, rtol=rtol), f"Tensors not equal:\n{a}\n!=\n{b}"


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


# ============================================================================
# Markers
# ============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "api: marks tests that call external APIs")


# ============================================================================
# Teacher Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_claude_response_challenge():
    """Mock response for Claude challenge generation."""
    return {
        "challenge": "Why did you choose that particular approach?",
        "context": "Testing the student's reasoning",
        "difficulty": 0.6
    }


@pytest.fixture
def mock_claude_response_evaluate():
    """Mock response for Claude evaluation."""
    return {
        "overall_score": 7.5,
        "dimension_scores": [
            {"dimension": "correctness", "score": 8.0, "explanation": "Factually correct"},
            {"dimension": "reasoning", "score": 7.0, "explanation": "Good reasoning"},
        ],
        "reasoning": "The response demonstrates good understanding.",
        "strengths": ["Clear explanation", "Good examples"],
        "weaknesses": ["Could be more detailed"],
        "suggestions": ["Add more examples"],
        "improved_response": "An improved version of the response."
    }


@pytest.fixture
def mock_openai_response_challenge():
    """Mock response for OpenAI challenge generation."""
    return {
        "challenge": "What happens in edge cases?",
        "context": "Testing edge case handling",
        "difficulty": 0.5
    }


@pytest.fixture
def mock_openai_response_evaluate():
    """Mock response for OpenAI evaluation."""
    return {
        "overall_score": 8.0,
        "dimension_scores": [
            {"dimension": "correctness", "score": 8.5, "explanation": "Very accurate"},
            {"dimension": "clarity", "score": 7.5, "explanation": "Well explained"},
        ],
        "reasoning": "Strong response overall.",
        "strengths": ["Accurate", "Well-structured"],
        "weaknesses": ["Minor gaps"],
        "suggestions": ["Consider edge cases"],
        "improved_response": "A better version would include..."
    }


@pytest.fixture
def sample_prompt():
    """Sample prompt for testing."""
    return "Explain the concept of recursion in programming."


@pytest.fixture
def sample_student_response():
    """Sample student response for testing."""
    return "Recursion is when a function calls itself to solve a problem."


@pytest.fixture
def sample_dialogue_history():
    """Sample dialogue history for testing."""
    from aspire.teachers.base import (
        DialogueHistory, DialogueTurn, TeacherChallenge, ChallengeType
    )

    history = DialogueHistory(
        prompt="Explain recursion",
        initial_response="Recursion is a function calling itself."
    )

    turn = DialogueTurn(
        turn_number=1,
        challenge=TeacherChallenge(
            challenge_type=ChallengeType.PROBE_REASONING,
            content="Why would you use recursion instead of iteration?",
            difficulty=0.5
        ),
        student_response="Recursion is useful for problems with recursive structure."
    )
    history.add_turn(turn)

    return history
