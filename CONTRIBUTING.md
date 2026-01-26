# Contributing to ASPIRE

Thank you for your interest in contributing to ASPIRE! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)
- [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to build something meaningful together.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended, 16GB+ VRAM)
- Git

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/aspire-ai.git
   cd aspire-ai
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/mcp-tool-shop/aspire-ai.git
   ```

## Development Setup

### Install Dependencies

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check everything is working
aspire doctor

# Run tests
pytest tests/ -v
```

### Environment Variables

For full functionality, set up API keys:

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here
set OPENAI_API_KEY=your-key-here  # Optional

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
export OPENAI_API_KEY=your-key-here  # Optional
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-teacher-persona`
- `fix/critic-loss-calculation`
- `docs/improve-readme`
- `refactor/simplify-dialogue-generator`

### Commit Messages

Write clear, concise commit messages:

```
Add Socratic teacher persona with question-based challenges

- Implements never-answer-directly philosophy
- Adds 5 new challenge types for probing reasoning
- Includes unit tests for all new functionality
```

### Keep Changes Focused

- One feature or fix per pull request
- Keep PRs small and reviewable (under 500 lines when possible)
- Split large changes into multiple PRs

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_teachers.py -v

# Run with coverage
pytest tests/ --cov=aspire --cov-report=html

# Skip slow tests
pytest tests/ -v -m "not slow"

# Skip tests requiring GPU
pytest tests/ -v -m "not gpu"
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures from `tests/conftest.py`

Example test:

```python
import pytest
from aspire.teachers import get_teacher

def test_get_teacher_socratic():
    """Test that Socratic teacher can be retrieved."""
    teacher = get_teacher("socratic")
    assert teacher.name == "Socratic Teacher"
    assert "question" in teacher.description.lower()

@pytest.mark.asyncio
async def test_teacher_challenge():
    """Test that teacher generates valid challenges."""
    teacher = get_teacher("socratic")
    challenge = await teacher.challenge(
        prompt="Explain recursion",
        student_response="Recursion is when a function calls itself."
    )
    assert challenge.content
    assert challenge.challenge_type
```

### Test Markers

Use markers for special test categories:

```python
@pytest.mark.slow  # Long-running tests
@pytest.mark.gpu   # Requires GPU
@pytest.mark.api   # Calls external APIs
@pytest.mark.integration  # Integration tests
```

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. **Run the full test suite**:
   ```bash
   pytest tests/ -v
   ```

3. **Run linting and type checking**:
   ```bash
   ruff check aspire/
   pyright aspire/
   ```

4. **Format your code**:
   ```bash
   ruff format aspire/
   ```

### Submitting

1. Push your branch to your fork
2. Open a pull request against `master`
3. Fill out the PR template with:
   - Description of changes
   - Related issues
   - Testing performed

### Review Process

- PRs require at least one approval
- CI must pass (tests, linting, type checking)
- Address review feedback promptly

## Style Guide

### Python Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
ruff check aspire/

# Auto-fix issues
ruff check --fix aspire/

# Format code
ruff format aspire/
```

### Key Conventions

- **Line length**: 100 characters
- **Imports**: Sorted with `isort` rules (handled by Ruff)
- **Docstrings**: Google style
- **Type hints**: Required for public APIs

### Docstring Example

```python
def train(
    self,
    train_prompts: list[str],
    eval_prompts: list[str] | None = None,
) -> dict[str, Any]:
    """
    Train the student and critic models.

    Args:
        train_prompts: List of prompts for training.
        eval_prompts: Optional prompts for evaluation during training.

    Returns:
        Dictionary containing training metrics including loss curves
        and evaluation scores.

    Raises:
        ValueError: If train_prompts is empty.
    """
```

### Windows Compatibility

ASPIRE is fully Windows-compatible. Please ensure:

- Use `dataloader_num_workers=0` in DataLoader
- Wrap scripts with `if __name__ == "__main__":` and `freeze_support()`
- Test on Windows if possible, or note if you couldn't

## Areas for Contribution

We especially welcome contributions in these areas:

### 1. New Teacher Personas

Create teachers with different philosophies:

```python
# aspire/teachers/your_persona.py
from aspire.teachers.claude import ClaudeTeacher
from aspire.teachers.base import ChallengeType, EvaluationDimension

class YourTeacher(ClaudeTeacher):
    def __init__(self, **kwargs):
        super().__init__(
            name="Your Teacher Name",
            description="Your teaching philosophy...",
            preferred_challenges=[
                ChallengeType.PROBE_REASONING,
                # ...
            ],
            evaluation_dimensions=[
                EvaluationDimension.REASONING,
                # ...
            ],
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return """Your system prompt defining the persona..."""
```

Ideas:
- **Historian** - Demands context, precedent, and lessons from the past
- **Engineer** - Focuses on implementation, edge cases, and practicality
- **Philosopher** - Explores meaning, ethics, and first principles
- **Debugger** - Finds flaws, inconsistencies, and failure modes

### 2. Curriculum Datasets

We need curated prompts for each curriculum stage:

- **Foundation**: Simple factual Q&A
- **Reasoning**: Multi-step problems
- **Nuance**: Ambiguous scenarios with tradeoffs
- **Adversarial**: Challenging edge cases
- **Transfer**: Cross-domain generalization tests

Format: JSON list of prompts with metadata

```json
[
  {
    "prompt": "Explain why water expands when it freezes",
    "stage": "reasoning",
    "domain": "physics",
    "expected_challenges": ["probe_reasoning", "edge_case"]
  }
]
```

### 3. Evaluation Benchmarks

Help us measure whether ASPIRE actually produces better judgment:

- Critic accuracy vs teacher
- Student improvement trajectories
- Generalization to unseen domains
- Comparison with standard SFT

### 4. Interpretability Tools

Understanding what the critic learned:

- Visualizations of critic attention
- Analysis of what features predict teacher approval
- Comparison of critic judgments across domains

### 5. Documentation

- Tutorials for specific use cases
- Video walkthroughs
- Integration guides for other frameworks

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

---

*"Teaching AI to develop judgment, not just knowledge."*
