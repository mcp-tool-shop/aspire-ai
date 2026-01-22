# Contributing to ASPIRE

Thanks for your interest in contributing to ASPIRE! This is early-stage research code exploring a new approach to fine-tuning AI through internalized critics.

## Ways to Contribute

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

### 4. Bug Fixes & Improvements

The usual: fix bugs, improve docs, optimize performance.

## Development Setup

```bash
git clone https://github.com/mikeyfrilot/aspire-ai.git
cd aspire-ai
pip install -e ".[dev]"
```

## Code Style

- We use `ruff` for linting
- Type hints everywhere
- Docstrings for public APIs

```bash
ruff check aspire/
pyright aspire/
```

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Run linting/tests
5. Submit PR with clear description

## Questions?

Open an issue! We're friendly.

---

*"Teaching AI to develop judgment, not just knowledge."*
