# ASPIRE for Code

**Teaching code models to develop programming judgment.**

## The Idea

Traditional code assistants match patterns. They suggest what *looks* right based on training data.

**ASPIRE for code**: A committee of experts (correctness checker, style guide, security auditor) critiques generated code. The model internalizes this judgment and self-refines before outputting.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Code Model    │ ──> │   Generated     │ ──> │  Code Teachers  │
│    (Student)    │     │     Code        │     │  (Expert Panel) │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
        ▲                                                 │
        │              ┌─────────────────┐               │ Critique
        └───────────── │   Code Critic   │ <─────────────┘
           Learn from  │  (Internalized  │   Learn to
                       │   Judgment)     │   Predict
                       └─────────────────┘

After training: Critic catches bugs and style issues BEFORE output
```

## Quick Start

### Installation

```bash
pip install aspire-ai

# Optional: static analysis tools
pip install ruff mypy bandit

# Optional: for student model training
pip install transformers peft
```

### Basic Code Critique

```python
from aspire.integrations.code import CodeTeacher, CodeSample
from aspire.integrations.code.config import Language

# Create expert panel
teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

# Critique some code
sample = CodeSample(
    code='''
def process(data):
    result = eval(data)  # Security issue!
    return result
''',
    language=Language.PYTHON,
)

critique = teacher.critique(sample)

print(f"Score: {critique.overall_score}/10")
print(f"Issues: {critique.weaknesses}")
print(f"Suggestions: {critique.suggestions}")
```

### Training a Code Critic

```python
from aspire.integrations.code import AspireCodeTrainer, CodeTeacher

# Create teacher
teacher = CodeTeacher(personas=["correctness_checker", "style_guide"])

# Train
trainer = AspireCodeTrainer(teacher=teacher)
trainer.train(train_data="training_pairs.json")

# Use trained critic
score = trainer.score_code("def foo(): pass")
```

## Code Teachers

Different teachers evaluate code from different perspectives:

| Teacher | Focus | Key Questions |
|---------|-------|---------------|
| **Correctness Checker** | Bugs, types, logic | "Does this code work?" |
| **Style Guide** | Conventions, readability | "Is this idiomatic Python?" |
| **Security Auditor** | Vulnerabilities, injection | "Can this be exploited?" |
| **Architecture Reviewer** | Structure, SOLID | "Is this well-organized?" |
| **Performance Analyst** | Efficiency, complexity | "Is this O(n) or O(n²)?" |
| **Documentation Critic** | Docstrings, comments | "Is this well-explained?" |

### Composite Teachers

Combine multiple perspectives:

```python
teacher = CodeTeacher(
    personas=[
        "correctness_checker",
        "style_guide",
        "security_auditor",
    ],
    strategy="vote",  # or "rotate", "debate"
)
```

### Custom Teachers

Create domain-specific teachers:

```python
from aspire.integrations.code.code_teacher import BaseCodeTeacher

class APIStyleGuide(BaseCodeTeacher):
    """Enforces REST API conventions."""

    def critique(self, sample):
        # Check for proper HTTP status codes
        # Validate endpoint naming
        # Check response formats
        ...
```

## Static Analysis Integration

ASPIRE integrates with popular static analysis tools:

| Tool | Purpose | Auto-detected |
|------|---------|---------------|
| **Ruff** | Fast linting | Yes |
| **Mypy** | Type checking | Yes |
| **Bandit** | Security scanning | Yes |
| **Semgrep** | Pattern matching | Optional |

```python
from aspire.integrations.code.analysis import CodeAnalyzer

analyzer = CodeAnalyzer(
    use_ruff=True,
    use_mypy=True,
    use_bandit=True,
)

result = analyzer.analyze(code, language=Language.PYTHON)
print(f"Issues found: {len(result.issues)}")
print(f"Security score: {result.security_score}/10")
```

## Training Data

### From GitHub

```python
from aspire.integrations.code.data import GitHubRepoCollector

collector = GitHubRepoCollector()

# Collect from curated quality repos
for repo, filename, code in collector.collect_from_quality_repos(
    Language.PYTHON,
    files_per_repo=100,
):
    # Process code...
```

### Generate Pairs

```python
from aspire.integrations.code.data import generate_training_pairs

pairs = generate_training_pairs(
    teacher=teacher,
    code_samples=samples,
    language=Language.PYTHON,
)
```

## Examples

### Basic Critique
```bash
python -m aspire.integrations.code.examples.basic_critique
```

### Train Critic
```bash
python -m aspire.integrations.code.examples.train_critic
```

## The Philosophy

> *"We don't carry our code reviewers around forever. We internalize their standards."*

A junior developer learns from code reviews. Over time, they start catching issues themselves—before the PR review. They've internalized their team's standards.

ASPIRE gives code models that same growth. The critic becomes an inner voice that catches bugs, security issues, and style problems before the code is even output.

## Supported Languages

| Language | Full Support | Static Analysis |
|----------|--------------|-----------------|
| Python | ✅ | Ruff, Mypy, Bandit |
| JavaScript | ✅ | ESLint (coming) |
| TypeScript | ✅ | TSC (coming) |
| Rust | Partial | Clippy (coming) |
| Go | Partial | golint (coming) |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Optional: transformers, peft (for student training)
- Optional: ruff, mypy, bandit (for static analysis)

### Windows Compatibility

Fully Windows-compatible:
- `num_workers=0` in dataloaders
- Proper `freeze_support()` in examples

---

*Part of the [ASPIRE project](https://github.com/mcp-tool-shop/aspire-ai)*
