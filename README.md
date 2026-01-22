# ASPIRE

**Adversarial Student-Professor Internalized Reasoning Engine**

A fine-tuning paradigm that mirrors human learning: instead of memorizing examples, the student model develops an *internalized critic* - a learned sense of judgment that predicts what a teacher model would think.

## The Idea

Traditional fine-tuning: "Here are the right answers. Match them."

ASPIRE: "Here is a wise mind. Learn to think like it does."

Humans don't carry their mentors around forever. We internalize them. We develop an inner voice that predicts their judgment, and eventually that prediction becomes our own taste, standards, and discernment.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Generate an adversarial dialogue
aspire dialogue "Explain recursion" --teacher socratic

# List available teacher personas
aspire teachers

# Train a model
aspire train --prompts data/prompts.json --teacher adversarial --epochs 3
```

## Teacher Personas

- **Socratic** - Teaches through questions, never gives answers directly
- **Scientific** - Demands evidence, rigor, and falsifiable claims
- **Creative** - Encourages novel thinking and unconventional approaches
- **Adversarial** - Stress-tests reasoning through devil's advocacy
- **Compassionate** - Balances challenge with encouragement

## Architecture

```
Student Model ──generates──> Response
      │                           │
      │                           ▼
      │                    Teacher Model
      │                    (evaluates, challenges)
      │                           │
      │                           ▼
      │                    Critic Model
      │                    (learns to predict teacher)
      │                           │
      ▼                           ▼
Backprop from Critic's learned judgment
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (16GB+ VRAM recommended)
- Anthropic API key (for Claude teacher) or OpenAI API key

## License

MIT
