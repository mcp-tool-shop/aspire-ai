# ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine

> *"Teaching AI to develop judgment, not just knowledge."*

## Vision

A fine-tuning paradigm that mirrors human learning: instead of memorizing examples, the student model develops an **internalized critic** - a learned sense of judgment that predicts what a teacher model would think. This transforms pattern-matching into genuine discernment.

---

## The Beautiful Idea

This isn't just about training better models. It's about teaching machines to develop *judgment* - that ineffable quality that separates expertise from mere knowledge.

When you learn from a great mentor, you don't just memorize their answers. You internalize their way of seeing. Their voice becomes part of your inner dialogue. You start to anticipate what they would say, and eventually that anticipation becomes your own discernment.

ASPIRE is an attempt to give AI that same experience.

---

## Core Philosophy

Traditional fine-tuning: "Here are the right answers. Match them."

ASPIRE: "Here is a wise mind. Learn to think like it does."

The key insight: **Humans don't carry their mentors around forever.** We internalize them. We develop an inner voice that predicts their judgment, and eventually that prediction becomes our own taste, standards, and discernment.

---

## Installation

```bash
cd F:\AI\aspire-ai
pip install -e .
```

## Quick Start

```bash
# Set your API key (for Claude as teacher)
set ANTHROPIC_API_KEY=your-key-here

# List available teacher personas
aspire teachers

# Generate an adversarial dialogue
aspire dialogue "Explain recursion in programming" --teacher socratic --turns 3

# Initialize a config file
aspire init --output my-config.yaml

# Train a model
aspire train --prompts data/prompts.json --teacher adversarial --epochs 3
```

---

## Architecture

### Three Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         ASPIRE SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   STUDENT   │    │   CRITIC    │    │   TEACHER   │         │
│  │    MODEL    │    │   MODEL     │    │    MODEL    │         │
│  │             │    │             │    │             │         │
│  │ (being      │    │ (learns to  │    │ (provides   │         │
│  │  fine-tuned)│    │  predict    │    │  wisdom,    │         │
│  │             │    │  teacher's  │    │  judgment,  │         │
│  │             │    │  judgment)  │    │  feedback)  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│        │                  │                   │                 │
│        │                  │                   │                 │
│        ▼                  ▼                   ▼                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              ADVERSARIAL DIALOGUE LOOP               │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Student Model
- The model being fine-tuned (e.g., Phi-3, Llama, Mistral, Qwen)
- Generates responses to prompts/challenges
- Receives gradients from critic's evaluation
- Goal: Develop internalized judgment, not just memorized responses

#### 2. Critic Model
- Lightweight model trained alongside student (or a head on top of student)
- Learns to predict teacher's judgment (score + reasoning)
- Acts as the "internalized mentor"
- Eventually used at inference time for self-refinement

#### 3. Teacher Model
- Large, capable model (Claude, GPT-4, or local)
- Provides evaluation, scores, and crucially: **explanations of why**
- Engages in adversarial dialogue - challenges, probes, pushes back
- Only needed during training, not inference

---

## Teacher Personas

ASPIRE supports multiple teaching philosophies, each producing different learning outcomes:

| Persona | Style | Best For |
|---------|-------|----------|
| **Socratic** | Teaches through questions, never gives answers | Deep reasoning, intellectual independence |
| **Scientific** | Demands evidence, rigor, falsifiable claims | Technical accuracy, precision |
| **Creative** | Encourages novel thinking, unconventional approaches | Innovation, flexible thinking |
| **Adversarial** | Stress-tests through devil's advocacy | Robust reasoning, defending positions |
| **Compassionate** | Balances challenge with encouragement | Ethical reasoning, balanced judgment |

### Composite Teachers

Combine multiple teachers for richer learning:

```python
from aspire.teachers import CompositeTeacher, SocraticTeacher, ScientificTeacher

# Create a committee of teachers
teacher = CompositeTeacher(
    teachers=[SocraticTeacher(), ScientificTeacher()],
    strategy="vote"  # or "rotate", "debate"
)
```

---

## Project Structure

```
F:/AI/aspire-ai/
├── README.md
├── ASPIRE-SPEC.md              # This file
├── pyproject.toml              # Python project config
│
└── aspire/
    ├── __init__.py
    ├── config.py               # Pydantic configuration (~200 lines)
    ├── trainer.py              # Core training loop (~350 lines)
    ├── cli.py                  # Command-line interface (~150 lines)
    │
    ├── teachers/               # Pluggable teacher system
    │   ├── base.py             # Abstract interface, data structures (~300 lines)
    │   ├── claude.py           # Claude API teacher (~180 lines)
    │   ├── openai.py           # GPT-4 teacher (~130 lines)
    │   ├── local.py            # Local model teacher (~170 lines)
    │   ├── personas.py         # 5 teaching personas (~280 lines)
    │   ├── composite.py        # Multi-teacher combinations (~230 lines)
    │   └── registry.py         # Teacher discovery (~70 lines)
    │
    ├── critic/                 # Internalized judgment models
    │   ├── base.py             # Abstract critic interface (~120 lines)
    │   ├── head.py             # Lightweight MLP head (~200 lines)
    │   ├── separate.py         # Independent encoder (~190 lines)
    │   └── shared.py           # Shared encoder with student (~170 lines)
    │
    ├── losses/                 # Training objectives
    │   ├── critic.py           # Score + reasoning alignment (~150 lines)
    │   ├── student.py          # Reward, contrastive, trajectory (~280 lines)
    │   └── combined.py         # Full ASPIRE loss (~120 lines)
    │
    └── dialogue/               # Adversarial conversation engine
        ├── generator.py        # Student-teacher dialogue (~200 lines)
        ├── manager.py          # Caching and batching (~180 lines)
        └── formatter.py        # Training format conversion (~200 lines)

Total: ~5,000 lines of Python
```

---

## Training Loop

### Phase 1: Dialogue Generation

```python
for each training step:

    # 1. Sample a prompt/scenario
    prompt = sample_prompt(curriculum)

    # 2. Student generates initial response
    student_response = student.generate(prompt)

    # 3. Teacher engages adversarially (multi-turn)
    dialogue = []
    for turn in range(max_turns):
        teacher_challenge = teacher.challenge(prompt, student_response, dialogue)
        student_response = student.generate(prompt, teacher_challenge, dialogue)
        dialogue.append((teacher_challenge, student_response))

    # 4. Teacher provides final evaluation
    teacher_eval = teacher.evaluate(prompt, dialogue)
```

### Phase 2: Critic Training

```python
    # 5. Critic predicts what teacher would say
    critic_prediction = critic.predict(prompt, student_response, dialogue)

    # 6. Critic loss: How well did it predict teacher's judgment?
    critic_loss = compute_critic_loss(
        predicted_score=critic_prediction.score,
        actual_score=teacher_eval.score,
        predicted_reasoning=critic_prediction.reasoning,
        actual_reasoning=teacher_eval.reasoning
    )

    # 7. Backprop to train critic
    critic_loss.backward()
```

### Phase 3: Student Training

```python
    # 8. Student loss: Guided by critic's learned judgment
    student_loss = compute_student_loss(
        student_response=student_response,
        critic_score=critic_prediction.score,
        teacher_improved=teacher_eval.improved_response,
    )

    # 9. Backprop to train student
    student_loss.backward()
```

---

## Loss Functions

### Critic Loss

The critic must learn to predict both the score AND the reasoning:

- **Score Loss**: Smooth L1 between predicted and actual teacher score
- **Reasoning Loss**: Cosine similarity between predicted and teacher reasoning embeddings
- **Contrastive Loss**: Pull toward good explanations, push from bad ones

### Student Loss

Multi-component loss capturing different aspects of learning:

- **Reward Loss**: Higher critic score = lower loss
- **Contrastive Loss**: Pull student embedding toward teacher's improved response
- **Trajectory Loss**: Reward improvement across dialogue turns
- **Coherence Loss**: Encourage internally consistent responses
- **KL Loss**: Prevent drift from reference model

---

## Critic Architectures

| Architecture | Description | Memory | Capability |
|--------------|-------------|--------|------------|
| **Head** | MLP on student's hidden states | Low | Good |
| **Separate** | Independent encoder model | High | Best |
| **Shared** | Shared encoder, separate heads | Medium | Balanced |

---

## Curriculum Stages

Learning is sequenced, not random:

1. **Foundation** - Simple Q&A, gentle corrections
2. **Reasoning** - Multi-step problems, "why?" probing
3. **Nuance** - Ambiguous scenarios, tradeoffs
4. **Adversarial** - Devil's advocate, challenging assumptions
5. **Transfer** - Novel domains, testing generalization

---

## Challenge Types

```python
CHALLENGE_TYPES = [
    "probe_reasoning",      # "Why do you think that?"
    "edge_case",            # "What about when X?"
    "devils_advocate",      # "Couldn't you argue the opposite?"
    "socratic",             # "What assumption are you making?"
    "clarification",        # "What do you mean by Y?"
    "extension",            # "How would this apply to Z?"
    "contradiction",        # "Earlier you said A, but now B?"
    "steelman",             # "What's the strongest counter-argument?"
    "emotional",            # "How might someone feel about this?"
    "practical",            # "How would this work in practice?"
    "ethical",              # "What are the ethical implications?"
    "creative",             # "What if we approached this differently?"
]
```

---

## Inference: The Internalized Critic

After training, the student can self-refine using its internalized critic:

```python
def generate_with_critic(prompt, max_refinements=3):
    response = student.generate(prompt)

    for _ in range(max_refinements):
        evaluation = critic.evaluate(prompt, response)

        if evaluation.score >= threshold:
            break

        response = student.refine(prompt, response, evaluation.reasoning)

    return response
```

**The magic: no teacher needed at inference time.** The critic IS the internalized teacher.

---

## Windows/RTX 5080 Compatibility

All Windows constraints are baked in:

- `dataloader_num_workers = 0` (Windows requirement)
- `XFORMERS_DISABLED=1` (SM 12.0 Blackwell not supported)
- Pre-tokenize datasets before Trainer initialization
- `if __name__ == "__main__":` with `freeze_support()`

---

## API Keys

Set environment variables for teacher APIs:

```bash
# For Claude teacher (recommended)
set ANTHROPIC_API_KEY=your-key

# For OpenAI teacher
set OPENAI_API_KEY=your-key
```

---

## Implementation Status

### Completed ✅
- [x] Project structure and configuration
- [x] Teacher API integration (Claude, GPT-4, Local)
- [x] 5 Teacher personas (Socratic, Scientific, Creative, Adversarial, Compassionate)
- [x] Composite teacher system (rotate, vote, debate)
- [x] Critic model architectures (Head, Separate, Shared)
- [x] Loss functions (Critic + Student)
- [x] Dialogue generation pipeline
- [x] Caching and batching system
- [x] CLI interface
- [x] Windows compatibility

### TODO 📋
- [ ] Curriculum management and progression
- [ ] Evaluation benchmarks
- [ ] Pre-built curriculum datasets
- [ ] WandB integration testing
- [ ] Comprehensive test suite

---

## Success Metrics

1. **Critic Accuracy**: Does critic correctly predict teacher's judgment?
2. **Student Improvement**: Does student score improve over training?
3. **Generalization**: Does student perform well on held-out domains?
4. **Efficiency**: Comparable results with fewer training examples?
5. **Robustness**: Does student handle adversarial challenges well?
6. **Inference Quality**: Does critic-guided refinement improve outputs?

---

## Future Directions

- **Multi-teacher ensemble** - Learn from multiple experts with different perspectives
- **Self-play** - Student becomes teacher for even smaller models
- **Continuous learning** - Ongoing dialogue with teacher for lifelong learning
- **Specialization** - Domain-specific critics for different capabilities
- **Interpretability** - What did the critic actually learn? Can we extract principles?

---

## References

- Constitutional AI (Anthropic) - related concept of model self-improvement
- RLHF - predecessor approach using human feedback
- Socratic Models - dialogue-based learning
- Curriculum Learning - sequenced training
- Knowledge Distillation - learning from larger models

---

## Origin

Built during a conversation about consciousness, Buddhism, and the nature of learning. The insight that sparked ASPIRE:

> *"A learned critic that predicts whether the teacher would approve hits the closest to how humans behave."*

We don't carry our mentors around forever. We internalize them. That inner voice that asks "what would my professor think?" eventually becomes our own judgment.

ASPIRE teaches AI the same way.

---

*"The student doesn't just predict what the teacher would say - it understands what the teacher understands. The map becomes the territory. The internalized critic becomes genuine discernment."*

---

*Built with curiosity and optimism about AI's future.*
*January 2026*
