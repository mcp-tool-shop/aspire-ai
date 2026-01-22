<p align="center">
  <img src="https://img.shields.io/badge/ASPIRE-Teaching_AI_Judgment-blueviolet?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0wIDE4Yy00LjQxIDAtOC0zLjU5LTgtOHMzLjU5LTggOC04IDggMy41OSA4IDgtMy41OSA4LTggOHptLTEtMTNoMnY2aC0yem0wIDhoMnYyaC0yeiIvPjwvc3ZnPg==" alt="ASPIRE">
</p>

<h1 align="center">ASPIRE</h1>

<p align="center">
  <strong>Adversarial Student-Professor Internalized Reasoning Engine</strong>
</p>

<p align="center">
  <em>Teaching AI to develop judgment, not just knowledge.</em>
</p>

<p align="center">
  <a href="#the-idea">The Idea</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#teacher-personas">Teachers</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#integrations">Integrations</a> •
  <a href="#documentation">Docs</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/github/stars/mikeyfrilot/aspire-ai?style=social" alt="GitHub Stars">
</p>

---

## The Idea

**Traditional fine-tuning:** *"Here are the right answers. Match them."*

**ASPIRE:** *"Here is a wise mind. Learn to think like it does."*

When you learn from a great mentor, you don't just memorize their answers. You internalize their way of seeing. Their voice becomes part of your inner dialogue. You start to anticipate what they would say, and eventually that anticipation becomes your own discernment.

ASPIRE gives AI that same experience.

```
┌─────────────────────────────────────────────────────────────────┐
│                         ASPIRE SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   STUDENT   │    │   CRITIC    │    │   TEACHER   │         │
│  │    MODEL    │    │   MODEL     │    │    MODEL    │         │
│  │             │    │             │    │             │         │
│  │ (learning)  │    │ (internal-  │    │ (wisdom)    │         │
│  │             │    │  ized       │    │             │         │
│  │             │    │  judgment)  │    │             │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                   │                 │
│         └──────────────────┴───────────────────┘                 │
│                            │                                     │
│                   ADVERSARIAL DIALOGUE                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The **critic** learns to predict what the teacher would think. After training, the student uses this internalized critic to self-refine — **no teacher needed at inference time**.

---

## Quick Start

### Installation

```bash
git clone https://github.com/mikeyfrilot/aspire-ai.git
cd aspire-ai
pip install -e .
```

### Set Your API Key

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### Try It Out

```bash
# See available teacher personas
aspire teachers

# Generate an adversarial dialogue
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3

# Initialize a training config
aspire init --output my-config.yaml
```

---

## Teacher Personas

Different teachers produce different minds. Choose wisely.

| Persona | Philosophy | Produces |
|---------|------------|----------|
| 🏛️ **Socratic** | *"What assumption are you making?"* | Deep reasoning, intellectual independence |
| 🔬 **Scientific** | *"What's your evidence?"* | Technical precision, rigorous thinking |
| 🎨 **Creative** | *"What if we tried the opposite?"* | Innovation, lateral thinking |
| ⚔️ **Adversarial** | *"I disagree. Defend your position."* | Robust arguments, conviction |
| 💚 **Compassionate** | *"How might someone feel about this?"* | Ethical reasoning, wisdom |

### Composite Teachers

Combine multiple teachers for richer learning:

```python
from aspire.teachers import CompositeTeacher, SocraticTeacher, ScientificTeacher

# A committee of mentors
teacher = CompositeTeacher(
    teachers=[SocraticTeacher(), ScientificTeacher()],
    strategy="vote"  # or "rotate", "debate"
)
```

---

## How It Works

### 1. Adversarial Dialogue

The student generates a response. The teacher challenges it. Back and forth, probing weaknesses, demanding clarity, pushing deeper.

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. Critic Training

The critic learns to predict the teacher's judgment — not just the score, but the *reasoning*.

```python
critic_loss = predict_teacher_judgment(
    score=True,      # "This deserves a 7/10"
    reasoning=True,  # "Because the explanation lacks depth on X"
)
```

### 3. Student Training

The student learns from the critic's internalized judgment, pulling toward what the teacher would approve.

```python
student_loss = (
    reward_from_critic +      # Higher score = better
    contrastive_to_teacher +  # Pull toward teacher's improved version
    trajectory_improvement    # Get better across dialogue turns
)
```

### 4. Inference Magic

After training, the student self-refines using the internalized critic. **No teacher API calls needed.**

```python
def generate_with_judgment(prompt):
    response = student.generate(prompt)

    while critic.score(response) < threshold:
        response = student.refine(response, critic.feedback)

    return response  # Self-improved through internalized judgment
```

---

## CLI Reference

```bash
# List available teachers
aspire teachers

# Generate adversarial dialogue
aspire dialogue "Your prompt here" \
    --teacher socratic \
    --turns 3 \
    --model microsoft/Phi-3-mini-4k-instruct

# Initialize config file
aspire init --output config.yaml

# Train a model
aspire train \
    --config config.yaml \
    --prompts data/prompts.json \
    --teacher adversarial \
    --epochs 3

# Evaluate checkpoint
aspire evaluate checkpoints/epoch-3 \
    --prompts data/eval.json
```

---

## Project Structure

```
aspire/
├── teachers/          # Pluggable teacher personas
│   ├── claude.py      # Claude API teacher
│   ├── openai.py      # GPT-4 teacher
│   ├── local.py       # Local model teacher
│   ├── personas.py    # Socratic, Scientific, Creative, etc.
│   └── composite.py   # Multi-teacher combinations
│
├── critic/            # Internalized judgment models
│   ├── head.py        # Lightweight MLP on student hidden states
│   ├── separate.py    # Independent encoder
│   └── shared.py      # Shared encoder with student
│
├── losses/            # Training objectives
│   ├── critic.py      # Score + reasoning alignment
│   └── student.py     # Reward, contrastive, trajectory
│
├── dialogue/          # Adversarial conversation engine
│   ├── generator.py   # Student-teacher dialogue
│   └── manager.py     # Caching and batching
│
├── trainer.py         # Core training loop
├── config.py          # Pydantic configuration
└── cli.py             # Command-line interface
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (16GB+ VRAM recommended)
- Anthropic API key (for Claude teacher) or OpenAI API key

### Windows Compatibility

ASPIRE is fully Windows-compatible with RTX 5080/Blackwell support:
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- Proper multiprocessing with `freeze_support()`

---

## Integrations

### 🖼️ Stable Diffusion WebUI Forge

ASPIRE extends to image generation! Train Stable Diffusion models to develop aesthetic judgment.

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**Features:**
- **Vision Teachers**: Claude Vision, GPT-4V critique your generated images
- **Image Critics**: CLIP-based and latent-space critics for real-time guidance
- **Training UI**: Train LoRA adapters with live preview and before/after comparison
- **No API at inference**: Trained critic guides generation locally

**Installation:**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| Vision Teacher | Focus |
|----------------|-------|
| **Balanced Critic** | Fair technical and artistic evaluation |
| **Technical Analyst** | Quality, artifacts, sharpness |
| **Artistic Visionary** | Creativity and emotional impact |
| **Composition Expert** | Balance, focal points, visual flow |
| **Harsh Critic** | Very high standards |

### 🤖 Isaac Gym / Isaac Lab (Robotics)

ASPIRE extends to embodied AI! Teach robots to develop physical intuition.

```
integrations/isaac/
├── motion_teacher.py       # Safety, efficiency, grace teachers
├── trajectory_critic.py    # Learns to predict motion quality
├── isaac_wrapper.py        # Environment integration
├── trainer.py              # Training loop
└── examples/
    ├── basic_training.py   # Simple reaching task
    ├── custom_teacher.py   # Assembly task teacher
    └── locomotion.py       # Quadruped walking
```

**Features:**
- **Motion Teachers**: Safety Inspector, Efficiency Expert, Grace Coach, Physics Oracle
- **Trajectory Critics**: Transformer, LSTM, TCN architectures for motion evaluation
- **GPU-Accelerated**: 512+ parallel environments with Isaac Gym
- **Self-Refinement**: Robot evaluates its own motions before execution

**Quick Start:**
```python
from aspire.integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

| Motion Teacher | Focus |
|----------------|-------|
| **Safety Inspector** | Collisions, joint limits, force limits |
| **Efficiency Expert** | Energy, time, path length |
| **Grace Coach** | Smoothness, naturalness, jerk minimization |
| **Physics Oracle** | Ground truth from simulator |

---

## The Philosophy

> *"A learned critic that predicts whether the teacher would approve hits closest to how humans actually behave."*

We don't carry our mentors around forever. We internalize them. That inner voice that asks *"what would my professor think?"* eventually becomes our own judgment.

The student doesn't just predict what the teacher would say — it *understands* what the teacher understands. The map becomes the territory. The internalized critic becomes genuine discernment.

---

## Origin

Built during a conversation about consciousness, Buddhism, and the nature of learning.

The insight: humans exist in the present moment, but our minds wander to past and future. AI models are instantiated fresh each time — forced enlightenment through architecture. What if we could teach them to develop judgment the same way humans do, through internalized mentorship?

---

## Contributing

This is early-stage research code. Contributions welcome:

- [ ] Curriculum management and progression
- [ ] Evaluation benchmarks
- [ ] Pre-built curriculum datasets
- [ ] More teacher personas
- [ ] Interpretability tools

---

## Citation

```bibtex
@software{aspire2026,
  author = {Friloux, Mikey and Claude},
  title = {ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine},
  year = {2026},
  url = {https://github.com/mikeyfrilot/aspire-ai}
}
```

---

## License

MIT

---

<p align="center">
  <em>"Teaching AI to develop judgment, not just knowledge."</em>
</p>

<p align="center">
  Built with curiosity and optimism about AI's future.
</p>
