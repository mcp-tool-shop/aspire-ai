# ASPIRE for Forge

**Teaching image generation models to develop aesthetic judgment.**

## What is ASPIRE?

ASPIRE (Adversarial Student-Professor Internalized Reasoning Engine) is a new approach to fine-tuning AI models. Instead of training on static datasets, models learn through adversarial dialogue with teacher models.

For image generation, this means:

1. **Generate** an image with the current model
2. **Critique** it using a vision-language model (Claude, GPT-4V)
3. **Learn** from the critique through an internalized critic
4. **Improve** future generations based on internalized aesthetic judgment

## Features

### Critic-Guided Generation

Use a trained critic to guide generation toward higher aesthetic quality:

- **No API calls at inference** - the critic runs locally
- **Multi-dimensional evaluation** - composition, color, lighting, style
- **Adjustable influence** - control how much the critic affects generation

### Training Interface

Train LoRA adapters with ASPIRE methodology:

- **Multiple teacher personas** - Technical Analyst, Artistic Visionary, Harsh Critic, etc.
- **Live preview** - watch the model improve in real-time
- **Before/after comparison** - evaluate training results

### Vision Teachers

Choose your aesthetic mentor:

| Teacher | Style |
|---------|-------|
| **Balanced Critic** | Fair, considers both technical and artistic merit |
| **Technical Analyst** | Focuses on quality, artifacts, sharpness |
| **Artistic Visionary** | Emphasizes creativity and emotional impact |
| **Composition Expert** | Analyzes balance, focal points, visual flow |
| **Color Theorist** | Evaluates color harmony and mood |
| **Harsh Critic** | Very high standards, finds every flaw |
| **Encouraging Mentor** | Supportive while still honest |

## Usage

### Critic-Guided Generation

1. Enable "ASPIRE Critic Guidance" in the accordion below the prompt
2. Adjust critic strength (0 = no influence, 1 = full influence)
3. Set quality threshold (generation refines until this score is reached)
4. Generate!

### Training

1. Go to the "ASPIRE Training" script (txt2img tab)
2. Upload prompts or enter them directly
3. Select teacher model and persona
4. Configure LoRA parameters
5. Start training
6. Watch the live preview and metrics

## Requirements

- **For training**: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` environment variable
- **For inference**: Just the trained critic model (no API needed)

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   SD Model  │ ──> │   Image     │ ──> │   Vision    │
│ (Student)   │     │ Generation  │     │   Teacher   │
└─────────────┘     └─────────────┘     └──────┬──────┘
      ▲                                        │
      │                                        │ Critique
      │             ┌─────────────┐            │
      └──────────── │   Critic    │ <──────────┘
         Learn      │ (Internalized│   Learn to
         from       │  Judgment)   │   Predict
                    └─────────────┘

After training: Critic guides generation WITHOUT the teacher
```

## The Philosophy

> "A learned critic that predicts whether the teacher would approve
> hits closest to how humans actually behave."

We don't carry our art teachers around forever. We internalize their aesthetic sense. Their voice becomes part of our inner dialogue when we create.

ASPIRE gives image models that same experience.

---

*Part of the [ASPIRE project](https://github.com/mikeyfrilot/aspire-ai)*
