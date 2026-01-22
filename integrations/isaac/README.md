# ASPIRE for Isaac Gym/Lab

**Teaching robots to develop physical intuition through internalized judgment.**

## The Idea

Traditional robot learning:
- **Reward engineering**: Hand-craft reward functions (brittle, hard to tune)
- **Imitation learning**: Copy demonstrations (limited to seen behaviors)
- **RL from scratch**: Learn by trial and error (expensive, dangerous)

**ASPIRE for robotics**: A teacher (safety expert, efficiency analyst, movement coach) critiques the robot's motions. The robot internalizes this judgment and can self-evaluate before executing.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Robot Policy  │ ──> │  Action/State   │ ──> │  Motion Teacher │
│    (Student)    │     │   Trajectory    │     │  (Expert Panel) │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
        ▲                                                 │
        │              ┌─────────────────┐               │ Critique
        └───────────── │  Motion Critic  │ <─────────────┘
           Learn from  │  (Internalized  │   Learn to
                       │   Judgment)     │   Predict
                       └─────────────────┘

After training: Critic guides action refinement WITHOUT the teacher
```

## Quick Start

### Installation

```bash
# ASPIRE core
pip install aspire-ai

# Isaac Gym (follow NVIDIA's installation guide)
# https://developer.nvidia.com/isaac-gym

# OR Isaac Lab (newer version)
# https://isaac-sim.github.io/IsaacLab/
```

### Basic Training

```python
from aspire.integrations.isaac import (
    AspireIsaacTrainer,
    MotionTeacher,
    IsaacAspireConfig,
)

# Configure training
config = IsaacAspireConfig()
config.training.num_envs = 512      # Parallel GPU environments
config.training.epochs = 100

# Create teacher committee
teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

# Train!
trainer = AspireIsaacTrainer(
    env="FrankaCubeStack-v0",
    config=config,
    teacher=teacher,
)
trainer.train()
```

### Try Without Isaac Gym

We include a `DummyIsaacEnv` for testing the ASPIRE methodology without installing Isaac Gym:

```python
from aspire.integrations.isaac.isaac_wrapper import DummyIsaacEnv, AspireIsaacEnv

# Simulates a simple reaching task
env = DummyIsaacEnv(num_envs=16)
wrapped_env = AspireIsaacEnv(env)

# Now train as normal
trainer = AspireIsaacTrainer(env=wrapped_env, ...)
```

## Motion Teachers

Different teachers evaluate motion from different perspectives:

| Teacher | Focus | Key Questions |
|---------|-------|---------------|
| **Safety Inspector** | Collision avoidance, joint limits, forces | "Will this hurt someone or break something?" |
| **Efficiency Expert** | Energy, time, path length | "Is this the most efficient way?" |
| **Grace Coach** | Smoothness, naturalness, jerk | "Does this look robotic or natural?" |
| **Physics Oracle** | Ground truth from simulator | "What actually happened?" |

### Composite Teachers

Combine multiple perspectives:

```python
teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",  # or "rotate", "debate"
)
```

### Custom Teachers

Create domain-specific teachers for your tasks:

```python
from aspire.integrations.isaac.motion_teacher import BaseMotionTeacher

class AssemblyTeacher(BaseMotionTeacher):
    """Evaluates precision assembly motions."""

    def critique(self, trajectory):
        # Evaluate precision, stability, force control
        ...
        return MotionCritique(
            overall_score=8.5,
            reasoning="Good precision but approach was jerky",
            ...
        )
```

See `examples/custom_teacher.py` for a complete example.

## Trajectory Critic

The critic learns to predict what the teacher would think:

```python
from aspire.integrations.isaac import TrajectoryCritic, CriticConfig

config = CriticConfig(
    architecture="transformer",  # or "lstm", "tcn", "mlp"
    hidden_dim=256,
    num_layers=4,
    predict_score=True,          # "This deserves a 7/10"
    predict_reasoning=True,       # "Because the motion was jerky"
    predict_improvement=True,     # "Try this action instead"
)

critic = TrajectoryCritic(config)
```

### Critic Architectures

| Architecture | Best For | Notes |
|--------------|----------|-------|
| **Transformer** | Long trajectories | Captures global dependencies |
| **LSTM** | Sequential tasks | Good for variable-length |
| **TCN** | Real-time | Fast temporal convolutions |
| **MLP** | Short horizons | Simple baseline |

## Training Configuration

```python
from aspire.integrations.isaac import IsaacAspireConfig

config = IsaacAspireConfig()

# Environment
config.training.num_envs = 512          # GPU-parallel environments
config.training.max_episode_length = 256

# Training
config.training.epochs = 100
config.training.episodes_per_epoch = 100
config.training.critic_lr = 3e-4
config.training.policy_lr = 1e-4

# Teacher weights (safety is paramount!)
config.teacher.safety_weight = 2.0
config.teacher.efficiency_weight = 1.0
config.teacher.smoothness_weight = 0.5

# Critic architecture
config.critic.architecture = "transformer"
config.critic.hidden_dim = 256
```

## Examples

### Reaching Task
```bash
python -m aspire.integrations.isaac.examples.basic_training
```

### Custom Assembly Teacher
```bash
python -m aspire.integrations.isaac.examples.custom_teacher
```

### Quadruped Locomotion
```bash
python -m aspire.integrations.isaac.examples.locomotion
```

## Supported Environments

Works with any Isaac Gym/Lab environment:

| Environment | Task | Notes |
|-------------|------|-------|
| FrankaCubeStack | Pick and place | Great for starting |
| FrankaCabinetPCH | Cabinet manipulation | Contact-rich |
| Anymal | Quadruped walking | Locomotion |
| AllegroHand | Dexterous manipulation | High-DoF |
| Humanoid | Bipedal locomotion | Challenging |

## The Philosophy

> *"We don't carry our movement coaches around forever. We internalize their sense of what good motion feels like."*

A dancer learns from their teacher, but eventually develops their own sense of grace. A martial artist internalizes their master's judgment of what makes a good technique.

ASPIRE gives robots that same experience. The critic becomes an inner coach that evaluates motions before execution—no teacher API calls needed in deployment.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (16GB+ VRAM recommended)
- Isaac Gym or Isaac Lab (optional for testing)

### Windows Compatibility

Fully Windows-compatible with RTX 5080/Blackwell support:
- `dataloader_num_workers=0` (no multiprocessing issues)
- Proper `freeze_support()` in examples

---

*Part of the [ASPIRE project](https://github.com/mikeyfrilot/aspire-ai)*
