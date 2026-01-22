"""
Example: Basic ASPIRE training with Isaac Gym.

This example shows how to train a robot arm to reach a target
while developing internalized judgment about motion quality.

Requirements:
    pip install isaacgym  # Follow NVIDIA installation guide
    # OR use the DummyIsaacEnv for testing without Isaac Gym
"""

import torch
from multiprocessing import freeze_support


def main():
    # Import ASPIRE Isaac components
    from aspire.integrations.isaac import (
        AspireIsaacTrainer,
        MotionTeacher,
        IsaacAspireConfig,
    )
    from aspire.integrations.isaac.isaac_wrapper import DummyIsaacEnv, AspireIsaacEnv

    # Configuration
    config = IsaacAspireConfig()
    config.training.num_envs = 16          # Parallel environments
    config.training.episodes_per_epoch = 50
    config.training.epochs = 20
    config.training.save_frequency = 5

    # Teacher configuration - what makes good robot motion?
    config.teacher.personas = [
        "safety_inspector",    # Don't hit things!
        "efficiency_expert",   # Don't waste energy
        "grace_coach",         # Move smoothly
    ]
    config.teacher.strategy = "vote"       # Combine all perspectives
    config.teacher.safety_weight = 2.0     # Safety is most important

    # Check if Isaac Gym is available
    try:
        from isaacgymenvs.tasks import FrankaCubeStack
        print("Isaac Gym found! Using real physics simulation.")
        env_name = "FrankaCubeStack-v0"
        use_dummy = False
    except ImportError:
        print("Isaac Gym not found. Using DummyIsaacEnv for demonstration.")
        print("Install Isaac Gym for real robot training.")
        use_dummy = True

    # Create environment
    if use_dummy:
        base_env = DummyIsaacEnv(
            num_envs=config.training.num_envs,
            state_dim=14,   # 7 joint positions + 7 velocities
            action_dim=7,   # 7 joint torques
            episode_length=100,
            device=config.device,
        )
        env = AspireIsaacEnv(base_env)
    else:
        from aspire.integrations.isaac.isaac_wrapper import create_isaac_env
        env = create_isaac_env(
            env_name,
            num_envs=config.training.num_envs,
            device=config.device,
        )

    # Create teacher
    teacher = MotionTeacher(
        personas=config.teacher.personas,
        strategy=config.teacher.strategy,
    )

    # Create trainer
    trainer = AspireIsaacTrainer(
        env=env,
        config=config,
        teacher=teacher,
    )

    # Training callback
    def on_epoch(metrics):
        print(f"  Callback: Epoch {metrics.epoch} complete")
        if metrics.mean_trajectory_score > 8.0:
            print("  Excellent motion quality achieved!")

    # Train!
    print("\n" + "=" * 60)
    print("Starting ASPIRE Embodied Training")
    print("=" * 60 + "\n")

    metrics = trainer.train(callback=on_epoch)

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating trained policy")
    print("=" * 60 + "\n")

    eval_results = trainer.evaluate(
        num_episodes=50,
        use_critic_refinement=True,
        deterministic=True,
    )

    print(f"Final Results:")
    print(f"  Mean reward: {eval_results['mean_reward']:.2f}")
    print(f"  Success rate: {eval_results['success_rate']:.1%}")
    print(f"  Mean motion score: {eval_results['mean_score']:.2f}/10")

    # The robot now has internalized judgment!
    print("\n" + "=" * 60)
    print("Training complete!")
    print("The robot has internalized motion judgment from the teachers.")
    print("It can now self-evaluate and refine its motions without teacher API calls.")
    print("=" * 60)


if __name__ == "__main__":
    freeze_support()  # Windows compatibility
    main()
