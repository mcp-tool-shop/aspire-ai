"""
Example: Training a quadruped robot with ASPIRE.

Teaches a legged robot to walk with good gait patterns
by internalizing feedback from motion quality experts.
"""

import numpy as np
import torch
from dataclasses import dataclass
from multiprocessing import freeze_support

from aspire.integrations.isaac.motion_teacher import (
    BaseMotionTeacher,
    MotionTeacher,
    MotionDimension,
    MotionCritique,
    TrajectoryData,
    SafetyInspector,
    EfficiencyExpert,
    GraceCoach,
)


class GaitAnalyst(BaseMotionTeacher):
    """
    Teacher specialized for legged locomotion.

    Evaluates:
    - Gait symmetry (legs should move in coordinated patterns)
    - Ground clearance (feet shouldn't drag)
    - Balance (center of mass should stay stable)
    - Stride efficiency (distance per step)
    """

    def __init__(
        self,
        num_legs: int = 4,
        target_stride_length: float = 0.3,    # meters
        min_ground_clearance: float = 0.05,   # meters
        symmetry_threshold: float = 0.1,      # phase difference tolerance
    ):
        super().__init__(
            name="Gait Analyst",
            description="Evaluates legged locomotion patterns",
            focus_dimensions=[
                MotionDimension.EFFICIENCY,
                MotionDimension.NATURALNESS,
                MotionDimension.STABILITY,
            ],
        )
        self.num_legs = num_legs
        self.target_stride_length = target_stride_length
        self.min_ground_clearance = min_ground_clearance
        self.symmetry_threshold = symmetry_threshold

    def critique(self, trajectory: TrajectoryData) -> MotionCritique:
        """Evaluate a locomotion trajectory."""
        strengths = []
        weaknesses = []
        suggestions = []
        scores = {}

        states = trajectory.states
        T = len(states)

        if T < 20:
            return MotionCritique(
                overall_score=5.0,
                reasoning="Trajectory too short for gait analysis",
                teacher_name=self.name,
            )

        # Assume state includes: [body_pos(3), body_vel(3), joint_angles(12), ...]
        # For a quadruped with 3 joints per leg

        # === Forward Progress ===
        if states.shape[1] >= 3:
            start_pos = states[0, :3]
            end_pos = states[-1, :3]
            forward_distance = np.linalg.norm(end_pos[:2] - start_pos[:2])
            duration = trajectory.timestamps[-1] - trajectory.timestamps[0]
            speed = forward_distance / (duration + 1e-6)

            if speed > 0.5:
                strengths.append(f"Good forward speed: {speed:.2f} m/s")
                progress_score = min(10, 7 + speed)
            elif speed > 0.2:
                progress_score = 5 + speed * 5
            else:
                weaknesses.append(f"Slow forward progress: {speed:.2f} m/s")
                suggestions.append("Increase stride frequency or length")
                progress_score = max(0, speed * 10)

            scores[MotionDimension.EFFICIENCY] = progress_score

        # === Stability (vertical oscillation) ===
        if states.shape[1] >= 3:
            z_positions = states[:, 2]  # Assume z is vertical
            z_variance = np.var(z_positions)
            z_range = np.max(z_positions) - np.min(z_positions)

            if z_range < 0.1:
                stability_score = 9.0
                strengths.append("Stable body height")
            elif z_range < 0.2:
                stability_score = 7.0
            else:
                stability_score = max(0, 7 - z_range * 10)
                weaknesses.append(f"Excessive vertical oscillation: {z_range:.2f}m")
                suggestions.append("Improve leg coordination for smoother gait")

            scores[MotionDimension.STABILITY] = stability_score

        # === Gait Symmetry (simplified) ===
        # In a real implementation, we'd analyze leg phases
        # Here we check velocity smoothness as a proxy
        if states.shape[1] >= 6:
            velocities = states[:, 3:6]  # Assume indices 3-5 are velocities
            vel_magnitude = np.linalg.norm(velocities, axis=1)
            vel_variance = np.var(vel_magnitude)

            if vel_variance < 0.1:
                symmetry_score = 9.0
                strengths.append("Consistent gait rhythm")
            elif vel_variance < 0.3:
                symmetry_score = 7.0
            else:
                symmetry_score = max(0, 7 - vel_variance * 5)
                weaknesses.append("Irregular gait pattern")
                suggestions.append("Train for more symmetric leg coordination")

            scores[MotionDimension.NATURALNESS] = symmetry_score

        # === Energy Efficiency ===
        if trajectory.energy_usage is not None:
            total_energy = np.sum(trajectory.energy_usage)
            distance = forward_distance if 'forward_distance' in dir() else 1.0
            cost_of_transport = total_energy / (distance + 1e-6)

            # Lower is better for CoT
            if cost_of_transport < 10:
                efficiency_score = 9.0
                strengths.append(f"Excellent efficiency (CoT: {cost_of_transport:.1f})")
            elif cost_of_transport < 50:
                efficiency_score = 7.0
            else:
                efficiency_score = max(0, 7 - cost_of_transport / 20)
                weaknesses.append(f"High energy cost (CoT: {cost_of_transport:.1f})")

        # === Overall Score ===
        if scores:
            overall = np.mean(list(scores.values()))
        else:
            overall = 5.0

        # Generate reasoning
        if overall >= 8.0:
            reasoning = "Excellent locomotion with efficient, stable gait. "
        elif overall >= 6.0:
            reasoning = "Functional locomotion with room for refinement. "
        else:
            reasoning = "Gait needs significant improvement for practical use. "

        return MotionCritique(
            overall_score=overall,
            dimension_scores=scores,
            reasoning=reasoning,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=suggestions,
            teacher_name=self.name,
        )


def main():
    """Train a quadruped with ASPIRE."""

    from aspire.integrations.isaac import (
        AspireIsaacTrainer,
        IsaacAspireConfig,
    )
    from aspire.integrations.isaac.isaac_wrapper import DummyIsaacEnv, AspireIsaacEnv

    print("=" * 60)
    print("ASPIRE Locomotion Training")
    print("Teaching a quadruped to walk with internalized judgment")
    print("=" * 60 + "\n")

    # Configuration
    config = IsaacAspireConfig()
    config.training.num_envs = 32
    config.training.episodes_per_epoch = 100
    config.training.epochs = 50

    # Create composite teacher for locomotion
    locomotion_teacher = MotionTeacher(
        personas=[
            "safety_inspector",    # Don't fall over!
            "efficiency_expert",   # Minimize energy
            "grace_coach",         # Smooth motion
        ],
        strategy="vote",
    )

    # Add our custom gait analyst
    gait_analyst = GaitAnalyst(
        num_legs=4,
        target_stride_length=0.25,
    )

    # For this example, use dummy environment
    # In practice, use Isaac Gym's Anymal or similar
    print("Using DummyIsaacEnv for demonstration.")
    print("For real quadruped training, use Isaac Gym's Anymal environment.\n")

    base_env = DummyIsaacEnv(
        num_envs=config.training.num_envs,
        state_dim=42,   # Quadruped: body(6) + joints(12x3)
        action_dim=12,  # 12 joint torques
        episode_length=200,
        device=config.device,
    )
    env = AspireIsaacEnv(base_env)

    # Create trainer
    trainer = AspireIsaacTrainer(
        env=env,
        config=config,
        teacher=locomotion_teacher,
    )

    # Train
    print("Starting training...\n")
    metrics = trainer.train(epochs=10)  # Reduced for demo

    # Show results
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    for m in metrics[-3:]:  # Last 3 epochs
        print(f"Epoch {m.epoch}:")
        print(f"  Motion Score: {m.mean_trajectory_score:.2f}/10")
        print(f"  Success Rate: {m.success_rate:.1%}")

    print("\nThe quadruped has learned to evaluate its own gait!")
    print("It can now self-improve without calling the teacher API.")


if __name__ == "__main__":
    freeze_support()
    main()
