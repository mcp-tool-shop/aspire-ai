"""
Example: Creating a custom motion teacher for specialized robot tasks.

This example shows how to create a teacher for assembly tasks
that evaluates precision, stability, and part alignment.
"""

import numpy as np
from dataclasses import dataclass

from aspire.integrations.isaac.motion_teacher import (
    BaseMotionTeacher,
    MotionDimension,
    MotionCritique,
    TrajectoryData,
)


class AssemblyTeacher(BaseMotionTeacher):
    """
    Teacher specialized for assembly tasks.

    Evaluates:
    - Precision: How accurately parts are positioned
    - Stability: No wobbling or oscillation during placement
    - Alignment: Parts properly oriented before insertion
    - Force control: Gentle contact, appropriate insertion force
    """

    def __init__(
        self,
        position_tolerance: float = 0.001,   # 1mm
        angle_tolerance: float = 0.01,       # ~0.5 degrees
        max_contact_force: float = 10.0,     # Newtons
        oscillation_threshold: float = 0.1,
    ):
        super().__init__(
            name="Assembly Expert",
            description="Evaluates precision assembly motions",
            focus_dimensions=[
                MotionDimension.GOAL_ACHIEVEMENT,
                MotionDimension.STABILITY,
                MotionDimension.SMOOTHNESS,
            ],
        )
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance
        self.max_contact_force = max_contact_force
        self.oscillation_threshold = oscillation_threshold

    def critique(self, trajectory: TrajectoryData) -> MotionCritique:
        """Evaluate an assembly trajectory."""
        strengths = []
        weaknesses = []
        suggestions = []
        scores = {}

        # === Precision Analysis ===
        if trajectory.goal_distances is not None:
            final_distance = trajectory.goal_distances[-1]

            if final_distance < self.position_tolerance:
                precision_score = 10.0
                strengths.append(f"Excellent precision: {final_distance*1000:.2f}mm error")
            elif final_distance < self.position_tolerance * 5:
                precision_score = 8.0 - (final_distance / self.position_tolerance) * 0.4
                strengths.append("Good precision within acceptable tolerance")
            else:
                precision_score = max(0, 5.0 - final_distance * 100)
                weaknesses.append(f"Position error: {final_distance*1000:.1f}mm")
                suggestions.append("Improve approach trajectory for better precision")

            scores[MotionDimension.GOAL_ACHIEVEMENT] = precision_score

        # === Stability Analysis ===
        if len(trajectory.states) > 10:
            # Check for oscillations in the final approach phase
            final_states = trajectory.states[-20:]
            velocities = np.diff(final_states, axis=0)
            velocity_magnitude = np.linalg.norm(velocities, axis=1)

            # Look for sign changes (oscillation)
            sign_changes = np.sum(np.diff(np.sign(velocities[:, :3]), axis=0) != 0)
            oscillation_ratio = sign_changes / (len(velocities) * 3)

            if oscillation_ratio > self.oscillation_threshold:
                stability_score = max(0, 8.0 - oscillation_ratio * 20)
                weaknesses.append(f"Oscillation detected in approach ({oscillation_ratio:.1%})")
                suggestions.append("Use damping or reduce approach speed")
            else:
                stability_score = 9.0
                strengths.append("Stable approach without oscillation")

            scores[MotionDimension.STABILITY] = stability_score

        # === Force Control Analysis ===
        if trajectory.forces is not None:
            max_force = np.max(np.linalg.norm(trajectory.forces, axis=-1))
            mean_force = np.mean(np.linalg.norm(trajectory.forces, axis=-1))

            if max_force > self.max_contact_force:
                force_score = max(0, 8.0 - (max_force - self.max_contact_force) * 0.5)
                weaknesses.append(f"Excessive contact force: {max_force:.1f}N")
                suggestions.append("Reduce approach speed or use force feedback")
            elif max_force > self.max_contact_force * 0.7:
                force_score = 8.0
                strengths.append("Good force control")
            else:
                force_score = 10.0
                strengths.append("Excellent gentle contact")

            # Note: We'd need a custom dimension for force
            # For now, factor into overall score

        # === Smoothness During Approach ===
        if len(trajectory.states) > 4:
            dt = np.diff(trajectory.timestamps)
            velocities = np.diff(trajectory.states, axis=0) / dt[:, np.newaxis]

            if len(velocities) > 2:
                accelerations = np.diff(velocities, axis=0) / dt[1:, np.newaxis]
                jerks = np.diff(accelerations, axis=0) / dt[2:, np.newaxis]
                jerk_magnitude = np.mean(np.linalg.norm(jerks, axis=1))

                if jerk_magnitude < 50:
                    smoothness_score = 9.5
                    strengths.append("Very smooth motion profile")
                elif jerk_magnitude < 100:
                    smoothness_score = 8.0
                else:
                    smoothness_score = max(0, 8.0 - jerk_magnitude / 50)
                    weaknesses.append("Jerky motion detected")
                    suggestions.append("Use minimum-jerk trajectory planning")

                scores[MotionDimension.SMOOTHNESS] = smoothness_score

        # === Compute Overall Score ===
        if scores:
            # Weight precision highest for assembly
            weights = {
                MotionDimension.GOAL_ACHIEVEMENT: 2.0,
                MotionDimension.STABILITY: 1.5,
                MotionDimension.SMOOTHNESS: 1.0,
            }

            total_weight = sum(weights.get(d, 1.0) for d in scores)
            overall = sum(
                scores[d] * weights.get(d, 1.0) for d in scores
            ) / total_weight
        else:
            overall = 5.0

        # Generate reasoning
        if overall >= 8.5:
            reasoning = "Excellent assembly motion with high precision and stability. "
        elif overall >= 7.0:
            reasoning = "Good assembly motion with minor improvements possible. "
        elif overall >= 5.0:
            reasoning = "Acceptable but needs refinement for production use. "
        else:
            reasoning = "Significant issues that would cause assembly failures. "

        if weaknesses:
            reasoning += f"Main concerns: {'; '.join(weaknesses[:2])}."

        return MotionCritique(
            overall_score=overall,
            dimension_scores=scores,
            reasoning=reasoning,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=suggestions,
            teacher_name=self.name,
        )


# Example usage
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    # Create a sample trajectory (simulated peg-in-hole task)
    T = 100
    timestamps = np.linspace(0, 2.0, T)

    # Simulate approaching a hole and inserting
    approach_phase = np.linspace(0, 1, T // 2)
    insertion_phase = np.linspace(1, 1.1, T // 2)
    z_position = np.concatenate([approach_phase, insertion_phase])

    # Add some noise
    z_position += np.random.randn(T) * 0.001

    # States: [x, y, z, vx, vy, vz]
    states = np.zeros((T, 6))
    states[:, 2] = z_position
    states[:, 5] = np.gradient(z_position) / np.diff(timestamps, prepend=0)

    # Actions
    actions = np.random.randn(T, 3) * 0.01

    # Goal distances (decreasing)
    goal_distances = np.linspace(0.1, 0.002, T)

    # Create trajectory data
    trajectory = TrajectoryData(
        states=states,
        actions=actions,
        timestamps=timestamps,
        goal_distances=goal_distances,
    )

    # Create teacher and critique
    teacher = AssemblyTeacher(
        position_tolerance=0.005,  # 5mm for this demo
    )

    critique = teacher.critique(trajectory)

    print("Assembly Teacher Critique")
    print("=" * 50)
    print(f"Overall Score: {critique.overall_score:.1f}/10")
    print(f"\nReasoning: {critique.reasoning}")
    print(f"\nStrengths:")
    for s in critique.strengths:
        print(f"  + {s}")
    print(f"\nWeaknesses:")
    for w in critique.weaknesses:
        print(f"  - {w}")
    print(f"\nSuggestions:")
    for s in critique.improvement_suggestions:
        print(f"  > {s}")
