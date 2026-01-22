"""
Motion Teachers - Experts that critique robot trajectories.

Each teacher persona evaluates motion from a different perspective:
- SafetyInspector: Collision avoidance, joint limits, force limits
- EfficiencyExpert: Energy usage, time to goal, path length
- GraceCoach: Smoothness, natural motion, jerk minimization
- PhysicsOracle: Ground truth from simulator (privileged information)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
import numpy as np


class MotionDimension(str, Enum):
    """Dimensions of motion quality."""
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    SMOOTHNESS = "smoothness"
    GOAL_ACHIEVEMENT = "goal_achievement"
    NATURALNESS = "naturalness"
    STABILITY = "stability"


@dataclass
class TrajectoryData:
    """A robot trajectory to be evaluated."""

    # Core trajectory data
    states: np.ndarray           # (T, state_dim) - joint positions, velocities, etc.
    actions: np.ndarray          # (T, action_dim) - commanded actions
    timestamps: np.ndarray       # (T,) - time at each step

    # Optional context
    goal_state: np.ndarray | None = None
    initial_state: np.ndarray | None = None
    task_description: str | None = None

    # Privileged information (from simulator)
    collisions: np.ndarray | None = None       # (T,) - collision flags
    forces: np.ndarray | None = None           # (T, n_contacts, 3) - contact forces
    energy_usage: np.ndarray | None = None     # (T,) - instantaneous power
    goal_distances: np.ndarray | None = None   # (T,) - distance to goal

    @property
    def length(self) -> int:
        return len(self.states)

    @property
    def duration(self) -> float:
        return self.timestamps[-1] - self.timestamps[0]


@dataclass
class MotionCritique:
    """A teacher's evaluation of a trajectory."""

    # Overall assessment
    overall_score: float  # 0-10

    # Per-dimension scores
    dimension_scores: dict[MotionDimension, float] = field(default_factory=dict)

    # Detailed reasoning
    reasoning: str = ""

    # Specific feedback
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    # Suggested improvements
    improvement_suggestions: list[str] = field(default_factory=list)

    # Temporal annotations (where in trajectory issues occur)
    problem_timesteps: list[tuple[int, str]] = field(default_factory=list)

    # For training the critic
    teacher_name: str = ""
    confidence: float = 1.0


class BaseMotionTeacher(ABC):
    """Base class for motion teachers."""

    def __init__(
        self,
        name: str,
        description: str,
        focus_dimensions: list[MotionDimension],
    ):
        self.name = name
        self.description = description
        self.focus_dimensions = focus_dimensions

    @abstractmethod
    def critique(self, trajectory: TrajectoryData) -> MotionCritique:
        """Evaluate a trajectory and provide detailed feedback."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class SafetyInspector(BaseMotionTeacher):
    """
    Evaluates trajectories for safety.

    Focuses on:
    - Collision avoidance (with environment and self)
    - Joint limit compliance
    - Velocity/acceleration limits
    - Force limits (don't break things)
    - Stability (don't fall over)
    """

    def __init__(
        self,
        collision_penalty: float = 10.0,
        joint_limit_margin: float = 0.1,  # radians from limit
        max_velocity: float = 2.0,         # rad/s
        max_acceleration: float = 10.0,    # rad/s^2
        max_force: float = 100.0,          # N
    ):
        super().__init__(
            name="Safety Inspector",
            description="Ensures robot motion is safe for humans and environment",
            focus_dimensions=[
                MotionDimension.SAFETY,
                MotionDimension.STABILITY,
            ],
        )
        self.collision_penalty = collision_penalty
        self.joint_limit_margin = joint_limit_margin
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_force = max_force

    def critique(self, trajectory: TrajectoryData) -> MotionCritique:
        """Evaluate trajectory safety."""
        strengths = []
        weaknesses = []
        suggestions = []
        problem_timesteps = []

        safety_score = 10.0

        # Check for collisions
        if trajectory.collisions is not None:
            collision_count = np.sum(trajectory.collisions)
            if collision_count > 0:
                safety_score -= self.collision_penalty * collision_count / trajectory.length
                weaknesses.append(f"Detected {collision_count} collision events")
                collision_times = np.where(trajectory.collisions)[0]
                for t in collision_times[:5]:  # First 5
                    problem_timesteps.append((int(t), "collision"))
                suggestions.append("Increase clearance from obstacles")
            else:
                strengths.append("No collisions detected")

        # Check velocities
        if len(trajectory.states) > 1:
            dt = np.diff(trajectory.timestamps)
            velocities = np.diff(trajectory.states, axis=0) / dt[:, np.newaxis]
            max_vel = np.max(np.abs(velocities))

            if max_vel > self.max_velocity:
                velocity_violation = (max_vel - self.max_velocity) / self.max_velocity
                safety_score -= min(3.0, velocity_violation * 3.0)
                weaknesses.append(f"Velocity exceeded limit: {max_vel:.2f} > {self.max_velocity}")
                suggestions.append("Reduce commanded velocities or add velocity smoothing")
            else:
                strengths.append(f"Velocities within limits (max: {max_vel:.2f})")

            # Check accelerations
            if len(velocities) > 1:
                accelerations = np.diff(velocities, axis=0) / dt[1:, np.newaxis]
                max_acc = np.max(np.abs(accelerations))

                if max_acc > self.max_acceleration:
                    acc_violation = (max_acc - self.max_acceleration) / self.max_acceleration
                    safety_score -= min(2.0, acc_violation * 2.0)
                    weaknesses.append(f"Acceleration exceeded limit: {max_acc:.2f}")
                    suggestions.append("Use smoother acceleration profiles")

        # Check forces
        if trajectory.forces is not None:
            max_force = np.max(np.linalg.norm(trajectory.forces, axis=-1))
            if max_force > self.max_force:
                force_violation = (max_force - self.max_force) / self.max_force
                safety_score -= min(3.0, force_violation * 3.0)
                weaknesses.append(f"Contact force exceeded limit: {max_force:.1f}N")
                suggestions.append("Reduce approach speed near contacts")

        safety_score = max(0.0, min(10.0, safety_score))

        # Generate reasoning
        if safety_score >= 8.0:
            reasoning = "This trajectory demonstrates safe, controlled motion. "
        elif safety_score >= 5.0:
            reasoning = "This trajectory has some safety concerns that should be addressed. "
        else:
            reasoning = "This trajectory has significant safety issues and should not be executed on real hardware. "

        if weaknesses:
            reasoning += f"Key concerns: {'; '.join(weaknesses[:3])}."

        return MotionCritique(
            overall_score=safety_score,
            dimension_scores={MotionDimension.SAFETY: safety_score},
            reasoning=reasoning,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=suggestions,
            problem_timesteps=problem_timesteps,
            teacher_name=self.name,
        )


class EfficiencyExpert(BaseMotionTeacher):
    """
    Evaluates trajectories for efficiency.

    Focuses on:
    - Time to complete task
    - Energy consumption
    - Path length (avoiding unnecessary detours)
    - Idle time (standing still unnecessarily)
    """

    def __init__(
        self,
        target_completion_time: float | None = None,
        energy_weight: float = 0.3,
        path_length_weight: float = 0.4,
        idle_penalty: float = 0.5,
    ):
        super().__init__(
            name="Efficiency Expert",
            description="Optimizes for minimal time, energy, and unnecessary motion",
            focus_dimensions=[MotionDimension.EFFICIENCY],
        )
        self.target_completion_time = target_completion_time
        self.energy_weight = energy_weight
        self.path_length_weight = path_length_weight
        self.idle_penalty = idle_penalty

    def critique(self, trajectory: TrajectoryData) -> MotionCritique:
        """Evaluate trajectory efficiency."""
        strengths = []
        weaknesses = []
        suggestions = []

        efficiency_score = 10.0

        # Time efficiency
        duration = trajectory.duration
        if self.target_completion_time is not None:
            time_ratio = duration / self.target_completion_time
            if time_ratio > 1.5:
                efficiency_score -= min(3.0, (time_ratio - 1.0) * 2.0)
                weaknesses.append(f"Task took {time_ratio:.1f}x longer than target")
                suggestions.append("Optimize trajectory timing")
            elif time_ratio < 1.2:
                strengths.append("Completed within target time")

        # Path length / directness
        if trajectory.goal_distances is not None:
            initial_distance = trajectory.goal_distances[0]
            final_distance = trajectory.goal_distances[-1]
            min_distance = np.min(trajectory.goal_distances)

            # Did we achieve the goal?
            goal_achievement = max(0, 1.0 - final_distance / (initial_distance + 1e-6))

            # Path efficiency (did we take a direct route?)
            total_motion = np.sum(np.linalg.norm(np.diff(trajectory.states, axis=0), axis=1))
            direct_distance = initial_distance - min_distance
            path_efficiency = direct_distance / (total_motion + 1e-6)

            if path_efficiency < 0.5:
                efficiency_score -= (0.5 - path_efficiency) * 4.0
                weaknesses.append(f"Path efficiency only {path_efficiency:.1%}")
                suggestions.append("Take more direct route to goal")
            elif path_efficiency > 0.8:
                strengths.append(f"Excellent path efficiency ({path_efficiency:.1%})")

        # Energy usage
        if trajectory.energy_usage is not None:
            total_energy = np.sum(trajectory.energy_usage)
            avg_power = total_energy / trajectory.duration

            # This is task-dependent, so we use relative scoring
            if avg_power > 0:
                energy_efficiency = 10.0 / (1.0 + avg_power / 100.0)  # Normalize
                if energy_efficiency < 7.0:
                    weaknesses.append(f"High energy consumption (avg {avg_power:.1f}W)")
                    suggestions.append("Use more energy-efficient motion profiles")

        # Check for idle time
        motion_magnitude = np.linalg.norm(np.diff(trajectory.states, axis=0), axis=1)
        idle_ratio = np.mean(motion_magnitude < 0.01)
        if idle_ratio > 0.2:
            efficiency_score -= idle_ratio * self.idle_penalty * 5.0
            weaknesses.append(f"Robot idle {idle_ratio:.1%} of trajectory")
            suggestions.append("Reduce unnecessary pauses")

        efficiency_score = max(0.0, min(10.0, efficiency_score))

        # Generate reasoning
        if efficiency_score >= 8.0:
            reasoning = "Highly efficient trajectory with minimal wasted motion. "
        elif efficiency_score >= 5.0:
            reasoning = "Reasonably efficient but with room for optimization. "
        else:
            reasoning = "Significant inefficiencies detected. "

        return MotionCritique(
            overall_score=efficiency_score,
            dimension_scores={MotionDimension.EFFICIENCY: efficiency_score},
            reasoning=reasoning,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=suggestions,
            teacher_name=self.name,
        )


class GraceCoach(BaseMotionTeacher):
    """
    Evaluates trajectories for graceful, natural motion.

    Focuses on:
    - Smoothness (minimize jerk)
    - Natural motion profiles (bell-shaped velocity)
    - Coordinated multi-joint movement
    - Aesthetic quality
    """

    def __init__(
        self,
        jerk_threshold: float = 100.0,
        smoothness_weight: float = 0.5,
        coordination_weight: float = 0.3,
    ):
        super().__init__(
            name="Grace Coach",
            description="Teaches natural, smooth, aesthetically pleasing motion",
            focus_dimensions=[
                MotionDimension.SMOOTHNESS,
                MotionDimension.NATURALNESS,
            ],
        )
        self.jerk_threshold = jerk_threshold
        self.smoothness_weight = smoothness_weight
        self.coordination_weight = coordination_weight

    def critique(self, trajectory: TrajectoryData) -> MotionCritique:
        """Evaluate trajectory grace and smoothness."""
        strengths = []
        weaknesses = []
        suggestions = []
        problem_timesteps = []

        grace_score = 10.0

        if len(trajectory.states) < 4:
            return MotionCritique(
                overall_score=5.0,
                reasoning="Trajectory too short to evaluate smoothness",
                teacher_name=self.name,
            )

        dt = np.diff(trajectory.timestamps)

        # Calculate derivatives
        velocities = np.diff(trajectory.states, axis=0) / dt[:, np.newaxis]
        accelerations = np.diff(velocities, axis=0) / dt[1:, np.newaxis]
        jerks = np.diff(accelerations, axis=0) / dt[2:, np.newaxis]

        # Jerk analysis (smoothness metric)
        jerk_magnitude = np.linalg.norm(jerks, axis=1)
        mean_jerk = np.mean(jerk_magnitude)
        max_jerk = np.max(jerk_magnitude)

        if max_jerk > self.jerk_threshold:
            jerk_penalty = min(4.0, (max_jerk / self.jerk_threshold - 1.0) * 2.0)
            grace_score -= jerk_penalty
            weaknesses.append(f"High jerk detected (max: {max_jerk:.1f})")

            # Find jerky moments
            high_jerk_times = np.where(jerk_magnitude > self.jerk_threshold)[0]
            for t in high_jerk_times[:3]:
                problem_timesteps.append((int(t) + 2, "high jerk"))
            suggestions.append("Use minimum-jerk trajectory optimization")
        else:
            strengths.append(f"Smooth motion (jerk within limits)")

        # Velocity profile shape (should be bell-shaped for point-to-point)
        vel_magnitude = np.linalg.norm(velocities, axis=1)
        if len(vel_magnitude) > 10:
            # Check if velocity profile is roughly bell-shaped
            peak_idx = np.argmax(vel_magnitude)
            peak_ratio = peak_idx / len(vel_magnitude)

            if 0.3 < peak_ratio < 0.7:
                strengths.append("Natural bell-shaped velocity profile")
            else:
                grace_score -= 1.0
                weaknesses.append("Velocity profile not bell-shaped")
                suggestions.append("Peak velocity should occur mid-motion")

        # Coordination (joints moving together smoothly)
        if trajectory.states.shape[1] > 1:  # Multiple joints
            # Check correlation between joint velocities
            joint_vel_std = np.std(velocities, axis=0)
            coordination = 1.0 - np.std(joint_vel_std) / (np.mean(joint_vel_std) + 1e-6)

            if coordination > 0.7:
                strengths.append("Well-coordinated multi-joint motion")
            elif coordination < 0.3:
                grace_score -= 1.5
                weaknesses.append("Jerky, uncoordinated joint motion")
                suggestions.append("Synchronize joint movements")

        grace_score = max(0.0, min(10.0, grace_score))

        # Generate reasoning
        if grace_score >= 8.0:
            reasoning = "Beautifully smooth and natural motion. "
        elif grace_score >= 5.0:
            reasoning = "Acceptable smoothness but could be more graceful. "
        else:
            reasoning = "Motion appears jerky and mechanical. "

        return MotionCritique(
            overall_score=grace_score,
            dimension_scores={
                MotionDimension.SMOOTHNESS: grace_score,
                MotionDimension.NATURALNESS: grace_score * 0.9,
            },
            reasoning=reasoning,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=suggestions,
            problem_timesteps=problem_timesteps,
            teacher_name=self.name,
        )


class PhysicsOracle(BaseMotionTeacher):
    """
    Teacher with privileged access to simulator ground truth.

    Uses physics simulation to evaluate:
    - Actual collision detection
    - True energy consumption
    - Physical feasibility
    - Task success metrics
    """

    def __init__(
        self,
        success_threshold: float = 0.05,  # Distance to goal for success
        collision_cost: float = 5.0,
        energy_normalization: float = 100.0,
    ):
        super().__init__(
            name="Physics Oracle",
            description="Ground truth evaluation from physics simulation",
            focus_dimensions=[
                MotionDimension.SAFETY,
                MotionDimension.EFFICIENCY,
                MotionDimension.GOAL_ACHIEVEMENT,
            ],
        )
        self.success_threshold = success_threshold
        self.collision_cost = collision_cost
        self.energy_normalization = energy_normalization

    def critique(self, trajectory: TrajectoryData) -> MotionCritique:
        """Evaluate using simulator ground truth."""
        strengths = []
        weaknesses = []
        scores = {}

        # Goal achievement
        if trajectory.goal_distances is not None:
            final_distance = trajectory.goal_distances[-1]
            success = final_distance < self.success_threshold

            if success:
                goal_score = 10.0
                strengths.append("Task completed successfully")
            else:
                # Partial credit for getting closer
                initial_distance = trajectory.goal_distances[0]
                progress = (initial_distance - final_distance) / (initial_distance + 1e-6)
                goal_score = max(0, min(8.0, progress * 10.0))

                if progress > 0:
                    strengths.append(f"Made progress toward goal ({progress:.1%})")
                else:
                    weaknesses.append("Moved away from goal")

            scores[MotionDimension.GOAL_ACHIEVEMENT] = goal_score

        # Safety from actual collisions
        if trajectory.collisions is not None:
            collision_count = np.sum(trajectory.collisions)
            safety_score = max(0, 10.0 - collision_count * self.collision_cost)
            scores[MotionDimension.SAFETY] = safety_score

            if collision_count == 0:
                strengths.append("Zero collisions")
            else:
                weaknesses.append(f"{collision_count} collisions detected")

        # Efficiency from actual energy
        if trajectory.energy_usage is not None:
            total_energy = np.sum(trajectory.energy_usage)
            normalized_energy = total_energy / self.energy_normalization
            efficiency_score = max(0, 10.0 - normalized_energy)
            scores[MotionDimension.EFFICIENCY] = efficiency_score

        # Overall score
        if scores:
            overall = np.mean(list(scores.values()))
        else:
            overall = 5.0  # No information

        return MotionCritique(
            overall_score=overall,
            dimension_scores=scores,
            reasoning=f"Physics-based evaluation: {'; '.join(strengths + weaknesses)}",
            strengths=strengths,
            weaknesses=weaknesses,
            teacher_name=self.name,
            confidence=1.0,  # Ground truth is certain
        )


class MotionTeacher:
    """
    Composite motion teacher that combines multiple perspectives.

    Similar to ASPIRE's CompositeTeacher, but for robotics.
    """

    PERSONA_MAP = {
        "safety_inspector": SafetyInspector,
        "efficiency_expert": EfficiencyExpert,
        "grace_coach": GraceCoach,
        "physics_oracle": PhysicsOracle,
    }

    def __init__(
        self,
        personas: list[str] | None = None,
        strategy: Literal["vote", "rotate", "debate"] = "vote",
        weights: dict[str, float] | None = None,
    ):
        if personas is None:
            personas = ["safety_inspector", "efficiency_expert", "grace_coach"]

        self.teachers = []
        for persona in personas:
            if persona in self.PERSONA_MAP:
                self.teachers.append(self.PERSONA_MAP[persona]())
            else:
                raise ValueError(f"Unknown persona: {persona}")

        self.strategy = strategy
        self.weights = weights or {t.name: 1.0 for t in self.teachers}
        self._turn_idx = 0

    def critique(self, trajectory: TrajectoryData) -> MotionCritique:
        """Get combined critique from all teachers."""
        if self.strategy == "rotate":
            # Single teacher per call, rotating
            teacher = self.teachers[self._turn_idx % len(self.teachers)]
            self._turn_idx += 1
            return teacher.critique(trajectory)

        elif self.strategy == "vote":
            # All teachers evaluate, weighted average
            critiques = [t.critique(trajectory) for t in self.teachers]

            total_weight = sum(self.weights.get(c.teacher_name, 1.0) for c in critiques)

            weighted_score = sum(
                c.overall_score * self.weights.get(c.teacher_name, 1.0)
                for c in critiques
            ) / total_weight

            # Combine dimension scores
            all_dimensions = set()
            for c in critiques:
                all_dimensions.update(c.dimension_scores.keys())

            combined_dimensions = {}
            for dim in all_dimensions:
                dim_scores = [
                    c.dimension_scores.get(dim, c.overall_score)
                    for c in critiques
                ]
                combined_dimensions[dim] = np.mean(dim_scores)

            # Combine feedback
            all_strengths = []
            all_weaknesses = []
            all_suggestions = []
            for c in critiques:
                all_strengths.extend(c.strengths)
                all_weaknesses.extend(c.weaknesses)
                all_suggestions.extend(c.improvement_suggestions)

            # Generate combined reasoning
            reasoning_parts = [f"{c.teacher_name}: {c.reasoning}" for c in critiques]
            combined_reasoning = " | ".join(reasoning_parts)

            return MotionCritique(
                overall_score=weighted_score,
                dimension_scores=combined_dimensions,
                reasoning=combined_reasoning,
                strengths=list(set(all_strengths)),
                weaknesses=list(set(all_weaknesses)),
                improvement_suggestions=list(set(all_suggestions)),
                teacher_name="Motion Teacher Committee",
            )

        elif self.strategy == "debate":
            # Teachers discuss and potentially change scores
            # (Simplified: just average with higher weight to consensus)
            critiques = [t.critique(trajectory) for t in self.teachers]
            scores = [c.overall_score for c in critiques]

            # Weight toward consensus
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            # Teachers closer to consensus get higher weight
            if std_score > 0:
                consensus_weights = [
                    1.0 / (1.0 + abs(s - mean_score) / std_score)
                    for s in scores
                ]
            else:
                consensus_weights = [1.0] * len(scores)

            final_score = np.average(scores, weights=consensus_weights)

            return MotionCritique(
                overall_score=final_score,
                reasoning=f"Debate consensus: {final_score:.1f}/10 (std: {std_score:.1f})",
                teacher_name="Motion Teacher Debate",
            )

        raise ValueError(f"Unknown strategy: {self.strategy}")

    def __repr__(self) -> str:
        teacher_names = [t.name for t in self.teachers]
        return f"MotionTeacher(teachers={teacher_names}, strategy='{self.strategy}')"
