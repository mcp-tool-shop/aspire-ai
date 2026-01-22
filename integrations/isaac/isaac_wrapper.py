"""
Isaac Gym/Lab Environment Wrapper for ASPIRE.

Provides a unified interface for collecting trajectories from Isaac environments
and integrating with ASPIRE's adversarial training.

Requirements:
    pip install isaacgym  # or isaac-lab for newer versions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
import numpy as np

import torch
import torch.nn as nn


@dataclass
class StateActionPair:
    """A single state-action pair from a trajectory."""

    state: np.ndarray       # Robot state (joint positions, velocities, etc.)
    action: np.ndarray      # Action taken
    timestamp: float        # Time of this step

    # Optional privileged information
    collision: bool = False
    contact_forces: np.ndarray | None = None
    energy: float = 0.0
    goal_distance: float | None = None


@dataclass
class Trajectory:
    """A complete trajectory from an episode."""

    pairs: list[StateActionPair] = field(default_factory=list)

    # Episode metadata
    task_description: str = ""
    success: bool = False
    total_reward: float = 0.0

    # Goal information
    goal_state: np.ndarray | None = None
    initial_state: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.pairs)

    def to_tensors(
        self,
        device: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert to tensors for critic evaluation.

        Returns:
            states: (seq, state_dim)
            actions: (seq, action_dim)
            timestamps: (seq,)
        """
        states = torch.tensor(
            [p.state for p in self.pairs],
            dtype=torch.float32,
            device=device,
        )
        actions = torch.tensor(
            [p.action for p in self.pairs],
            dtype=torch.float32,
            device=device,
        )
        timestamps = torch.tensor(
            [p.timestamp for p in self.pairs],
            dtype=torch.float32,
            device=device,
        )
        return states, actions, timestamps

    def to_teacher_format(self):
        """Convert to format expected by motion teachers."""
        from .motion_teacher import TrajectoryData

        states = np.array([p.state for p in self.pairs])
        actions = np.array([p.action for p in self.pairs])
        timestamps = np.array([p.timestamp for p in self.pairs])

        collisions = np.array([p.collision for p in self.pairs])
        energy = np.array([p.energy for p in self.pairs])

        goal_distances = None
        if self.pairs[0].goal_distance is not None:
            goal_distances = np.array([p.goal_distance for p in self.pairs])

        forces = None
        if self.pairs[0].contact_forces is not None:
            forces = np.array([p.contact_forces for p in self.pairs])

        return TrajectoryData(
            states=states,
            actions=actions,
            timestamps=timestamps,
            goal_state=self.goal_state,
            initial_state=self.initial_state,
            task_description=self.task_description,
            collisions=collisions,
            forces=forces,
            energy_usage=energy,
            goal_distances=goal_distances,
        )


class TrajectoryBuffer:
    """
    Buffer for collecting and batching trajectories.

    Supports:
    - Adding trajectories from multiple parallel environments
    - Sampling batches for training
    - Prioritized sampling (higher-scoring trajectories more likely)
    """

    def __init__(
        self,
        max_size: int = 10000,
        prioritized: bool = True,
        priority_alpha: float = 0.6,
    ):
        self.max_size = max_size
        self.prioritized = prioritized
        self.priority_alpha = priority_alpha

        self.trajectories: list[Trajectory] = []
        self.scores: list[float] = []
        self.priorities: np.ndarray = np.array([])

    def add(self, trajectory: Trajectory, score: float | None = None):
        """Add a trajectory to the buffer."""
        self.trajectories.append(trajectory)
        self.scores.append(score if score is not None else 5.0)

        # Update priorities
        if self.prioritized:
            # Higher score = higher priority for learning
            priority = (abs(score - 5.0) + 1.0) ** self.priority_alpha if score else 1.0
            self.priorities = np.append(self.priorities, priority)

        # Remove oldest if over capacity
        while len(self.trajectories) > self.max_size:
            self.trajectories.pop(0)
            self.scores.pop(0)
            if self.prioritized:
                self.priorities = self.priorities[1:]

    def sample(
        self,
        batch_size: int,
        max_length: int | None = None,
    ) -> list[Trajectory]:
        """Sample a batch of trajectories."""
        if len(self.trajectories) == 0:
            return []

        batch_size = min(batch_size, len(self.trajectories))

        if self.prioritized and len(self.priorities) > 0:
            # Prioritized sampling
            probs = self.priorities / self.priorities.sum()
            indices = np.random.choice(
                len(self.trajectories),
                size=batch_size,
                replace=False,
                p=probs,
            )
        else:
            # Uniform sampling
            indices = np.random.choice(
                len(self.trajectories),
                size=batch_size,
                replace=False,
            )

        trajectories = [self.trajectories[i] for i in indices]

        # Optionally truncate
        if max_length is not None:
            trajectories = [
                self._truncate(t, max_length) for t in trajectories
            ]

        return trajectories

    def _truncate(self, trajectory: Trajectory, max_length: int) -> Trajectory:
        """Truncate trajectory to max length."""
        if len(trajectory) <= max_length:
            return trajectory

        truncated = Trajectory(
            pairs=trajectory.pairs[:max_length],
            task_description=trajectory.task_description,
            success=trajectory.success,
            total_reward=trajectory.total_reward,
            goal_state=trajectory.goal_state,
            initial_state=trajectory.initial_state,
        )
        return truncated

    def get_statistics(self) -> dict[str, float]:
        """Get buffer statistics."""
        if not self.scores:
            return {}

        return {
            "size": len(self.trajectories),
            "mean_score": np.mean(self.scores),
            "std_score": np.std(self.scores),
            "min_score": np.min(self.scores),
            "max_score": np.max(self.scores),
            "mean_length": np.mean([len(t) for t in self.trajectories]),
        }

    def clear(self):
        """Clear the buffer."""
        self.trajectories.clear()
        self.scores.clear()
        self.priorities = np.array([])

    def __len__(self) -> int:
        return len(self.trajectories)


@runtime_checkable
class IsaacEnvProtocol(Protocol):
    """Protocol that Isaac environments should implement."""

    num_envs: int
    observation_space: Any
    action_space: Any

    def reset(self) -> dict[str, torch.Tensor]:
        ...

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        ...


class AspireIsaacEnv:
    """
    Wrapper around Isaac Gym/Lab environments for ASPIRE training.

    Handles:
    - Trajectory collection from parallel environments
    - Privileged information extraction
    - Integration with motion teachers
    """

    def __init__(
        self,
        env: IsaacEnvProtocol,
        collect_privileged: bool = True,
        goal_key: str = "goal",
    ):
        """
        Args:
            env: Isaac Gym environment
            collect_privileged: Whether to collect privileged info (collisions, forces)
            goal_key: Key for goal in observation dict
        """
        self.env = env
        self.collect_privileged = collect_privileged
        self.goal_key = goal_key

        # Trajectory storage for each parallel environment
        self.current_trajectories: list[Trajectory] = [
            Trajectory() for _ in range(env.num_envs)
        ]
        self.episode_start_times: list[float] = [0.0] * env.num_envs
        self.step_count = 0

        # Extract dimensions
        self._extract_dimensions()

    def _extract_dimensions(self):
        """Extract state and action dimensions from environment."""
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        # Handle different observation space formats
        if hasattr(obs_space, "shape"):
            self.state_dim = obs_space.shape[0]
        elif isinstance(obs_space, dict) and "obs" in obs_space:
            self.state_dim = obs_space["obs"].shape[0]
        else:
            self.state_dim = 0  # Will be set on first observation

        if hasattr(act_space, "shape"):
            self.action_dim = act_space.shape[0]
        else:
            self.action_dim = 0

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset environment and trajectory collection."""
        obs = self.env.reset()

        # Reset trajectory storage
        for i in range(self.env.num_envs):
            if self.current_trajectories[i].pairs:
                # Save incomplete trajectory
                pass
            self.current_trajectories[i] = Trajectory()
            self.episode_start_times[i] = 0.0

        # Extract goal if available
        if isinstance(obs, dict) and self.goal_key in obs:
            goal = obs[self.goal_key].cpu().numpy()
            for i, traj in enumerate(self.current_trajectories):
                traj.goal_state = goal[i]

        # Set state dim if not known
        if self.state_dim == 0:
            if isinstance(obs, dict) and "obs" in obs:
                self.state_dim = obs["obs"].shape[-1]
            elif isinstance(obs, torch.Tensor):
                self.state_dim = obs.shape[-1]

        self.step_count = 0
        return obs

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict, list[Trajectory]]:
        """
        Step the environment and collect trajectory data.

        Returns:
            obs: Next observations
            rewards: Step rewards
            dones: Episode done flags
            infos: Additional info
            completed_trajectories: List of trajectories that just completed
        """
        # Step environment
        obs, rewards, dones, infos = self.env.step(actions)

        # Get current time
        current_time = self.step_count * 0.02  # Assume 50Hz, adjust as needed
        self.step_count += 1

        # Extract state
        if isinstance(obs, dict) and "obs" in obs:
            states = obs["obs"].cpu().numpy()
        elif isinstance(obs, torch.Tensor):
            states = obs.cpu().numpy()
        else:
            states = np.zeros((self.env.num_envs, self.state_dim))

        actions_np = actions.cpu().numpy()

        # Collect privileged information
        collisions = np.zeros(self.env.num_envs, dtype=bool)
        forces = None
        energies = np.zeros(self.env.num_envs)
        goal_distances = None

        if self.collect_privileged and infos:
            if "collision" in infos:
                collisions = infos["collision"].cpu().numpy().astype(bool)
            if "contact_forces" in infos:
                forces = infos["contact_forces"].cpu().numpy()
            if "energy" in infos:
                energies = infos["energy"].cpu().numpy()
            if "goal_distance" in infos:
                goal_distances = infos["goal_distance"].cpu().numpy()

        # Add to trajectories
        for i in range(self.env.num_envs):
            pair = StateActionPair(
                state=states[i],
                action=actions_np[i],
                timestamp=current_time - self.episode_start_times[i],
                collision=collisions[i],
                contact_forces=forces[i] if forces is not None else None,
                energy=energies[i],
                goal_distance=goal_distances[i] if goal_distances is not None else None,
            )
            self.current_trajectories[i].pairs.append(pair)

        # Collect completed trajectories
        completed = []
        dones_np = dones.cpu().numpy()

        for i in range(self.env.num_envs):
            if dones_np[i]:
                traj = self.current_trajectories[i]
                traj.total_reward = rewards[i].item() if rewards.dim() > 0 else rewards.item()

                # Check success (task-dependent)
                if "success" in infos:
                    traj.success = infos["success"][i].item()

                completed.append(traj)

                # Start new trajectory
                self.current_trajectories[i] = Trajectory()
                self.episode_start_times[i] = current_time

                # Set goal for new trajectory if available
                if isinstance(obs, dict) and self.goal_key in obs:
                    self.current_trajectories[i].goal_state = obs[self.goal_key][i].cpu().numpy()

        return obs, rewards, dones, infos, completed

    def get_current_trajectories(self) -> list[Trajectory]:
        """Get current in-progress trajectories."""
        return self.current_trajectories.copy()

    @property
    def num_envs(self) -> int:
        return self.env.num_envs


def create_isaac_env(
    env_name: str,
    num_envs: int = 512,
    device: str = "cuda",
    headless: bool = True,
) -> AspireIsaacEnv:
    """
    Create an Isaac Gym environment wrapped for ASPIRE.

    Args:
        env_name: Name of the environment (e.g., "FrankaCubeStack-v0")
        num_envs: Number of parallel environments
        device: Device to run on
        headless: Run without rendering

    Returns:
        Wrapped environment ready for ASPIRE training
    """
    try:
        # Try Isaac Lab (newer)
        from omni.isaac.lab.envs import ManagerBasedRLEnv
        from omni.isaac.lab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(env_name, num_envs=num_envs, use_gpu=device == "cuda")
        env = ManagerBasedRLEnv(cfg=env_cfg)

    except ImportError:
        try:
            # Fall back to Isaac Gym
            from isaacgym import gymapi
            from isaacgymenvs.utils.utils import set_seed
            from isaacgymenvs.tasks import isaacgym_task_map

            # Parse env name
            task_name = env_name.replace("-v0", "").replace("-", "")

            if task_name in isaacgym_task_map:
                env = isaacgym_task_map[task_name](
                    num_envs=num_envs,
                    headless=headless,
                    device=device,
                )
            else:
                raise ValueError(f"Unknown Isaac Gym task: {task_name}")

        except ImportError:
            raise ImportError(
                "Neither Isaac Lab nor Isaac Gym found. "
                "Install with: pip install isaacgym "
                "or follow NVIDIA Isaac Lab installation guide."
            )

    return AspireIsaacEnv(env)


class DummyIsaacEnv:
    """
    Dummy environment for testing without Isaac Gym installed.

    Simulates a simple reaching task.
    """

    def __init__(
        self,
        num_envs: int = 4,
        state_dim: int = 14,  # 7 joint positions + 7 velocities
        action_dim: int = 7,  # 7 joint torques
        episode_length: int = 100,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_length = episode_length
        self.device = device

        self.observation_space = type("Space", (), {"shape": (state_dim,)})()
        self.action_space = type("Space", (), {"shape": (action_dim,)})()

        self._step = 0
        self._states = None
        self._goals = None

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset to random initial states."""
        self._step = 0
        self._states = torch.randn(
            self.num_envs, self.state_dim,
            device=self.device,
        ) * 0.1
        self._goals = torch.randn(
            self.num_envs, self.state_dim // 2,
            device=self.device,
        ) * 0.5

        return {
            "obs": self._states,
            "goal": self._goals,
        }

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """Simulate a step."""
        self._step += 1

        # Simple dynamics: state changes based on action
        position = self._states[:, : self.state_dim // 2]
        velocity = self._states[:, self.state_dim // 2 :]

        # Simple physics
        velocity = velocity + actions * 0.1
        position = position + velocity * 0.02

        self._states = torch.cat([position, velocity], dim=-1)

        # Compute reward (negative distance to goal)
        goal_distance = torch.norm(position - self._goals, dim=-1)
        rewards = -goal_distance

        # Done after episode_length steps
        dones = torch.full(
            (self.num_envs,),
            self._step >= self.episode_length,
            device=self.device,
        )

        # Random collisions (10% chance)
        collisions = torch.rand(self.num_envs, device=self.device) < 0.1

        infos = {
            "goal_distance": goal_distance,
            "collision": collisions,
            "energy": torch.sum(actions ** 2, dim=-1),
            "success": goal_distance < 0.1,
        }

        return {"obs": self._states, "goal": self._goals}, rewards, dones, infos
