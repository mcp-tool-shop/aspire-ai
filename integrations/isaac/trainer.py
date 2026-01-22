"""
ASPIRE Isaac Trainer - Training loop for embodied AI with internalized judgment.

The training loop:
1. Collect trajectories from parallel Isaac environments
2. Have motion teachers critique each trajectory
3. Train the critic to predict teacher judgments
4. Train the policy to maximize critic scores + teacher guidance

After training, the policy self-refines using the internalized critic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .config import IsaacAspireConfig, CriticConfig
from .motion_teacher import MotionTeacher, MotionCritique, TrajectoryData
from .trajectory_critic import TrajectoryCritic, TrajectoryCriticLoss, CriticOutput
from .isaac_wrapper import AspireIsaacEnv, TrajectoryBuffer, Trajectory


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""

    epoch: int
    critic_loss: float
    policy_loss: float
    mean_trajectory_score: float
    mean_episode_reward: float
    success_rate: float
    trajectories_collected: int
    time_elapsed: float


class TrajectoryDataset(Dataset):
    """Dataset of trajectories for critic training."""

    def __init__(
        self,
        trajectories: list[Trajectory],
        critiques: list[MotionCritique],
        max_length: int = 256,
    ):
        self.trajectories = trajectories
        self.critiques = critiques
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        critique = self.critiques[idx]

        states, actions, _ = traj.to_tensors(device="cpu")

        # Pad or truncate
        seq_len = len(traj)
        if seq_len > self.max_length:
            states = states[: self.max_length]
            actions = actions[: self.max_length]
            seq_len = self.max_length
        elif seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            states = torch.cat(
                [states, torch.zeros(pad_len, states.shape[-1])], dim=0
            )
            actions = torch.cat(
                [actions, torch.zeros(pad_len, actions.shape[-1])], dim=0
            )

        # Create mask
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:seq_len] = True

        return {
            "states": states,
            "actions": actions,
            "mask": mask,
            "score": torch.tensor(critique.overall_score, dtype=torch.float32),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }


class SimplePolicy(nn.Module):
    """
    Simple MLP policy for demonstration.

    In practice, you'd use your own policy architecture or integrate
    with existing RL frameworks (cleanrl, stable-baselines3, rl_games).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = []
        in_dim = state_dim

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Mean and log_std for Gaussian policy
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get action distribution parameters.

        Returns:
            mean: (batch, action_dim)
            std: (batch, action_dim)
        """
        features = self.backbone(state)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample an action."""
        mean, std = self.forward(state)

        if deterministic:
            return mean

        noise = torch.randn_like(mean)
        action = mean + std * noise
        return torch.tanh(action)  # Bound to [-1, 1]

    def log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of action."""
        mean, std = self.forward(state)

        # Gaussian log prob
        var = std ** 2
        log_prob = -0.5 * (
            ((action - mean) ** 2) / var
            + torch.log(var)
            + np.log(2 * np.pi)
        )
        return log_prob.sum(dim=-1)


class AspireIsaacTrainer:
    """
    ASPIRE training loop for Isaac Gym environments.

    Combines:
    - Trajectory collection from parallel environments
    - Motion teacher critiques
    - Critic training to internalize teacher judgment
    - Policy training guided by critic + teacher
    """

    def __init__(
        self,
        env: AspireIsaacEnv | str,
        config: IsaacAspireConfig | None = None,
        policy: nn.Module | None = None,
        teacher: MotionTeacher | None = None,
        critic: TrajectoryCritic | None = None,
    ):
        if config is None:
            config = IsaacAspireConfig()

        self.config = config
        self.device = config.device

        # Create or wrap environment
        if isinstance(env, str):
            from .isaac_wrapper import create_isaac_env
            self.env = create_isaac_env(
                env,
                num_envs=config.training.num_envs,
                device=config.device,
            )
        else:
            self.env = env

        # Initialize state/action dims from environment
        config.critic.state_dim = self.env.state_dim
        config.critic.action_dim = self.env.action_dim

        # Create components
        self.teacher = teacher or MotionTeacher(
            personas=config.teacher.personas,
            strategy=config.teacher.strategy,
        )

        self.critic = critic or TrajectoryCritic(config.critic)
        self.critic = self.critic.to(self.device)

        self.policy = policy or SimplePolicy(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
        )
        self.policy = self.policy.to(self.device)

        # Optimizers
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(),
            lr=config.training.critic_lr,
        )
        self.policy_optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.training.policy_lr,
        )

        # Loss functions
        self.critic_loss_fn = TrajectoryCriticLoss()

        # Trajectory buffer
        self.buffer = TrajectoryBuffer(
            max_size=config.training.trajectory_batch_size * 100,
        )

        # Metrics tracking
        self.metrics_history: list[TrainingMetrics] = []

        # Logging
        self.use_wandb = config.training.use_wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project=config.training.wandb_project)
            except ImportError:
                print("wandb not installed, disabling")
                self.use_wandb = False

    def collect_trajectories(
        self,
        num_episodes: int,
    ) -> tuple[list[Trajectory], list[MotionCritique]]:
        """
        Collect trajectories and get teacher critiques.

        Returns:
            trajectories: List of collected trajectories
            critiques: Teacher critiques for each trajectory
        """
        trajectories = []
        critiques = []

        obs = self.env.reset()
        episodes_collected = 0

        while episodes_collected < num_episodes:
            # Get current state
            if isinstance(obs, dict) and "obs" in obs:
                state = obs["obs"]
            else:
                state = obs

            # Get action from policy
            with torch.no_grad():
                action = self.policy.get_action(state)

            # Step environment
            obs, rewards, dones, infos, completed = self.env.step(action)

            # Process completed trajectories
            for traj in completed:
                if len(traj) > 10:  # Skip very short episodes
                    # Get teacher critique
                    traj_data = traj.to_teacher_format()
                    critique = self.teacher.critique(traj_data)

                    trajectories.append(traj)
                    critiques.append(critique)

                    # Add to buffer
                    self.buffer.add(traj, critique.overall_score)

                    episodes_collected += 1

                    if episodes_collected >= num_episodes:
                        break

        return trajectories, critiques

    def train_critic_epoch(
        self,
        trajectories: list[Trajectory],
        critiques: list[MotionCritique],
    ) -> float:
        """
        Train critic for one epoch on collected data.

        Returns:
            Average critic loss
        """
        if not trajectories:
            return 0.0

        dataset = TrajectoryDataset(
            trajectories,
            critiques,
            max_length=self.config.critic.max_trajectory_len,
        )

        # Simple batching (Windows-compatible, no multiprocessing)
        batch_size = min(32, len(dataset))
        indices = torch.randperm(len(dataset))

        total_loss = 0.0
        num_batches = 0

        self.critic.train()

        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i : i + batch_size]

            # Collate batch
            batch = [dataset[j] for j in batch_indices]
            states = torch.stack([b["states"] for b in batch]).to(self.device)
            actions = torch.stack([b["actions"] for b in batch]).to(self.device)
            masks = torch.stack([b["mask"] for b in batch]).to(self.device)
            scores = torch.stack([b["score"] for b in batch]).to(self.device)

            # Forward pass
            self.critic_optimizer.zero_grad()
            output = self.critic(states, actions, masks)

            # Compute loss
            losses = self.critic_loss_fn(output, scores)
            loss = losses["total"]

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train_policy_epoch(
        self,
        trajectories: list[Trajectory],
        critiques: list[MotionCritique],
    ) -> float:
        """
        Train policy using critic guidance.

        Uses a simplified policy gradient with critic as reward signal.

        Returns:
            Average policy loss
        """
        if not trajectories:
            return 0.0

        self.policy.train()
        self.critic.eval()

        total_loss = 0.0
        num_updates = 0

        for traj, critique in zip(trajectories, critiques):
            if len(traj) < 10:
                continue

            states, actions, _ = traj.to_tensors(device=self.device)
            states = states.unsqueeze(0)  # (1, seq, state_dim)
            actions = actions.unsqueeze(0)  # (1, seq, action_dim)

            # Get critic score
            with torch.no_grad():
                critic_output = self.critic(states, actions)
                critic_score = critic_output.score  # (1,)

            # Policy gradient loss
            # Higher critic score = lower loss
            self.policy_optimizer.zero_grad()

            # Get log probs for taken actions
            log_probs = []
            for t in range(states.shape[1]):
                state_t = states[0, t]
                action_t = actions[0, t]
                log_prob = self.policy.log_prob(state_t.unsqueeze(0), action_t.unsqueeze(0))
                log_probs.append(log_prob)

            log_probs = torch.cat(log_probs)

            # Use normalized critic score as advantage
            advantage = (critic_score - 5.0) / 5.0  # Normalize to [-1, 1]

            # Policy gradient with baseline
            policy_loss = -(log_probs * advantage).mean()

            # Add entropy bonus for exploration
            _, std = self.policy.forward(states[0])
            entropy = 0.5 * torch.log(2 * np.pi * np.e * std ** 2).mean()
            policy_loss -= self.config.training.entropy_bonus * entropy

            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()

            total_loss += policy_loss.item()
            num_updates += 1

        return total_loss / max(num_updates, 1)

    def train(
        self,
        epochs: int | None = None,
        callback: Callable[[TrainingMetrics], None] | None = None,
    ) -> list[TrainingMetrics]:
        """
        Main training loop.

        Args:
            epochs: Number of epochs (uses config if None)
            callback: Called after each epoch with metrics

        Returns:
            List of training metrics for each epoch
        """
        epochs = epochs or self.config.training.epochs

        print(f"Starting ASPIRE Isaac training for {epochs} epochs")
        print(f"  Parallel environments: {self.env.num_envs}")
        print(f"  Teacher: {self.teacher}")
        print(f"  Critic architecture: {self.config.critic.architecture.value}")
        print()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Collect trajectories
            print(f"Epoch {epoch + 1}/{epochs}: Collecting trajectories...")
            trajectories, critiques = self.collect_trajectories(
                num_episodes=self.config.training.episodes_per_epoch,
            )

            if not trajectories:
                print("  No trajectories collected, skipping epoch")
                continue

            # Compute statistics
            scores = [c.overall_score for c in critiques]
            rewards = [t.total_reward for t in trajectories]
            successes = [t.success for t in trajectories]

            # Train critic
            if epoch % self.config.training.critic_update_frequency == 0:
                print(f"  Training critic...")
                critic_loss = self.train_critic_epoch(trajectories, critiques)
            else:
                critic_loss = 0.0

            # Train policy
            if epoch % self.config.training.policy_update_frequency == 0:
                print(f"  Training policy...")
                policy_loss = self.train_policy_epoch(trajectories, critiques)
            else:
                policy_loss = 0.0

            # Metrics
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                critic_loss=critic_loss,
                policy_loss=policy_loss,
                mean_trajectory_score=np.mean(scores),
                mean_episode_reward=np.mean(rewards),
                success_rate=np.mean(successes) if successes else 0.0,
                trajectories_collected=len(trajectories),
                time_elapsed=time.time() - epoch_start,
            )
            self.metrics_history.append(metrics)

            # Log
            print(f"  Score: {metrics.mean_trajectory_score:.2f}/10")
            print(f"  Reward: {metrics.mean_episode_reward:.2f}")
            print(f"  Success: {metrics.success_rate:.1%}")
            print(f"  Critic loss: {critic_loss:.4f}")
            print(f"  Policy loss: {policy_loss:.4f}")
            print(f"  Time: {metrics.time_elapsed:.1f}s")
            print()

            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "critic_loss": critic_loss,
                    "policy_loss": policy_loss,
                    "mean_score": metrics.mean_trajectory_score,
                    "mean_reward": metrics.mean_episode_reward,
                    "success_rate": metrics.success_rate,
                })

            # Callback
            if callback is not None:
                callback(metrics)

            # Checkpoint
            if (epoch + 1) % self.config.training.save_frequency == 0:
                self.save_checkpoint(epoch + 1)

        print("Training complete!")
        return self.metrics_history

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "critic_state_dict": self.critic.state_dict(),
                "policy_state_dict": self.policy.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "config": self.config,
                "metrics_history": self.metrics_history,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.metrics_history = checkpoint.get("metrics_history", [])

        print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")

    def evaluate(
        self,
        num_episodes: int = 100,
        use_critic_refinement: bool = True,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """
        Evaluate the trained policy.

        Args:
            num_episodes: Number of evaluation episodes
            use_critic_refinement: Use critic to refine actions
            deterministic: Use deterministic policy

        Returns:
            Evaluation metrics
        """
        self.policy.eval()
        self.critic.eval()

        all_rewards = []
        all_successes = []
        all_scores = []

        obs = self.env.reset()
        episodes_done = 0

        while episodes_done < num_episodes:
            if isinstance(obs, dict) and "obs" in obs:
                state = obs["obs"]
            else:
                state = obs

            with torch.no_grad():
                action = self.policy.get_action(state, deterministic=deterministic)

                # Optional: refine action using critic
                if use_critic_refinement:
                    # Get critic's suggested improvement
                    # (simplified: just evaluate, real implementation would iterate)
                    pass

            obs, rewards, dones, infos, completed = self.env.step(action)

            for traj in completed:
                if len(traj) > 0:
                    traj_data = traj.to_teacher_format()
                    critique = self.teacher.critique(traj_data)

                    all_rewards.append(traj.total_reward)
                    all_successes.append(traj.success)
                    all_scores.append(critique.overall_score)

                    episodes_done += 1
                    if episodes_done >= num_episodes:
                        break

        return {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "success_rate": np.mean(all_successes),
            "mean_score": np.mean(all_scores),
            "std_score": np.std(all_scores),
        }
