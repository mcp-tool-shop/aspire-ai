"""
ASPIRE Trainer - the core training loop.

Orchestrates the training of student and critic models through
adversarial dialogue with teacher models.
"""

import asyncio
import os
from pathlib import Path
from typing import Any
from multiprocessing import freeze_support

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from aspire.config import AspireConfig
from aspire.critic import CriticHead, SeparateCritic, SharedEncoderCritic
from aspire.losses import AspireLoss
from aspire.teachers import get_teacher, BaseTeacher
from aspire.dialogue import DialogueGenerator, DialogueManager, DialogueFormatter

# Windows compatibility
os.environ["XFORMERS_DISABLED"] = "1"

console = Console()


class AspireDataset(Dataset):
    """Dataset for ASPIRE training."""

    def __init__(
        self,
        prompts: list[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prompt = self.prompts[idx]

        # Tokenize
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "prompt": prompt,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


class AspireTrainer:
    """
    Main trainer for ASPIRE.

    Coordinates:
    1. Student model training
    2. Critic model training
    3. Dialogue generation with teachers
    4. Loss computation and backpropagation
    """

    def __init__(self, config: AspireConfig):
        self.config = config
        self.device = config.device

        # Set seed
        torch.manual_seed(config.seed)

        # Initialize components
        self._init_student()
        self._init_critic()
        self._init_teacher()
        self._init_loss()
        self._init_optimizers()

        # Dialogue components
        self.dialogue_generator = DialogueGenerator(
            student_model=self.student_model,
            student_tokenizer=self.tokenizer,
            teacher=self.teacher,
            max_turns=config.teacher.max_dialogue_turns,
            device=self.device,
        )
        self.dialogue_manager = DialogueManager(
            generator=self.dialogue_generator,
            cache_dir=config.training.output_dir / "dialogue_cache",
        )
        self.dialogue_formatter = DialogueFormatter(format_type="chat")

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        console.print("[green]ASPIRE Trainer initialized[/green]")

    def _init_student(self) -> None:
        """Initialize student model with optional LoRA."""
        cfg = self.config.student

        console.print(f"Loading student model: {cfg.model_name_or_path}")

        # Quantization config
        quantization_config = None
        if cfg.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif cfg.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        self.student_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare for training
        if cfg.load_in_4bit or cfg.load_in_8bit:
            self.student_model = prepare_model_for_kbit_training(self.student_model)

        # Apply LoRA if configured
        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.student_model = get_peft_model(self.student_model, lora_config)
            console.print("[blue]LoRA applied to student model[/blue]")

        # Gradient checkpointing
        if cfg.use_gradient_checkpointing:
            self.student_model.gradient_checkpointing_enable()

        # Get hidden size for critic
        self.student_hidden_size = self.student_model.config.hidden_size

    def _init_critic(self) -> None:
        """Initialize critic model."""
        cfg = self.config.critic

        console.print(f"Initializing critic: {cfg.architecture}")

        if cfg.architecture == "head":
            self.critic = CriticHead(
                input_dim=self.student_hidden_size,
                hidden_dim=cfg.head_hidden_dim,
                num_layers=cfg.head_num_layers,
                reasoning_dim=cfg.reasoning_embedding_dim,
            )
        elif cfg.architecture == "separate":
            self.critic = SeparateCritic(
                model_name_or_path=cfg.separate_model_name,
                hidden_dim=cfg.head_hidden_dim,
                reasoning_dim=cfg.reasoning_embedding_dim,
                load_in_4bit=cfg.separate_load_in_4bit,
            )
        elif cfg.architecture == "shared_encoder":
            self.critic = SharedEncoderCritic(
                student_model=self.student_model,
                hidden_dim=cfg.head_hidden_dim,
                reasoning_dim=cfg.reasoning_embedding_dim,
            )
        else:
            raise ValueError(f"Unknown critic architecture: {cfg.architecture}")

        self.critic = self.critic.to(self.device)

    def _init_teacher(self) -> None:
        """Initialize teacher model(s)."""
        cfg = self.config.teacher

        console.print(f"Initializing teacher: {cfg.default_teacher}")

        self.teacher = get_teacher(
            cfg.default_teacher,
            model=cfg.claude_model if cfg.default_teacher == "claude" else cfg.openai_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    def _init_loss(self) -> None:
        """Initialize loss functions."""
        cfg = self.config.loss

        self.loss_fn = AspireLoss(
            critic_score_weight=cfg.critic_score_weight,
            critic_reasoning_weight=cfg.critic_reasoning_weight,
            student_reward_weight=cfg.student_reward_weight,
            student_contrastive_weight=cfg.student_contrastive_weight,
            student_trajectory_weight=cfg.student_trajectory_weight,
            student_coherence_weight=cfg.student_coherence_weight,
            contrastive_margin=cfg.contrastive_margin,
            contrastive_temperature=cfg.contrastive_temperature,
        )

    def _init_optimizers(self) -> None:
        """Initialize optimizers and schedulers."""
        cfg = self.config.training

        # Student optimizer
        student_params = [p for p in self.student_model.parameters() if p.requires_grad]

        if cfg.optimizer == "adamw":
            self.student_optimizer = AdamW(
                student_params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer in ["adamw_8bit", "paged_adamw_8bit"]:
            import bitsandbytes as bnb

            self.student_optimizer = bnb.optim.AdamW8bit(
                student_params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )

        # Critic optimizer
        critic_params = self.critic.get_trainable_parameters()
        self.critic_optimizer = AdamW(
            critic_params,
            lr=cfg.critic_learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def train(
        self,
        train_prompts: list[str],
        eval_prompts: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Main training loop.

        Args:
            train_prompts: List of training prompts
            eval_prompts: Optional list of evaluation prompts

        Returns:
            Training metrics
        """
        cfg = self.config.training

        # Create dataset and dataloader
        train_dataset = AspireDataset(
            prompts=train_prompts,
            tokenizer=self.tokenizer,
            max_length=self.config.student.max_length,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader_num_workers,  # 0 for Windows
        )

        # Calculate total steps
        num_update_steps = len(train_dataloader) // cfg.gradient_accumulation_steps
        total_steps = num_update_steps * cfg.num_epochs

        # Learning rate schedulers
        self.student_scheduler = get_scheduler(
            cfg.lr_scheduler,
            optimizer=self.student_optimizer,
            num_warmup_steps=int(total_steps * cfg.warmup_ratio),
            num_training_steps=total_steps,
        )

        self.critic_scheduler = get_scheduler(
            cfg.lr_scheduler,
            optimizer=self.critic_optimizer,
            num_warmup_steps=int(total_steps * cfg.warmup_ratio),
            num_training_steps=total_steps,
        )

        # Training loop
        console.print(f"\n[bold green]Starting ASPIRE training[/bold green]")
        console.print(f"  Epochs: {cfg.num_epochs}")
        console.print(f"  Train samples: {len(train_prompts)}")
        console.print(f"  Batch size: {cfg.batch_size}")
        console.print(f"  Total steps: {total_steps}")

        metrics = {"train_loss": [], "critic_loss": [], "student_loss": []}

        for epoch in range(cfg.num_epochs):
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch(train_dataloader)
            metrics["train_loss"].append(epoch_metrics["loss"])
            metrics["critic_loss"].append(epoch_metrics["critic_loss"])
            metrics["student_loss"].append(epoch_metrics["student_loss"])

            console.print(
                f"Epoch {epoch + 1}/{cfg.num_epochs} - "
                f"Loss: {epoch_metrics['loss']:.4f} - "
                f"Critic: {epoch_metrics['critic_loss']:.4f} - "
                f"Student: {epoch_metrics['student_loss']:.4f}"
            )

            # Evaluation
            if eval_prompts:
                eval_metrics = asyncio.run(self._evaluate(eval_prompts))
                console.print(f"  Eval score: {eval_metrics['avg_score']:.2f}")

            # Save checkpoint
            if (epoch + 1) % 1 == 0:  # Save every epoch
                self._save_checkpoint(epoch + 1)

        return metrics

    def _train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.student_model.train()
        self.critic.train()

        total_loss = 0.0
        total_critic_loss = 0.0
        total_student_loss = 0.0
        num_batches = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task(
                f"Epoch {self.current_epoch + 1}", total=len(dataloader)
            )

            for batch_idx, batch in enumerate(dataloader):
                # Generate dialogue for batch (async)
                prompts = batch["prompt"]
                dialogues = asyncio.run(
                    self.dialogue_manager.get_dialogues(prompts, max_concurrent=3)
                )

                # Compute losses
                losses = self._compute_batch_loss(batch, dialogues)

                # Backward pass
                loss = losses["total"]
                loss = loss / self.config.training.gradient_accumulation_steps
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.training.max_grad_norm,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        self.config.training.max_grad_norm,
                    )

                    # Optimizer step
                    self.student_optimizer.step()
                    self.critic_optimizer.step()
                    self.student_scheduler.step()
                    self.critic_scheduler.step()

                    self.student_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    self.global_step += 1

                # Track metrics
                total_loss += losses["total"].item()
                total_critic_loss += losses.get("critic_total", torch.tensor(0.0)).item()
                total_student_loss += losses.get("student_total", torch.tensor(0.0)).item()
                num_batches += 1

                progress.update(task, advance=1)

        return {
            "loss": total_loss / num_batches,
            "critic_loss": total_critic_loss / num_batches,
            "student_loss": total_student_loss / num_batches,
        }

    def _compute_batch_loss(
        self,
        batch: dict[str, torch.Tensor],
        dialogues: list,
    ) -> dict[str, torch.Tensor]:
        """Compute loss for a batch."""

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Get student outputs
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        student_hidden = student_outputs.hidden_states[-1]

        # Get critic predictions
        critic_output = self.critic(hidden_states=student_hidden, attention_mask=attention_mask)

        # Get teacher scores from dialogues
        teacher_scores = torch.tensor(
            [d.final_evaluation.overall_score for d in dialogues],
            device=self.device,
            dtype=torch.float32,
        )

        # Compute losses
        losses = self.loss_fn(
            critic_predicted_score=critic_output.score,
            teacher_score=teacher_scores,
            critic_predicted_embedding=critic_output.reasoning_embedding,
        )

        return losses

    async def _evaluate(self, prompts: list[str]) -> dict[str, float]:
        """Evaluate on a set of prompts."""
        self.student_model.eval()
        self.critic.eval()

        scores = []

        for prompt in prompts:
            dialogue = await self.dialogue_manager.get_dialogue(prompt)
            scores.append(dialogue.final_evaluation.overall_score)

        return {
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
        }

    def _save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        output_dir = self.config.training.output_dir / f"checkpoint-{epoch}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save student model
        self.student_model.save_pretrained(output_dir / "student")
        self.tokenizer.save_pretrained(output_dir / "student")

        # Save critic
        self.critic.save(str(output_dir / "critic.pt"))

        # Save config
        self.config.to_yaml(output_dir / "config.yaml")

        console.print(f"[green]Checkpoint saved to {output_dir}[/green]")

    def load_checkpoint(self, checkpoint_dir: Path) -> None:
        """Load from checkpoint."""
        # Load student
        from peft import PeftModel

        self.student_model = PeftModel.from_pretrained(
            self.student_model,
            checkpoint_dir / "student",
        )

        # Load critic
        self.critic = self.critic.__class__.load(str(checkpoint_dir / "critic.pt"))
        self.critic = self.critic.to(self.device)

        console.print(f"[green]Loaded checkpoint from {checkpoint_dir}[/green]")


def main():
    """Main entry point."""
    # Windows multiprocessing fix
    freeze_support()

    # Example usage
    config = AspireConfig()
    trainer = AspireTrainer(config)

    # Example prompts (in practice, load from dataset)
    train_prompts = [
        "Explain the concept of recursion in programming.",
        "What are the trade-offs between SQL and NoSQL databases?",
        "How does gradient descent work in machine learning?",
    ]

    trainer.train(train_prompts)


if __name__ == "__main__":
    main()
