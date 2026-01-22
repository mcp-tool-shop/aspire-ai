"""
ASPIRE CLI - command line interface for training.
"""

import json
from multiprocessing import freeze_support
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine")
console = Console()


@app.command()
def train(
    config: Path = typer.Option(None, "--config", "-c", help="Path to config YAML"),
    prompts_file: Path = typer.Option(None, "--prompts", "-p", help="Path to prompts JSON"),
    output_dir: Path = typer.Option(Path("outputs"), "--output", "-o", help="Output directory"),
    teacher: str = typer.Option("claude", "--teacher", "-t", help="Teacher model to use"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs"),
):
    """Train a model using ASPIRE."""
    freeze_support()

    from aspire.config import AspireConfig
    from aspire.trainer import AspireTrainer

    # Load or create config
    if config and config.exists():
        cfg = AspireConfig.from_yaml(config)
    else:
        cfg = AspireConfig()

    # Override with CLI args
    cfg.training.output_dir = output_dir
    cfg.training.num_epochs = epochs
    cfg.teacher.default_teacher = teacher

    # Load prompts
    if prompts_file and prompts_file.exists():
        with open(prompts_file) as f:
            prompts = json.load(f)
    else:
        # Demo prompts
        prompts = [
            "Explain recursion in programming.",
            "What is the difference between a list and a tuple in Python?",
            "How does HTTP work?",
        ]
        console.print("[yellow]No prompts file provided, using demo prompts[/yellow]")

    # Train
    trainer = AspireTrainer(cfg)
    metrics = trainer.train(prompts)

    console.print("[bold green]Training complete![/bold green]")


@app.command()
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Path to checkpoint directory"),
    prompts_file: Path = typer.Option(..., "--prompts", "-p", help="Path to prompts JSON"),
    output: Path = typer.Option(None, "--output", "-o", help="Output results file"),
):
    """Evaluate a trained model."""
    freeze_support()

    import asyncio
    from aspire.config import AspireConfig
    from aspire.trainer import AspireTrainer

    # Load config from checkpoint
    cfg = AspireConfig.from_yaml(checkpoint / "config.yaml")

    # Load prompts
    with open(prompts_file) as f:
        prompts = json.load(f)

    # Create trainer and load checkpoint
    trainer = AspireTrainer(cfg)
    trainer.load_checkpoint(checkpoint)

    # Evaluate
    metrics = asyncio.run(trainer._evaluate(prompts))

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in metrics.items():
        table.add_row(key, f"{value:.4f}")

    console.print(table)

    # Save if requested
    if output:
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2)


@app.command()
def dialogue(
    prompt: str = typer.Argument(..., help="Prompt to generate dialogue for"),
    teacher: str = typer.Option("socratic", "--teacher", "-t", help="Teacher persona"),
    turns: int = typer.Option(3, "--turns", "-n", help="Number of dialogue turns"),
    model: str = typer.Option(
        "microsoft/Phi-3-mini-4k-instruct", "--model", "-m", help="Student model"
    ),
):
    """Generate a single adversarial dialogue."""
    freeze_support()

    import asyncio
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from aspire.teachers import get_teacher
    from aspire.dialogue import DialogueGenerator

    console.print(f"[bold]Generating dialogue with {teacher} teacher[/bold]\n")

    # Load student model
    console.print("Loading student model...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    student = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        torch_dtype="auto",
    )

    # Create teacher
    teacher_model = get_teacher(teacher)

    # Generate dialogue
    generator = DialogueGenerator(
        student_model=student,
        student_tokenizer=tokenizer,
        teacher=teacher_model,
        max_turns=turns,
    )

    dialogue = asyncio.run(generator.generate_dialogue(prompt))

    # Display
    console.print(f"\n[bold cyan]Prompt:[/bold cyan] {prompt}")
    console.print(f"\n[bold green]Initial Response:[/bold green]\n{dialogue.initial_response}")

    for turn in dialogue.history.turns:
        console.print(f"\n[bold yellow]Challenge ({turn.challenge.challenge_type.value}):[/bold yellow]")
        console.print(turn.challenge.content)
        console.print(f"\n[bold green]Response:[/bold green]")
        console.print(turn.student_response)

    console.print(f"\n[bold magenta]Final Score:[/bold magenta] {dialogue.final_evaluation.overall_score:.1f}/10")
    console.print(f"\n[bold]Reasoning:[/bold]\n{dialogue.final_evaluation.reasoning}")

    if dialogue.final_evaluation.improved_response:
        console.print(f"\n[bold blue]Improved Response:[/bold blue]")
        console.print(dialogue.final_evaluation.improved_response)


@app.command()
def teachers():
    """List available teacher personas."""
    from aspire.teachers.registry import TeacherRegistry

    table = Table(title="Available Teachers")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")

    teacher_info = {
        "claude": "Default Claude-based teacher",
        "openai": "GPT-4 based teacher",
        "socratic": "Teaches through questions, never gives answers directly",
        "scientific": "Demands evidence, rigor, and falsifiable claims",
        "creative": "Encourages novel thinking and unconventional approaches",
        "adversarial": "Stress-tests reasoning through devil's advocacy",
        "compassionate": "Balances challenge with encouragement",
    }

    for name in TeacherRegistry.list():
        desc = teacher_info.get(name, "Custom teacher")
        table.add_row(name, desc)

    console.print(table)


@app.command()
def init(
    output: Path = typer.Option(Path("aspire-config.yaml"), "--output", "-o", help="Output path"),
):
    """Create a default configuration file."""
    from aspire.config import AspireConfig

    cfg = AspireConfig()
    cfg.to_yaml(output)
    console.print(f"[green]Created config file: {output}[/green]")


if __name__ == "__main__":
    app()
