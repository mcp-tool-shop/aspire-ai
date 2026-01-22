"""
ASPIRE Training Script for Forge.

Train LoRA adapters using ASPIRE methodology:
1. Generate images with current model
2. Teacher (vision-language model) critiques them
3. Critic learns to predict teacher's judgment
4. Model learns from critic's internalized standards
"""

import os
import json
from pathlib import Path
from datetime import datetime

import gradio as gr
import torch

from modules import scripts, shared, sd_models
from modules.ui_components import FormRow


class AspireTrainScript(scripts.Script):
    """
    ASPIRE Training Interface.

    Train aesthetic judgment into image generation models through
    adversarial dialogue with vision-language teachers.
    """

    def title(self):
        return "ASPIRE Training"

    def show(self, is_img2img):
        # Only show in txt2img tab
        return not is_img2img

    def ui(self, is_img2img):
        with gr.Blocks():
            gr.Markdown("""
            # ASPIRE Training

            **Adversarial Student-Professor Internalized Reasoning Engine**

            Train your model to develop aesthetic judgment through dialogue with AI art critics.
            """)

            with gr.Tabs():
                # Tab 1: Training Setup
                with gr.TabItem("Training Setup"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Dataset
                            gr.Markdown("### Dataset")
                            prompts_file = gr.File(
                                label="Prompts File (JSON or TXT)",
                                file_types=[".json", ".txt"],
                            )
                            prompts_text = gr.Textbox(
                                label="Or enter prompts directly (one per line)",
                                lines=5,
                                placeholder="a beautiful sunset over mountains\na portrait of a wise old man\n...",
                            )

                        with gr.Column(scale=1):
                            # Teacher Selection
                            gr.Markdown("### Teacher Model")
                            teacher_type = gr.Dropdown(
                                label="Teacher",
                                choices=[
                                    "Claude (Vision)",
                                    "GPT-4V",
                                    "Local VLM",
                                    "Composite (Multiple)",
                                ],
                                value="Claude (Vision)",
                            )

                            teacher_persona = gr.Dropdown(
                                label="Teaching Style",
                                choices=[
                                    "Balanced Critic",
                                    "Technical Analyst",
                                    "Artistic Visionary",
                                    "Composition Expert",
                                    "Color Theorist",
                                    "Harsh Critic",
                                    "Encouraging Mentor",
                                ],
                                value="Balanced Critic",
                            )

                            api_key_status = gr.Markdown(
                                "⚠️ Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` environment variable"
                            )

                    with gr.Row():
                        with gr.Column(scale=1):
                            # Training Parameters
                            gr.Markdown("### Training Parameters")

                            num_epochs = gr.Slider(
                                label="Epochs",
                                minimum=1,
                                maximum=20,
                                value=3,
                                step=1,
                            )

                            batch_size = gr.Slider(
                                label="Batch Size",
                                minimum=1,
                                maximum=8,
                                value=2,
                                step=1,
                            )

                            learning_rate = gr.Number(
                                label="Learning Rate",
                                value=1e-4,
                            )

                            dialogue_turns = gr.Slider(
                                label="Dialogue Turns per Image",
                                minimum=1,
                                maximum=5,
                                value=2,
                                step=1,
                            )

                        with gr.Column(scale=1):
                            # LoRA Settings
                            gr.Markdown("### LoRA Configuration")

                            lora_rank = gr.Slider(
                                label="LoRA Rank",
                                minimum=1,
                                maximum=128,
                                value=16,
                                step=1,
                            )

                            lora_alpha = gr.Slider(
                                label="LoRA Alpha",
                                minimum=1,
                                maximum=128,
                                value=32,
                                step=1,
                            )

                            target_modules = gr.CheckboxGroup(
                                label="Target Modules",
                                choices=[
                                    "q_proj", "k_proj", "v_proj", "o_proj",
                                    "to_q", "to_k", "to_v", "to_out",
                                ],
                                value=["to_q", "to_k", "to_v", "to_out"],
                            )

                    with gr.Row():
                        # Output Settings
                        gr.Markdown("### Output")
                        output_name = gr.Textbox(
                            label="Output LoRA Name",
                            value=f"aspire-{datetime.now().strftime('%Y%m%d-%H%M')}",
                        )

                        save_every = gr.Slider(
                            label="Save Checkpoint Every N Steps",
                            minimum=50,
                            maximum=1000,
                            value=200,
                            step=50,
                        )

                # Tab 2: Critic Configuration
                with gr.TabItem("Critic Model"):
                    gr.Markdown("""
                    ### Critic Architecture

                    The critic learns to predict what the teacher would think about generated images.
                    After training, it provides aesthetic guidance without API calls.
                    """)

                    critic_arch = gr.Radio(
                        label="Critic Architecture",
                        choices=[
                            "CLIP Head (Lightweight)",
                            "Separate Encoder (More Capable)",
                            "Shared with UNet (Balanced)",
                        ],
                        value="CLIP Head (Lightweight)",
                    )

                    with gr.Row():
                        critic_hidden = gr.Slider(
                            label="Hidden Dimension",
                            minimum=128,
                            maximum=1024,
                            value=512,
                            step=64,
                        )

                        critic_layers = gr.Slider(
                            label="Number of Layers",
                            minimum=1,
                            maximum=6,
                            value=2,
                            step=1,
                        )

                    gr.Markdown("### Evaluation Dimensions")

                    eval_dimensions = gr.CheckboxGroup(
                        label="What should the critic evaluate?",
                        choices=[
                            "Overall Aesthetic Quality",
                            "Composition & Balance",
                            "Color Harmony",
                            "Lighting & Contrast",
                            "Style Consistency",
                            "Prompt Adherence",
                            "Technical Quality",
                            "Emotional Impact",
                        ],
                        value=[
                            "Overall Aesthetic Quality",
                            "Composition & Balance",
                            "Style Consistency",
                            "Prompt Adherence",
                        ],
                    )

                # Tab 3: Live Training
                with gr.TabItem("Training Progress"):
                    with gr.Row():
                        start_btn = gr.Button("▶ Start Training", variant="primary")
                        pause_btn = gr.Button("⏸ Pause")
                        stop_btn = gr.Button("⏹ Stop", variant="stop")

                    progress_bar = gr.Progress()

                    with gr.Row():
                        with gr.Column(scale=2):
                            # Training metrics
                            metrics_plot = gr.Plot(label="Training Metrics")

                            log_output = gr.Textbox(
                                label="Training Log",
                                lines=10,
                                max_lines=20,
                                interactive=False,
                            )

                        with gr.Column(scale=1):
                            # Live preview
                            gr.Markdown("### Live Preview")
                            preview_image = gr.Image(
                                label="Current Generation",
                                type="pil",
                            )
                            preview_score = gr.Number(
                                label="Critic Score",
                                precision=2,
                            )
                            preview_feedback = gr.Textbox(
                                label="Teacher Feedback",
                                lines=3,
                            )

                # Tab 4: Evaluation
                with gr.TabItem("Evaluation"):
                    gr.Markdown("""
                    ### Evaluate Trained Model

                    Compare generations before and after ASPIRE training.
                    """)

                    eval_prompt = gr.Textbox(
                        label="Test Prompt",
                        value="a serene Japanese garden at sunset, with a koi pond and cherry blossoms",
                    )

                    with gr.Row():
                        eval_btn = gr.Button("Generate Comparison", variant="primary")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Before ASPIRE**")
                            before_image = gr.Image(label="Original Model")
                            before_score = gr.Number(label="Critic Score")

                        with gr.Column():
                            gr.Markdown("**After ASPIRE**")
                            after_image = gr.Image(label="ASPIRE-Trained Model")
                            after_score = gr.Number(label="Critic Score")

                    improvement_text = gr.Markdown("")

        # Return all components that need to be passed to run()
        return [
            prompts_file,
            prompts_text,
            teacher_type,
            teacher_persona,
            num_epochs,
            batch_size,
            learning_rate,
            dialogue_turns,
            lora_rank,
            lora_alpha,
            target_modules,
            output_name,
            save_every,
            critic_arch,
            critic_hidden,
            critic_layers,
            eval_dimensions,
        ]

    def run(self, p, *args):
        """
        Main training entry point.

        This is called when the user clicks generate with this script selected.
        For ASPIRE, we override the normal generation flow with training.
        """
        (
            prompts_file,
            prompts_text,
            teacher_type,
            teacher_persona,
            num_epochs,
            batch_size,
            learning_rate,
            dialogue_turns,
            lora_rank,
            lora_alpha,
            target_modules,
            output_name,
            save_every,
            critic_arch,
            critic_hidden,
            critic_layers,
            eval_dimensions,
        ) = args

        # This would kick off the actual training
        # For now, return a message indicating setup is complete
        print(f"ASPIRE Training would start with:")
        print(f"  Teacher: {teacher_type} ({teacher_persona})")
        print(f"  Epochs: {num_epochs}, Batch: {batch_size}, LR: {learning_rate}")
        print(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}")
        print(f"  Output: {output_name}")

        # Future: Import and run actual ASPIRE trainer
        # from sd_forge_aspire.trainer import AspireImageTrainer
        # trainer = AspireImageTrainer(config)
        # trainer.train()

        return None  # Don't generate images, we're training
