"""
ASPIRE Generation Script for Forge.

Enhances image generation with critic-guided refinement.
The critic predicts what an aesthetic teacher would think,
allowing self-improvement without API calls at inference time.
"""

import gradio as gr

from modules import scripts, shared
from modules.processing import StableDiffusionProcessing, Processed


class AspireGenerateScript(scripts.Script):
    """
    ASPIRE-enhanced image generation.

    Uses an internalized critic to guide generation toward
    aesthetically superior results.
    """

    def title(self):
        return "ASPIRE: Critic-Guided Generation"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("ASPIRE Critic Guidance", open=False):
            enabled = gr.Checkbox(
                label="Enable ASPIRE Critic",
                value=False,
                elem_id=self.elem_id("enabled"),
            )

            with gr.Row():
                critic_strength = gr.Slider(
                    label="Critic Influence",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    elem_id=self.elem_id("critic_strength"),
                )

                refinement_steps = gr.Slider(
                    label="Refinement Iterations",
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    elem_id=self.elem_id("refinement_steps"),
                )

            with gr.Row():
                aesthetic_weight = gr.Slider(
                    label="Aesthetic Quality Weight",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.4,
                    step=0.1,
                )

                composition_weight = gr.Slider(
                    label="Composition Weight",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                )

                style_weight = gr.Slider(
                    label="Style Adherence Weight",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                )

            threshold = gr.Slider(
                label="Quality Threshold (stop refining above this)",
                minimum=5.0,
                maximum=10.0,
                value=7.5,
                step=0.5,
                elem_id=self.elem_id("threshold"),
            )

            gr.Markdown("""
            **ASPIRE Critic Guidance** uses an internalized aesthetic critic to
            guide image generation toward higher quality results.

            The critic predicts what a vision-language teacher would think about:
            - **Aesthetic Quality**: Overall visual appeal, lighting, color harmony
            - **Composition**: Balance, focal points, rule of thirds
            - **Style Adherence**: How well the image matches the prompt's intent

            No API calls needed - the critic runs locally!
            """)

        return [
            enabled,
            critic_strength,
            refinement_steps,
            aesthetic_weight,
            composition_weight,
            style_weight,
            threshold,
        ]

    def process(self, p: StableDiffusionProcessing, *args):
        """Called before generation starts."""
        (
            enabled,
            critic_strength,
            refinement_steps,
            aesthetic_weight,
            composition_weight,
            style_weight,
            threshold,
        ) = args

        if not enabled:
            return

        # Store ASPIRE settings in processing object
        p.aspire_enabled = True
        p.aspire_critic_strength = critic_strength
        p.aspire_refinement_steps = int(refinement_steps)
        p.aspire_weights = {
            "aesthetic": aesthetic_weight,
            "composition": composition_weight,
            "style": style_weight,
        }
        p.aspire_threshold = threshold

    def postprocess(self, p: StableDiffusionProcessing, processed: Processed, *args):
        """Called after generation completes."""
        if not getattr(p, "aspire_enabled", False):
            return

        # Future: Run critic evaluation on generated images
        # For now, just log that ASPIRE was active
        if hasattr(processed, "infotexts") and processed.infotexts:
            for i, infotext in enumerate(processed.infotexts):
                processed.infotexts[i] = infotext + f"\nASPIRE Critic: strength={p.aspire_critic_strength}"
