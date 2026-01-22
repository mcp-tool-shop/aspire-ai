"""
Vision-Language Teacher Models for ASPIRE Image Training.

Teachers critique generated images and provide feedback that
the critic learns to internalize.
"""

import base64
import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from PIL import Image

import anthropic
import openai


@dataclass
class ImageCritique:
    """Structured critique of a generated image."""

    # Overall score (0-10)
    overall_score: float

    # Dimension-specific scores
    aesthetic_score: float = 0.0
    composition_score: float = 0.0
    color_score: float = 0.0
    lighting_score: float = 0.0
    style_score: float = 0.0
    prompt_adherence_score: float = 0.0
    technical_score: float = 0.0

    # Detailed feedback
    reasoning: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    # What a better image would look like (text description)
    improved_description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "scores": {
                "aesthetic": self.aesthetic_score,
                "composition": self.composition_score,
                "color": self.color_score,
                "lighting": self.lighting_score,
                "style": self.style_score,
                "prompt_adherence": self.prompt_adherence_score,
                "technical": self.technical_score,
            },
            "reasoning": self.reasoning,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
            "improved_description": self.improved_description,
        }


class BaseVisionTeacher(ABC):
    """Base class for vision-language teacher models."""

    def __init__(
        self,
        name: str = "Vision Teacher",
        persona: str = "balanced",
    ):
        self.name = name
        self.persona = persona

    @abstractmethod
    async def critique(
        self,
        image: Image.Image,
        prompt: str,
        context: str | None = None,
    ) -> ImageCritique:
        """
        Critique a generated image.

        Args:
            image: The generated PIL Image
            prompt: The prompt used to generate the image
            context: Optional additional context

        Returns:
            Structured critique with scores and feedback
        """
        pass

    def _get_system_prompt(self) -> str:
        """Get system prompt based on persona."""
        personas = {
            "balanced": """You are an expert art critic and visual analyst.
Evaluate images fairly but rigorously, considering both technical and artistic merit.
Provide constructive feedback that helps improve future generations.""",

            "technical": """You are a technical image quality analyst.
Focus on resolution, sharpness, noise levels, artifacts, color accuracy,
and proper rendering of details. Be precise and quantitative.""",

            "artistic": """You are an artistic visionary and curator.
Evaluate images for their creative merit, emotional impact, originality,
and artistic expression. Consider how the image makes you feel.""",

            "composition": """You are a composition and design expert.
Analyze balance, focal points, rule of thirds, leading lines, negative space,
and visual flow. Focus on how elements are arranged within the frame.""",

            "color": """You are a color theory specialist.
Evaluate color harmony, contrast, saturation, temperature, and mood.
Consider how colors work together and support the image's intent.""",

            "harsh": """You are a demanding critic with very high standards.
Find flaws others would miss. Be direct about weaknesses.
Only give high scores to truly exceptional work.""",

            "encouraging": """You are a supportive mentor who sees potential.
Acknowledge what works well before suggesting improvements.
Frame feedback constructively while maintaining honesty.""",
        }
        return personas.get(self.persona, personas["balanced"])

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 for API."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class ClaudeVisionTeacher(BaseVisionTeacher):
    """Claude-based vision teacher using Claude's image understanding."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(name="Claude Vision", **kwargs)
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def critique(
        self,
        image: Image.Image,
        prompt: str,
        context: str | None = None,
    ) -> ImageCritique:
        """Critique an image using Claude's vision capabilities."""

        image_data = self._encode_image(image)

        critique_prompt = f"""Analyze this AI-generated image.

**Original Prompt:** {prompt}

{f"**Additional Context:** {context}" if context else ""}

Evaluate the image and respond with JSON:

{{
    "overall_score": 7.5,  // 0-10
    "aesthetic_score": 7.0,
    "composition_score": 8.0,
    "color_score": 7.5,
    "lighting_score": 6.5,
    "style_score": 7.0,
    "prompt_adherence_score": 8.0,
    "technical_score": 7.0,
    "reasoning": "Overall assessment explaining the scores...",
    "strengths": ["Good color palette", "Strong composition"],
    "weaknesses": ["Lighting could be more dramatic", "Some artifacts visible"],
    "suggestions": ["Increase contrast", "Add more depth to shadows"],
    "improved_description": "A description of how an improved version would look..."
}}"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self._get_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": critique_prompt,
                        },
                    ],
                }
            ],
        )

        # Parse response
        try:
            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)

            return ImageCritique(
                overall_score=data["overall_score"],
                aesthetic_score=data.get("aesthetic_score", 0),
                composition_score=data.get("composition_score", 0),
                color_score=data.get("color_score", 0),
                lighting_score=data.get("lighting_score", 0),
                style_score=data.get("style_score", 0),
                prompt_adherence_score=data.get("prompt_adherence_score", 0),
                technical_score=data.get("technical_score", 0),
                reasoning=data.get("reasoning", ""),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                suggestions=data.get("suggestions", []),
                improved_description=data.get("improved_description"),
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback
            return ImageCritique(
                overall_score=5.0,
                reasoning=response.content[0].text,
            )


class GPT4VisionTeacher(BaseVisionTeacher):
    """GPT-4V based vision teacher."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(name="GPT-4 Vision", **kwargs)
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def critique(
        self,
        image: Image.Image,
        prompt: str,
        context: str | None = None,
    ) -> ImageCritique:
        """Critique an image using GPT-4V."""

        image_data = self._encode_image(image)

        critique_prompt = f"""Analyze this AI-generated image.

**Original Prompt:** {prompt}

{f"**Additional Context:** {context}" if context else ""}

Evaluate and respond with JSON containing:
- overall_score (0-10)
- aesthetic_score, composition_score, color_score, lighting_score, style_score, prompt_adherence_score, technical_score
- reasoning, strengths, weaknesses, suggestions
- improved_description"""

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                            },
                        },
                        {"type": "text", "text": critique_prompt},
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.choices[0].message.content)
            return ImageCritique(
                overall_score=data["overall_score"],
                aesthetic_score=data.get("aesthetic_score", 0),
                composition_score=data.get("composition_score", 0),
                color_score=data.get("color_score", 0),
                lighting_score=data.get("lighting_score", 0),
                style_score=data.get("style_score", 0),
                prompt_adherence_score=data.get("prompt_adherence_score", 0),
                technical_score=data.get("technical_score", 0),
                reasoning=data.get("reasoning", ""),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                suggestions=data.get("suggestions", []),
                improved_description=data.get("improved_description"),
            )
        except (json.JSONDecodeError, KeyError):
            return ImageCritique(
                overall_score=5.0,
                reasoning=response.choices[0].message.content,
            )


def get_vision_teacher(
    teacher_type: str = "claude",
    persona: str = "balanced",
    **kwargs,
) -> BaseVisionTeacher:
    """Factory function to create vision teachers."""
    teachers = {
        "claude": ClaudeVisionTeacher,
        "gpt4v": GPT4VisionTeacher,
        "gpt4": GPT4VisionTeacher,
    }

    teacher_class = teachers.get(teacher_type.lower())
    if teacher_class is None:
        raise ValueError(f"Unknown teacher type: {teacher_type}")

    return teacher_class(persona=persona, **kwargs)
