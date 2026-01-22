"""
Local model teacher implementation.

Enables using local LLMs (via transformers) as teachers, useful for:
- Faster iteration during development
- Privacy-sensitive applications
- Cost reduction at scale
- Specialized domain teachers from fine-tuned models
"""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from aspire.teachers.base import (
    BaseTeacher,
    ChallengeType,
    DialogueHistory,
    DimensionScore,
    EvaluationDimension,
    TeacherChallenge,
    TeacherEvaluation,
)


class LocalTeacher(BaseTeacher):
    """
    Teacher powered by a local model.

    Useful for faster iteration, privacy, or using specialized
    fine-tuned models as domain experts.
    """

    def __init__(
        self,
        model_name_or_path: str,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device: str = "cuda",
        name: str = "Local Teacher",
        description: str = "A teacher powered by a local language model",
        **kwargs,
    ):
        super().__init__(name=name, description=description, **kwargs)
        self.model_name_or_path = model_name_or_path
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()

    async def challenge(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        challenge_type: ChallengeType | None = None,
    ) -> TeacherChallenge:
        """Generate a challenge using local model."""

        if challenge_type is None:
            challenge_type = self.select_challenge_type(dialogue_history)

        history_context = ""
        if dialogue_history and dialogue_history.turns:
            history_context = "\n\nPrevious dialogue:\n"
            for turn in dialogue_history.turns:
                history_context += f"Challenge: {turn.challenge.content}\n"
                history_context += f"Student: {turn.student_response}\n\n"

        challenge_prompt = f"""<|system|>
{self.get_system_prompt()}<|end|>
<|user|>
Generate a {challenge_type.value} challenge for this student response.

Original prompt: {prompt}

Student's response: {student_response}
{history_context}

Generate a single challenging question or statement.<|end|>
<|assistant|>
"""

        inputs = self.tokenizer(
            challenge_prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return TeacherChallenge(
            challenge_type=challenge_type,
            content=response.strip(),
            difficulty=0.5,
        )

    async def evaluate(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        generate_improved: bool = True,
    ) -> TeacherEvaluation:
        """Evaluate using local model."""

        history_context = ""
        if dialogue_history and dialogue_history.turns:
            history_context = "\n\nDialogue history:\n"
            for turn in dialogue_history.turns:
                history_context += f"Challenge: {turn.challenge.content}\n"
                history_context += f"Student: {turn.student_response}\n\n"

        eval_prompt = f"""<|system|>
{self.get_system_prompt()}<|end|>
<|user|>
Evaluate this student response on a scale of 0-10.

Original prompt: {prompt}

Student's response: {student_response}
{history_context}

Provide:
1. A score from 0-10
2. Brief reasoning
3. Key strengths
4. Key weaknesses
{"5. An improved version of the response" if generate_improved else ""}
<|end|>
<|assistant|>
"""

        inputs = self.tokenizer(
            eval_prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Parse the response (basic parsing - local models may not follow JSON)
        score = self._extract_score(response)
        improved = self._extract_improved(response) if generate_improved else None

        return TeacherEvaluation(
            overall_score=score,
            dimension_scores=[],
            reasoning=response,
            improved_response=improved,
        )

    def _extract_score(self, response: str) -> float:
        """Extract numeric score from response."""
        import re

        # Look for patterns like "Score: 7" or "7/10" or just a number
        patterns = [
            r"[Ss]core:?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            r"^(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                score = float(match.group(1))
                return min(10.0, max(0.0, score))

        return 5.0  # Default

    def _extract_improved(self, response: str) -> str | None:
        """Extract improved response if present."""
        markers = ["improved version:", "better response:", "improved:"]
        lower_response = response.lower()

        for marker in markers:
            if marker in lower_response:
                idx = lower_response.index(marker)
                return response[idx + len(marker) :].strip()

        return None
