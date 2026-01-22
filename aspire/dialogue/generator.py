"""
Dialogue generation between student and teacher.

This is where the adversarial learning happens - teacher challenges,
student responds, back and forth until the student's reasoning is
thoroughly tested.
"""

import asyncio
from dataclasses import dataclass
from typing import Callable, Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aspire.teachers.base import (
    BaseTeacher,
    DialogueHistory,
    DialogueTurn,
    TeacherChallenge,
    TeacherEvaluation,
)


@dataclass
class GeneratedDialogue:
    """A complete generated dialogue with all metadata."""

    prompt: str
    initial_response: str
    history: DialogueHistory
    final_evaluation: TeacherEvaluation
    turn_evaluations: list[TeacherEvaluation | None]
    metadata: dict[str, Any]


class DialogueGenerator:
    """
    Generates adversarial dialogues between student and teacher.

    The student model generates responses, the teacher challenges them,
    and the dialogue continues until max turns or convergence.
    """

    def __init__(
        self,
        student_model: PreTrainedModel,
        student_tokenizer: PreTrainedTokenizer,
        teacher: BaseTeacher,
        max_turns: int = 3,
        evaluate_each_turn: bool = True,
        student_max_length: int = 512,
        student_temperature: float = 0.7,
        device: str = "cuda",
    ):
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        self.teacher = teacher
        self.max_turns = max_turns
        self.evaluate_each_turn = evaluate_each_turn
        self.student_max_length = student_max_length
        self.student_temperature = student_temperature
        self.device = device

        # Ensure tokenizer has pad token
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token

    async def generate_dialogue(
        self,
        prompt: str,
        initial_response: str | None = None,
    ) -> GeneratedDialogue:
        """
        Generate a complete adversarial dialogue.

        Args:
            prompt: The initial prompt/task
            initial_response: Optional pre-generated initial response

        Returns:
            GeneratedDialogue with full history and evaluations
        """
        # Generate initial response if not provided
        if initial_response is None:
            initial_response = self._generate_student_response(prompt)

        # Initialize history
        history = DialogueHistory(prompt=prompt, initial_response=initial_response)
        turn_evaluations = []

        current_response = initial_response

        # Run dialogue turns
        for turn_num in range(self.max_turns):
            # Teacher generates challenge
            challenge = await self.teacher.challenge(
                prompt=prompt,
                student_response=current_response,
                dialogue_history=history,
            )

            # Student responds to challenge
            student_reply = self._generate_student_response(
                prompt=prompt,
                challenge=challenge.content,
                history=history,
            )

            # Optional per-turn evaluation
            turn_eval = None
            if self.evaluate_each_turn:
                turn_eval = await self.teacher.evaluate(
                    prompt=prompt,
                    student_response=student_reply,
                    dialogue_history=history,
                    generate_improved=False,
                )
            turn_evaluations.append(turn_eval)

            # Record turn
            turn = DialogueTurn(
                turn_number=turn_num + 1,
                challenge=challenge,
                student_response=student_reply,
                evaluation=turn_eval,
            )
            history.add_turn(turn)

            current_response = student_reply

        # Final evaluation
        final_evaluation = await self.teacher.evaluate(
            prompt=prompt,
            student_response=current_response,
            dialogue_history=history,
            generate_improved=True,
        )
        history.final_evaluation = final_evaluation

        return GeneratedDialogue(
            prompt=prompt,
            initial_response=initial_response,
            history=history,
            final_evaluation=final_evaluation,
            turn_evaluations=turn_evaluations,
            metadata={
                "num_turns": len(history.turns),
                "teacher": self.teacher.name,
            },
        )

    def _generate_student_response(
        self,
        prompt: str,
        challenge: str | None = None,
        history: DialogueHistory | None = None,
    ) -> str:
        """Generate a response from the student model."""

        # Format input
        formatted_input = self._format_student_input(prompt, challenge, history)

        # Tokenize
        inputs = self.student_tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.student_max_length,
        ).to(self.device)

        # Generate
        self.student_model.eval()
        with torch.no_grad():
            outputs = self.student_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.student_temperature,
                do_sample=True,
                pad_token_id=self.student_tokenizer.pad_token_id,
                eos_token_id=self.student_tokenizer.eos_token_id,
            )

        # Decode (only new tokens)
        response = self.student_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return response.strip()

    def _format_student_input(
        self,
        prompt: str,
        challenge: str | None = None,
        history: DialogueHistory | None = None,
    ) -> str:
        """Format input for the student model."""

        # Build conversation history
        messages = []

        # System message
        messages.append(
            "You are a helpful assistant engaged in a learning dialogue. "
            "Respond thoughtfully and be willing to revise your thinking when challenged."
        )

        # Initial prompt and response
        messages.append(f"Task: {prompt}")

        if history and history.turns:
            messages.append(f"Your initial response: {history.initial_response}")

            # Previous turns
            for turn in history.turns:
                messages.append(f"Challenge: {turn.challenge.content}")
                messages.append(f"Your response: {turn.student_response}")

        # Current challenge
        if challenge:
            messages.append(f"Challenge: {challenge}")
            messages.append("Your response:")

        return "\n\n".join(messages)

    async def generate_batch(
        self,
        prompts: list[str],
        max_concurrent: int = 5,
    ) -> list[GeneratedDialogue]:
        """
        Generate dialogues for multiple prompts concurrently.

        Args:
            prompts: List of prompts to process
            max_concurrent: Maximum concurrent teacher API calls

        Returns:
            List of generated dialogues
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(prompt: str) -> GeneratedDialogue:
            async with semaphore:
                return await self.generate_dialogue(prompt)

        tasks = [generate_with_semaphore(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def get_student_hidden_states(
        self,
        text: str,
    ) -> torch.Tensor:
        """Get hidden states from student model for a given text."""

        inputs = self.student_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.student_max_length,
        ).to(self.device)

        self.student_model.eval()
        with torch.no_grad():
            outputs = self.student_model(
                **inputs,
                output_hidden_states=True,
            )

        return outputs.hidden_states[-1]  # Last layer
