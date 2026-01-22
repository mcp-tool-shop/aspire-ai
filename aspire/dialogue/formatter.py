"""
Dialogue formatting utilities.

Converts dialogues into formats suitable for training.
"""

from dataclasses import dataclass

from aspire.dialogue.generator import GeneratedDialogue
from aspire.teachers.base import DialogueHistory


@dataclass
class FormattedDialogue:
    """Dialogue formatted for training."""

    # Input text (prompt + context)
    input_text: str

    # Target text (ideal response)
    target_text: str

    # Full conversation for context
    full_conversation: str

    # Metadata
    score: float
    improved_response: str | None
    num_turns: int


class DialogueFormatter:
    """
    Formats dialogues for different training objectives.

    Supports multiple formats:
    - "standard": Input -> Output pairs
    - "chat": Chat-style with roles
    - "instruction": Instruction-following format
    """

    def __init__(
        self,
        format_type: str = "chat",
        system_message: str | None = None,
        include_reasoning: bool = False,
    ):
        self.format_type = format_type
        self.system_message = system_message or (
            "You are a helpful assistant engaged in a learning dialogue. "
            "Respond thoughtfully and improve based on feedback."
        )
        self.include_reasoning = include_reasoning

    def format_dialogue(
        self,
        dialogue: GeneratedDialogue,
        use_improved_as_target: bool = True,
    ) -> FormattedDialogue:
        """
        Format a dialogue for training.

        Args:
            dialogue: The generated dialogue
            use_improved_as_target: If True, use teacher's improved response as target

        Returns:
            FormattedDialogue ready for training
        """
        if self.format_type == "standard":
            return self._format_standard(dialogue, use_improved_as_target)
        elif self.format_type == "chat":
            return self._format_chat(dialogue, use_improved_as_target)
        elif self.format_type == "instruction":
            return self._format_instruction(dialogue, use_improved_as_target)
        else:
            raise ValueError(f"Unknown format type: {self.format_type}")

    def _format_standard(
        self,
        dialogue: GeneratedDialogue,
        use_improved_as_target: bool,
    ) -> FormattedDialogue:
        """Standard input/output format."""

        # Input: just the prompt
        input_text = dialogue.prompt

        # Target: improved response or last student response
        if use_improved_as_target and dialogue.final_evaluation.improved_response:
            target_text = dialogue.final_evaluation.improved_response
        else:
            if dialogue.history.turns:
                target_text = dialogue.history.turns[-1].student_response
            else:
                target_text = dialogue.initial_response

        # Full conversation
        full_conversation = self._build_full_conversation(dialogue)

        return FormattedDialogue(
            input_text=input_text,
            target_text=target_text,
            full_conversation=full_conversation,
            score=dialogue.final_evaluation.overall_score,
            improved_response=dialogue.final_evaluation.improved_response,
            num_turns=len(dialogue.history.turns),
        )

    def _format_chat(
        self,
        dialogue: GeneratedDialogue,
        use_improved_as_target: bool,
    ) -> FormattedDialogue:
        """Chat-style format with roles."""

        messages = []

        # System
        messages.append(f"<|system|>\n{self.system_message}\n<|end|>")

        # User prompt
        messages.append(f"<|user|>\n{dialogue.prompt}\n<|end|>")

        # If we have dialogue turns, include them
        if dialogue.history.turns:
            # Initial response
            messages.append(f"<|assistant|>\n{dialogue.initial_response}\n<|end|>")

            # Dialogue turns
            for turn in dialogue.history.turns:
                messages.append(f"<|user|>\n{turn.challenge.content}\n<|end|>")
                messages.append(f"<|assistant|>\n{turn.student_response}\n<|end|>")

        input_text = "\n".join(messages[:-1])  # Everything except last assistant turn

        # Target
        if use_improved_as_target and dialogue.final_evaluation.improved_response:
            target_text = dialogue.final_evaluation.improved_response
        else:
            if dialogue.history.turns:
                target_text = dialogue.history.turns[-1].student_response
            else:
                target_text = dialogue.initial_response

        return FormattedDialogue(
            input_text=input_text,
            target_text=target_text,
            full_conversation="\n".join(messages),
            score=dialogue.final_evaluation.overall_score,
            improved_response=dialogue.final_evaluation.improved_response,
            num_turns=len(dialogue.history.turns),
        )

    def _format_instruction(
        self,
        dialogue: GeneratedDialogue,
        use_improved_as_target: bool,
    ) -> FormattedDialogue:
        """Instruction-following format."""

        # Build instruction with context
        instruction_parts = [
            "### Instruction:",
            dialogue.prompt,
        ]

        # Include feedback context if available
        if dialogue.history.turns and self.include_reasoning:
            instruction_parts.append("\n### Feedback from previous attempt:")
            last_turn = dialogue.history.turns[-1]
            if last_turn.evaluation:
                instruction_parts.append(last_turn.evaluation.reasoning)

        instruction_parts.append("\n### Response:")

        input_text = "\n".join(instruction_parts)

        # Target
        if use_improved_as_target and dialogue.final_evaluation.improved_response:
            target_text = dialogue.final_evaluation.improved_response
        else:
            if dialogue.history.turns:
                target_text = dialogue.history.turns[-1].student_response
            else:
                target_text = dialogue.initial_response

        return FormattedDialogue(
            input_text=input_text,
            target_text=target_text,
            full_conversation=input_text + "\n" + target_text,
            score=dialogue.final_evaluation.overall_score,
            improved_response=dialogue.final_evaluation.improved_response,
            num_turns=len(dialogue.history.turns),
        )

    def _build_full_conversation(self, dialogue: GeneratedDialogue) -> str:
        """Build full conversation string."""
        parts = [f"Prompt: {dialogue.prompt}"]
        parts.append(f"Initial: {dialogue.initial_response}")

        for turn in dialogue.history.turns:
            parts.append(f"Challenge: {turn.challenge.content}")
            parts.append(f"Response: {turn.student_response}")

        if dialogue.final_evaluation.improved_response:
            parts.append(f"Improved: {dialogue.final_evaluation.improved_response}")

        return "\n\n".join(parts)

    def format_for_critic(
        self,
        dialogue: GeneratedDialogue,
        response_to_evaluate: str | None = None,
    ) -> str:
        """
        Format dialogue for critic evaluation.

        Creates input that the critic can use to predict teacher judgment.
        """
        parts = [f"Task: {dialogue.prompt}"]

        if response_to_evaluate:
            parts.append(f"Response: {response_to_evaluate}")
        else:
            if dialogue.history.turns:
                parts.append(f"Response: {dialogue.history.turns[-1].student_response}")
            else:
                parts.append(f"Response: {dialogue.initial_response}")

        # Include dialogue context
        if dialogue.history.turns:
            parts.append("\nDialogue context:")
            for turn in dialogue.history.turns[:-1]:  # All but last
                parts.append(f"Q: {turn.challenge.content}")
                parts.append(f"A: {turn.student_response}")

        return "\n".join(parts)
