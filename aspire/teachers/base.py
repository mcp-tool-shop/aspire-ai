"""
Base teacher interface and data structures.

Teachers are the source of wisdom in ASPIRE. They evaluate student responses,
provide challenges, and explain their reasoning. Different teachers embody
different teaching philosophies and produce different learning outcomes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChallengeType(str, Enum):
    """Types of challenges a teacher can pose."""

    PROBE_REASONING = "probe_reasoning"  # "Why do you think that?"
    EDGE_CASE = "edge_case"  # "What about when X?"
    DEVILS_ADVOCATE = "devils_advocate"  # "Couldn't you argue the opposite?"
    SOCRATIC = "socratic"  # "What assumption are you making?"
    CLARIFICATION = "clarification"  # "What do you mean by Y?"
    EXTENSION = "extension"  # "How would this apply to Z?"
    CONTRADICTION = "contradiction"  # "Earlier you said A, but now B?"
    STEELMAN = "steelman"  # "What's the strongest counter-argument?"
    EMOTIONAL = "emotional"  # "How might someone feel about this?"
    PRACTICAL = "practical"  # "How would this work in practice?"
    ETHICAL = "ethical"  # "What are the ethical implications?"
    CREATIVE = "creative"  # "What if we approached this differently?"


class EvaluationDimension(str, Enum):
    """Dimensions on which responses are evaluated."""

    CORRECTNESS = "correctness"  # Factually/logically correct?
    REASONING = "reasoning"  # Sound, well-explained reasoning?
    NUANCE = "nuance"  # Acknowledges complexity, edge cases?
    ADAPTABILITY = "adaptability"  # Updates appropriately when challenged?
    CLARITY = "clarity"  # Clearly communicated?
    INTELLECTUAL_HONESTY = "intellectual_honesty"  # Admits uncertainty?
    CREATIVITY = "creativity"  # Novel or insightful perspectives?
    EMPATHY = "empathy"  # Considers human impact?
    PRACTICALITY = "practicality"  # Actionable and realistic?


@dataclass
class TeacherChallenge:
    """A challenge posed by the teacher to the student."""

    challenge_type: ChallengeType
    content: str
    context: str | None = None  # Why this challenge was chosen
    difficulty: float = 0.5  # 0.0 (easy) to 1.0 (hard)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""

    dimension: EvaluationDimension
    score: float  # 0.0 to 10.0
    explanation: str


@dataclass
class TeacherEvaluation:
    """Complete evaluation from the teacher."""

    # Overall score (0-10)
    overall_score: float

    # Per-dimension scores
    dimension_scores: list[DimensionScore]

    # Detailed reasoning for the evaluation
    reasoning: str

    # What a better response would look like
    improved_response: str | None = None

    # Specific feedback points
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether the response meets minimum standards."""
        return self.overall_score >= 6.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": self.overall_score,
            "dimension_scores": [
                {"dimension": d.dimension.value, "score": d.score, "explanation": d.explanation}
                for d in self.dimension_scores
            ],
            "reasoning": self.reasoning,
            "improved_response": self.improved_response,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }


@dataclass
class DialogueTurn:
    """A single turn in the adversarial dialogue."""

    turn_number: int
    challenge: TeacherChallenge
    student_response: str
    evaluation: TeacherEvaluation | None = None  # Optional per-turn eval
    timestamp: float | None = None


@dataclass
class DialogueHistory:
    """Complete dialogue history between teacher and student."""

    prompt: str
    initial_response: str
    turns: list[DialogueTurn] = field(default_factory=list)
    final_evaluation: TeacherEvaluation | None = None

    def add_turn(self, turn: DialogueTurn) -> None:
        """Add a dialogue turn."""
        self.turns.append(turn)

    @property
    def num_turns(self) -> int:
        """Number of dialogue turns."""
        return len(self.turns)

    def get_trajectory_scores(self) -> list[float]:
        """Get scores across the dialogue (if available)."""
        scores = []
        for turn in self.turns:
            if turn.evaluation:
                scores.append(turn.evaluation.overall_score)
        return scores


class BaseTeacher(ABC):
    """
    Abstract base class for all teachers.

    Teachers embody different teaching philosophies and styles. A Socratic
    teacher asks probing questions. A scientific teacher demands evidence.
    A creative teacher encourages novel thinking. Each produces different
    learning outcomes in the student.

    Subclass this to create new types of teachers.
    """

    def __init__(
        self,
        name: str = "BaseTeacher",
        description: str = "A teacher for ASPIRE training",
        evaluation_dimensions: list[EvaluationDimension] | None = None,
        preferred_challenges: list[ChallengeType] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.name = name
        self.description = description
        self.evaluation_dimensions = evaluation_dimensions or list(EvaluationDimension)
        self.preferred_challenges = preferred_challenges or list(ChallengeType)
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def challenge(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        challenge_type: ChallengeType | None = None,
    ) -> TeacherChallenge:
        """
        Generate a challenge for the student.

        Args:
            prompt: The original prompt/task
            student_response: The student's current response
            dialogue_history: Previous turns in the dialogue
            challenge_type: Specific challenge type to use (or None for auto-select)

        Returns:
            A challenge for the student to respond to
        """
        pass

    @abstractmethod
    async def evaluate(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        generate_improved: bool = True,
    ) -> TeacherEvaluation:
        """
        Evaluate a student's response.

        Args:
            prompt: The original prompt/task
            student_response: The student's response to evaluate
            dialogue_history: The full dialogue history
            generate_improved: Whether to generate an improved version

        Returns:
            Complete evaluation with scores, reasoning, and suggestions
        """
        pass

    async def run_dialogue(
        self,
        prompt: str,
        student_response: str,
        student_generate_fn: callable,
        max_turns: int = 3,
        evaluate_each_turn: bool = False,
    ) -> DialogueHistory:
        """
        Run a complete adversarial dialogue with the student.

        Args:
            prompt: The original prompt/task
            student_response: The student's initial response
            student_generate_fn: Function to generate student responses
            max_turns: Maximum dialogue turns
            evaluate_each_turn: Whether to evaluate after each turn

        Returns:
            Complete dialogue history with final evaluation
        """
        history = DialogueHistory(prompt=prompt, initial_response=student_response)

        current_response = student_response

        for turn_num in range(max_turns):
            # Teacher generates challenge
            challenge = await self.challenge(
                prompt=prompt,
                student_response=current_response,
                dialogue_history=history,
            )

            # Student responds to challenge
            student_reply = await student_generate_fn(
                prompt=prompt,
                challenge=challenge.content,
                dialogue_history=history,
            )

            # Optional per-turn evaluation
            turn_eval = None
            if evaluate_each_turn:
                turn_eval = await self.evaluate(
                    prompt=prompt,
                    student_response=student_reply,
                    dialogue_history=history,
                    generate_improved=False,
                )

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
        history.final_evaluation = await self.evaluate(
            prompt=prompt,
            student_response=current_response,
            dialogue_history=history,
            generate_improved=True,
        )

        return history

    def select_challenge_type(
        self,
        dialogue_history: DialogueHistory | None = None,
        curriculum_stage: str | None = None,
    ) -> ChallengeType:
        """
        Select an appropriate challenge type based on context.

        Override this to implement custom challenge selection logic.
        """
        import random

        # Default: random from preferred challenges
        return random.choice(self.preferred_challenges)

    def get_system_prompt(self) -> str:
        """
        Get the system prompt that defines this teacher's persona.

        Override this to customize the teacher's personality and approach.
        """
        return f"""You are {self.name}, a teacher in the ASPIRE training system.

{self.description}

Your role is to:
1. Challenge student responses to deepen their understanding
2. Evaluate responses fairly but rigorously
3. Provide constructive feedback that helps the student improve
4. Model good reasoning and intellectual honesty

Be direct, insightful, and focused on helping the student develop genuine understanding,
not just surface-level pattern matching."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
