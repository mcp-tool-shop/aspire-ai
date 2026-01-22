"""
Composite teacher - combines multiple teachers for richer learning.

The idea: different teachers provide different perspectives, and the student
learns from all of them. Like having a committee of mentors, each bringing
their own expertise and style.
"""

import asyncio
import random
from typing import Callable

from aspire.teachers.base import (
    BaseTeacher,
    ChallengeType,
    DialogueHistory,
    DimensionScore,
    EvaluationDimension,
    TeacherChallenge,
    TeacherEvaluation,
)


class CompositeTeacher(BaseTeacher):
    """
    A teacher that combines multiple teachers.

    Strategies:
    - "rotate": Different teacher each turn
    - "vote": All teachers evaluate, combine scores
    - "specialize": Different teachers for different challenge types
    - "debate": Teachers discuss/debate the evaluation

    This enables rich, multi-perspective learning.
    """

    def __init__(
        self,
        teachers: list[BaseTeacher],
        strategy: str = "rotate",
        weights: list[float] | None = None,
        name: str = "Composite Teacher",
        description: str = "A committee of teachers providing multiple perspectives",
        **kwargs,
    ):
        super().__init__(name=name, description=description, **kwargs)

        if not teachers:
            raise ValueError("Must provide at least one teacher")

        self.teachers = teachers
        self.strategy = strategy
        self.weights = weights or [1.0 / len(teachers)] * len(teachers)

        if len(self.weights) != len(self.teachers):
            raise ValueError("Weights must match number of teachers")

        # Track which teacher is current (for rotate strategy)
        self._current_teacher_idx = 0

        # Merge preferred challenges from all teachers
        all_challenges = set()
        for teacher in teachers:
            all_challenges.update(teacher.preferred_challenges)
        self.preferred_challenges = list(all_challenges)

        # Merge evaluation dimensions
        all_dimensions = set()
        for teacher in teachers:
            all_dimensions.update(teacher.evaluation_dimensions)
        self.evaluation_dimensions = list(all_dimensions)

    async def challenge(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        challenge_type: ChallengeType | None = None,
    ) -> TeacherChallenge:
        """Generate challenge based on strategy."""

        if self.strategy == "rotate":
            # Rotate through teachers
            teacher = self.teachers[self._current_teacher_idx]
            self._current_teacher_idx = (self._current_teacher_idx + 1) % len(self.teachers)
            return await teacher.challenge(
                prompt, student_response, dialogue_history, challenge_type
            )

        elif self.strategy == "specialize":
            # Pick teacher best suited for challenge type
            if challenge_type is None:
                challenge_type = self.select_challenge_type(dialogue_history)

            # Find teacher with this challenge type in their preferences
            for teacher in self.teachers:
                if challenge_type in teacher.preferred_challenges:
                    return await teacher.challenge(
                        prompt, student_response, dialogue_history, challenge_type
                    )

            # Fallback to first teacher
            return await self.teachers[0].challenge(
                prompt, student_response, dialogue_history, challenge_type
            )

        elif self.strategy == "random":
            # Random teacher weighted by weights
            teacher = random.choices(self.teachers, weights=self.weights, k=1)[0]
            return await teacher.challenge(
                prompt, student_response, dialogue_history, challenge_type
            )

        else:
            # Default: use first teacher
            return await self.teachers[0].challenge(
                prompt, student_response, dialogue_history, challenge_type
            )

    async def evaluate(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        generate_improved: bool = True,
    ) -> TeacherEvaluation:
        """Evaluate based on strategy."""

        if self.strategy == "vote":
            # All teachers evaluate, combine results
            return await self._vote_evaluation(
                prompt, student_response, dialogue_history, generate_improved
            )

        elif self.strategy == "debate":
            # Teachers evaluate, then we synthesize
            return await self._debate_evaluation(
                prompt, student_response, dialogue_history, generate_improved
            )

        elif self.strategy == "rotate":
            # Use current teacher
            teacher = self.teachers[self._current_teacher_idx]
            return await teacher.evaluate(
                prompt, student_response, dialogue_history, generate_improved
            )

        else:
            # Default: use first teacher
            return await self.teachers[0].evaluate(
                prompt, student_response, dialogue_history, generate_improved
            )

    async def _vote_evaluation(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None,
        generate_improved: bool,
    ) -> TeacherEvaluation:
        """All teachers vote, combine weighted scores."""

        # Get all evaluations in parallel
        tasks = [
            teacher.evaluate(prompt, student_response, dialogue_history, generate_improved)
            for teacher in self.teachers
        ]
        evaluations = await asyncio.gather(*tasks)

        # Weighted average of scores
        total_weight = sum(self.weights)
        weighted_score = sum(
            eval.overall_score * weight for eval, weight in zip(evaluations, self.weights)
        ) / total_weight

        # Combine dimension scores
        dimension_scores_map: dict[EvaluationDimension, list[tuple[float, float]]] = {}
        for eval, weight in zip(evaluations, self.weights):
            for ds in eval.dimension_scores:
                if ds.dimension not in dimension_scores_map:
                    dimension_scores_map[ds.dimension] = []
                dimension_scores_map[ds.dimension].append((ds.score, weight))

        combined_dimensions = []
        for dim, scores_weights in dimension_scores_map.items():
            weighted_dim_score = sum(s * w for s, w in scores_weights) / sum(
                w for _, w in scores_weights
            )
            combined_dimensions.append(
                DimensionScore(
                    dimension=dim,
                    score=weighted_dim_score,
                    explanation=f"Combined from {len(scores_weights)} teachers",
                )
            )

        # Combine feedback
        all_strengths = []
        all_weaknesses = []
        all_suggestions = []
        all_reasoning = []

        for eval, teacher in zip(evaluations, self.teachers):
            all_strengths.extend(eval.strengths)
            all_weaknesses.extend(eval.weaknesses)
            all_suggestions.extend(eval.suggestions)
            all_reasoning.append(f"[{teacher.name}]: {eval.reasoning}")

        # Pick best improved response (from highest-scoring teacher)
        best_eval = max(evaluations, key=lambda e: e.overall_score)
        improved = best_eval.improved_response if generate_improved else None

        return TeacherEvaluation(
            overall_score=weighted_score,
            dimension_scores=combined_dimensions,
            reasoning="\n\n".join(all_reasoning),
            improved_response=improved,
            strengths=list(set(all_strengths)),  # Dedupe
            weaknesses=list(set(all_weaknesses)),
            suggestions=list(set(all_suggestions)),
            metadata={"strategy": "vote", "num_teachers": len(self.teachers)},
        )

    async def _debate_evaluation(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None,
        generate_improved: bool,
    ) -> TeacherEvaluation:
        """
        Teachers evaluate, then synthesize through debate.

        Currently implemented as vote + rich combination.
        Future: actual multi-turn debate between teachers.
        """
        # For now, use vote strategy with richer combination
        # Future enhancement: teachers actually debate each other
        return await self._vote_evaluation(
            prompt, student_response, dialogue_history, generate_improved
        )

    def get_system_prompt(self) -> str:
        teacher_list = ", ".join([t.name for t in self.teachers])
        return f"""You are a composite teacher combining the perspectives of: {teacher_list}.

Your role is to provide multi-faceted evaluation and challenge, drawing on
the different strengths and approaches of your component teachers."""


class CurriculumCompositeTeacher(CompositeTeacher):
    """
    A composite teacher that changes composition based on curriculum stage.

    Early stages might emphasize compassionate teaching.
    Later stages might add adversarial challenge.
    """

    def __init__(
        self,
        teachers: list[BaseTeacher],
        stage_weights: dict[str, list[float]],
        current_stage: str = "foundation",
        **kwargs,
    ):
        """
        Args:
            teachers: List of teachers
            stage_weights: Mapping from stage name to weights for each teacher
            current_stage: Initial curriculum stage
        """
        super().__init__(teachers=teachers, strategy="vote", **kwargs)
        self.stage_weights = stage_weights
        self.current_stage = current_stage
        self._update_weights()

    def set_stage(self, stage: str) -> None:
        """Update curriculum stage and adjust weights."""
        if stage in self.stage_weights:
            self.current_stage = stage
            self._update_weights()

    def _update_weights(self) -> None:
        """Update teacher weights based on current stage."""
        if self.current_stage in self.stage_weights:
            self.weights = self.stage_weights[self.current_stage]
