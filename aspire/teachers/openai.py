"""
OpenAI-based teacher implementation.
"""

import json

from openai import AsyncOpenAI

from aspire.teachers.base import (
    BaseTeacher,
    ChallengeType,
    DialogueHistory,
    DimensionScore,
    EvaluationDimension,
    TeacherChallenge,
    TeacherEvaluation,
)


class OpenAITeacher(BaseTeacher):
    """
    Teacher powered by OpenAI API (GPT-4, etc.).

    GPT-4 provides strong reasoning and evaluation capabilities,
    offering a different perspective from Claude.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        name: str = "GPT-4 Teacher",
        description: str = "A rigorous, analytical teacher powered by GPT-4",
        **kwargs,
    ):
        super().__init__(name=name, description=description, **kwargs)
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def challenge(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        challenge_type: ChallengeType | None = None,
    ) -> TeacherChallenge:
        """Generate a challenge using OpenAI."""

        if challenge_type is None:
            challenge_type = self.select_challenge_type(dialogue_history)

        history_context = ""
        if dialogue_history and dialogue_history.turns:
            history_context = "\n\nPrevious dialogue:\n"
            for turn in dialogue_history.turns:
                history_context += f"Challenge: {turn.challenge.content}\n"
                history_context += f"Student: {turn.student_response}\n\n"

        challenge_prompt = f"""Generate a {challenge_type.value} challenge for this student response.

Original prompt: {prompt}

Student's response: {student_response}
{history_context}

Respond with JSON only:
{{
    "challenge": "Your challenge question",
    "context": "Why this challenge",
    "difficulty": 0.5
}}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": challenge_prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.choices[0].message.content)
            return TeacherChallenge(
                challenge_type=challenge_type,
                content=data["challenge"],
                context=data.get("context"),
                difficulty=data.get("difficulty", 0.5),
            )
        except (json.JSONDecodeError, KeyError):
            return TeacherChallenge(
                challenge_type=challenge_type,
                content=response.choices[0].message.content,
                difficulty=0.5,
            )

    async def evaluate(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        generate_improved: bool = True,
    ) -> TeacherEvaluation:
        """Evaluate using OpenAI."""

        history_context = ""
        if dialogue_history and dialogue_history.turns:
            history_context = "\n\nDialogue history:\n"
            for turn in dialogue_history.turns:
                history_context += f"Challenge: {turn.challenge.content}\n"
                history_context += f"Student: {turn.student_response}\n\n"

        dimensions_list = ", ".join([d.value for d in self.evaluation_dimensions])

        eval_prompt = f"""Evaluate this student response.

Original prompt: {prompt}
Student's response: {student_response}
{history_context}

Dimensions: {dimensions_list}

Respond with JSON:
{{
    "overall_score": 7.5,
    "dimension_scores": [{{"dimension": "correctness", "score": 8.0, "explanation": "..."}}],
    "reasoning": "...",
    "strengths": [],
    "weaknesses": [],
    "suggestions": [],
    {"\"improved_response\": \"...\"" if generate_improved else ""}
}}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens * 2,
            temperature=0.3,
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": eval_prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.choices[0].message.content)

            dimension_scores = []
            for ds in data.get("dimension_scores", []):
                try:
                    dim = EvaluationDimension(ds["dimension"])
                    dimension_scores.append(
                        DimensionScore(
                            dimension=dim,
                            score=ds["score"],
                            explanation=ds.get("explanation", ""),
                        )
                    )
                except (ValueError, KeyError):
                    continue

            return TeacherEvaluation(
                overall_score=data["overall_score"],
                dimension_scores=dimension_scores,
                reasoning=data.get("reasoning", ""),
                improved_response=data.get("improved_response"),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                suggestions=data.get("suggestions", []),
            )
        except (json.JSONDecodeError, KeyError):
            return TeacherEvaluation(
                overall_score=5.0,
                dimension_scores=[],
                reasoning=response.choices[0].message.content,
            )
