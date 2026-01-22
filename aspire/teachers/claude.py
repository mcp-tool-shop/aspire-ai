"""
Claude-based teacher implementation.
"""

import json
from typing import Any

import anthropic

from aspire.teachers.base import (
    BaseTeacher,
    ChallengeType,
    DialogueHistory,
    DimensionScore,
    EvaluationDimension,
    TeacherChallenge,
    TeacherEvaluation,
)


class ClaudeTeacher(BaseTeacher):
    """
    Teacher powered by Claude API.

    Claude excels at nuanced evaluation, Socratic questioning, and
    generating thoughtful feedback. This is the recommended default teacher.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        name: str = "Claude Teacher",
        description: str = "A thoughtful, nuanced teacher powered by Claude",
        **kwargs,
    ):
        super().__init__(name=name, description=description, **kwargs)
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def challenge(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        challenge_type: ChallengeType | None = None,
    ) -> TeacherChallenge:
        """Generate a challenge using Claude."""

        if challenge_type is None:
            challenge_type = self.select_challenge_type(dialogue_history)

        # Build context from history
        history_context = ""
        if dialogue_history and dialogue_history.turns:
            history_context = "\n\nPrevious dialogue:\n"
            for turn in dialogue_history.turns:
                history_context += f"Challenge: {turn.challenge.content}\n"
                history_context += f"Student: {turn.student_response}\n\n"

        challenge_prompt = f"""You are generating a {challenge_type.value} challenge for a student.

Original prompt: {prompt}

Student's response: {student_response}
{history_context}

Generate a {challenge_type.value} challenge. This means:
{self._get_challenge_description(challenge_type)}

Respond with JSON:
{{
    "challenge": "Your challenge question or statement",
    "context": "Brief explanation of why you chose this challenge",
    "difficulty": 0.5  // 0.0 to 1.0
}}"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": challenge_prompt}],
        )

        # Parse response
        try:
            content = response.content[0].text
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)

            return TeacherChallenge(
                challenge_type=challenge_type,
                content=data["challenge"],
                context=data.get("context"),
                difficulty=data.get("difficulty", 0.5),
            )
        except (json.JSONDecodeError, KeyError, IndexError):
            # Fallback: use raw response as challenge
            return TeacherChallenge(
                challenge_type=challenge_type,
                content=response.content[0].text,
                difficulty=0.5,
            )

    async def evaluate(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        generate_improved: bool = True,
    ) -> TeacherEvaluation:
        """Evaluate a student response using Claude."""

        # Build context from history
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

Evaluate on these dimensions: {dimensions_list}

{"Also provide an improved version of the response." if generate_improved else ""}

Respond with JSON:
{{
    "overall_score": 7.5,  // 0-10
    "dimension_scores": [
        {{"dimension": "correctness", "score": 8.0, "explanation": "..."}},
        // ... for each dimension
    ],
    "reasoning": "Overall evaluation reasoning",
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "suggestions": ["suggestion 1", "suggestion 2"],
    {"\"improved_response\": \"Better version of the response\"" if generate_improved else ""}
}}"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens * 2,  # More tokens for detailed eval
            temperature=0.3,  # Lower temperature for consistent evaluation
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": eval_prompt}],
        )

        # Parse response
        try:
            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)

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
            # Fallback: return basic evaluation
            return TeacherEvaluation(
                overall_score=5.0,
                dimension_scores=[],
                reasoning=response.content[0].text,
            )

    def _get_challenge_description(self, challenge_type: ChallengeType) -> str:
        """Get description of what each challenge type means."""
        descriptions = {
            ChallengeType.PROBE_REASONING: "Ask 'why' or 'how' to probe the reasoning behind the answer",
            ChallengeType.EDGE_CASE: "Present an edge case or unusual scenario that tests the limits",
            ChallengeType.DEVILS_ADVOCATE: "Argue the opposite position to test conviction and reasoning",
            ChallengeType.SOCRATIC: "Ask questions that reveal hidden assumptions or gaps in logic",
            ChallengeType.CLARIFICATION: "Ask for clarification on vague or ambiguous points",
            ChallengeType.EXTENSION: "Ask how the reasoning extends to related domains or scenarios",
            ChallengeType.CONTRADICTION: "Point out apparent contradictions or inconsistencies",
            ChallengeType.STEELMAN: "Ask for the strongest counter-argument to their position",
            ChallengeType.EMOTIONAL: "Explore emotional or human impact dimensions",
            ChallengeType.PRACTICAL: "Ask about practical implementation or real-world application",
            ChallengeType.ETHICAL: "Explore ethical implications or moral dimensions",
            ChallengeType.CREATIVE: "Encourage alternative approaches or novel perspectives",
        }
        return descriptions.get(challenge_type, "Ask a challenging follow-up question")
