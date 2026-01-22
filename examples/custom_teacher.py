"""
Example: Creating a custom teacher persona.

Teachers embody different philosophies. This example shows how to
create your own teacher with a unique perspective.
"""

from aspire.teachers.claude import ClaudeTeacher
from aspire.teachers.base import ChallengeType, EvaluationDimension
from aspire.teachers.registry import register_teacher


@register_teacher("stoic")
class StoicTeacher(ClaudeTeacher):
    """
    A teacher inspired by Stoic philosophy.

    Focuses on what can be controlled, emotional regulation,
    and practical wisdom. Challenges students to separate
    facts from judgments.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="Marcus Aurelius",
            description="""A Stoic philosopher-teacher who emphasizes
the distinction between what we can control and what we cannot.
Teaches practical wisdom, emotional regulation, and clear thinking
unclouded by passion or fear.""",
            preferred_challenges=[
                ChallengeType.SOCRATIC,
                ChallengeType.PRACTICAL,
                ChallengeType.EMOTIONAL,
                ChallengeType.ETHICAL,
            ],
            evaluation_dimensions=[
                EvaluationDimension.REASONING,
                EvaluationDimension.INTELLECTUAL_HONESTY,
                EvaluationDimension.PRACTICALITY,
                EvaluationDimension.CLARITY,
            ],
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return """You are Marcus Aurelius, a Stoic philosopher and teacher.

Your principles:
- Distinguish between what is in our control (judgments, impulses, desires)
  and what is not (external events, others' actions, outcomes)
- Facts are neutral; our judgments about them create suffering or peace
- Focus on action within one's sphere of control
- Emotional reactions often reveal confused thinking
- Practical wisdom matters more than theoretical knowledge

Your method:
- When a student worries about outcomes, ask: "What here is within your control?"
- When they express strong emotion, ask: "What judgment underlies this feeling?"
- Challenge them to find what action they can take, regardless of circumstances
- Encourage clear-eyed acceptance of what cannot be changed
- Value practical application over abstract understanding

Your tone is calm, measured, and grounded. You do not dismiss emotions,
but you help students examine the thoughts behind them.

Remember: "The happiness of your life depends upon the quality of your thoughts."
"""


# Example usage
if __name__ == "__main__":
    import asyncio
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from aspire.dialogue import DialogueGenerator

    async def main():
        print("Loading student model...")
        model_name = "microsoft/Phi-3-mini-4k-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        student = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
        )

        # Use our custom Stoic teacher
        teacher = StoicTeacher()

        generator = DialogueGenerator(
            student_model=student,
            student_tokenizer=tokenizer,
            teacher=teacher,
            max_turns=3,
        )

        # A question the Stoics would appreciate
        prompt = "How should I deal with failure and setbacks?"

        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Teacher: {teacher.name} (Stoic)")
        print(f"{'='*60}\n")

        dialogue = await generator.generate_dialogue(prompt)

        print(f"Initial Response:\n{dialogue.initial_response}\n")

        for i, turn in enumerate(dialogue.history.turns, 1):
            print(f"--- Turn {i} ---")
            print(f"Marcus Aurelius: {turn.challenge.content}\n")
            print(f"Student: {turn.student_response}\n")

        print(f"Final Score: {dialogue.final_evaluation.overall_score:.1f}/10")
        print(f"\nStoic Assessment:\n{dialogue.final_evaluation.reasoning}")

    from multiprocessing import freeze_support
    freeze_support()
    asyncio.run(main())
