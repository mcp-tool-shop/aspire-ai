"""
Example: Using composite teachers for multi-perspective learning.

Different teachers produce different minds. Combining them creates
richer, more balanced learning experiences.
"""

import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

from aspire.teachers import (
    CompositeTeacher,
    SocraticTeacher,
    ScientificTeacher,
    AdversarialTeacher,
)
from aspire.dialogue import DialogueGenerator


async def main():
    # Load student model
    print("Loading student model...")
    model_name = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    student = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    # Create a committee of teachers
    # Strategy options:
    #   - "rotate": Different teacher each turn
    #   - "vote": All teachers evaluate, combine scores
    #   - "random": Randomly select teacher weighted by importance

    composite_teacher = CompositeTeacher(
        teachers=[
            SocraticTeacher(),      # Probes reasoning with questions
            ScientificTeacher(),    # Demands evidence and rigor
            AdversarialTeacher(),   # Challenges and stress-tests
        ],
        strategy="rotate",
        name="The Council",
    )

    # Generate dialogue
    generator = DialogueGenerator(
        student_model=student,
        student_tokenizer=tokenizer,
        teacher=composite_teacher,
        max_turns=3,
    )

    prompt = "Is it ever ethical to lie?"

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Teachers: Socratic → Scientific → Adversarial (rotating)")
    print(f"{'='*60}\n")

    dialogue = await generator.generate_dialogue(prompt)

    print(f"Initial Response:\n{dialogue.initial_response}\n")

    for i, turn in enumerate(dialogue.history.turns, 1):
        print(f"--- Turn {i} ---")
        print(f"Challenge ({turn.challenge.challenge_type.value}):")
        print(f"  {turn.challenge.content}\n")
        print(f"Response:")
        print(f"  {turn.student_response}\n")

    print(f"Final Score: {dialogue.final_evaluation.overall_score:.1f}/10")
    print(f"\nCombined Reasoning:\n{dialogue.final_evaluation.reasoning}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    asyncio.run(main())
