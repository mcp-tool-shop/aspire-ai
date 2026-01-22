"""
Basic example: Generate an adversarial dialogue with a Socratic teacher.

This demonstrates the core ASPIRE concept - a student model being challenged
by a wise teacher to develop deeper understanding.
"""

import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

from aspire.teachers import get_teacher
from aspire.dialogue import DialogueGenerator


async def main():
    # 1. Load a student model (any causal LM)
    print("Loading student model...")
    model_name = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    student = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    # 2. Create a teacher with a specific persona
    # Options: "socratic", "scientific", "creative", "adversarial", "compassionate"
    teacher = get_teacher("socratic")

    # 3. Create the dialogue generator
    generator = DialogueGenerator(
        student_model=student,
        student_tokenizer=tokenizer,
        teacher=teacher,
        max_turns=3,  # Number of challenge-response rounds
        evaluate_each_turn=True,
    )

    # 4. Generate an adversarial dialogue
    prompt = "Explain why neural networks need activation functions."

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Teacher: {teacher.name}")
    print(f"{'='*60}\n")

    dialogue = await generator.generate_dialogue(prompt)

    # 5. Display the results
    print(f"Initial Response:\n{dialogue.initial_response}\n")

    for i, turn in enumerate(dialogue.history.turns, 1):
        print(f"--- Turn {i} ---")
        print(f"Challenge ({turn.challenge.challenge_type.value}):")
        print(f"  {turn.challenge.content}\n")
        print(f"Student Response:")
        print(f"  {turn.student_response}\n")

        if turn.evaluation:
            print(f"  [Score: {turn.evaluation.overall_score:.1f}/10]\n")

    print(f"{'='*60}")
    print(f"Final Evaluation")
    print(f"{'='*60}")
    print(f"Score: {dialogue.final_evaluation.overall_score:.1f}/10")
    print(f"\nReasoning:\n{dialogue.final_evaluation.reasoning}")

    if dialogue.final_evaluation.improved_response:
        print(f"\nImproved Response:\n{dialogue.final_evaluation.improved_response}")


if __name__ == "__main__":
    # Windows compatibility
    from multiprocessing import freeze_support
    freeze_support()

    asyncio.run(main())
