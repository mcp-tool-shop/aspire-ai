"""
Dialogue manager - orchestrates dialogue generation and caching.
"""

import json
import hashlib
from pathlib import Path
from typing import Iterator
from dataclasses import asdict

from aspire.dialogue.generator import DialogueGenerator, GeneratedDialogue
from aspire.teachers.base import DialogueHistory, TeacherEvaluation


class DialogueManager:
    """
    Manages dialogue generation, caching, and retrieval.

    Handles:
    - Caching generated dialogues to avoid redundant API calls
    - Batching prompts for efficient generation
    - Loading/saving dialogue datasets
    """

    def __init__(
        self,
        generator: DialogueGenerator,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ):
        self.generator = generator
        self.cache_dir = cache_dir or Path("dialogue_cache")
        self.use_cache = use_cache

        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for a prompt."""
        # Include teacher name in key since different teachers = different dialogues
        key_string = f"{self.generator.teacher.name}:{prompt}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, prompt: str) -> Path:
        """Get cache file path for a prompt."""
        return self.cache_dir / f"{self._get_cache_key(prompt)}.json"

    def _load_from_cache(self, prompt: str) -> GeneratedDialogue | None:
        """Load dialogue from cache if available."""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(prompt)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)

            # Reconstruct dialogue
            history = DialogueHistory(
                prompt=data["prompt"],
                initial_response=data["initial_response"],
            )

            # Reconstruct turns (simplified - full reconstruction would need more)
            # For now, just return the cached evaluation data
            final_eval = TeacherEvaluation(
                overall_score=data["final_evaluation"]["overall_score"],
                dimension_scores=[],
                reasoning=data["final_evaluation"]["reasoning"],
                improved_response=data["final_evaluation"].get("improved_response"),
            )

            return GeneratedDialogue(
                prompt=data["prompt"],
                initial_response=data["initial_response"],
                history=history,
                final_evaluation=final_eval,
                turn_evaluations=[],
                metadata=data.get("metadata", {}),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _save_to_cache(self, dialogue: GeneratedDialogue) -> None:
        """Save dialogue to cache."""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(dialogue.prompt)

        # Serialize dialogue
        data = {
            "prompt": dialogue.prompt,
            "initial_response": dialogue.initial_response,
            "final_evaluation": dialogue.final_evaluation.to_dict(),
            "metadata": dialogue.metadata,
            "turns": [
                {
                    "challenge": turn.challenge.content,
                    "response": turn.student_response,
                    "evaluation": turn.evaluation.to_dict() if turn.evaluation else None,
                }
                for turn in dialogue.history.turns
            ],
        }

        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

    async def get_dialogue(
        self,
        prompt: str,
        force_regenerate: bool = False,
    ) -> GeneratedDialogue:
        """
        Get dialogue for a prompt, using cache if available.

        Args:
            prompt: The prompt to generate dialogue for
            force_regenerate: If True, ignore cache and regenerate

        Returns:
            Generated dialogue
        """
        # Check cache first
        if not force_regenerate:
            cached = self._load_from_cache(prompt)
            if cached is not None:
                return cached

        # Generate new dialogue
        dialogue = await self.generator.generate_dialogue(prompt)

        # Cache it
        self._save_to_cache(dialogue)

        return dialogue

    async def get_dialogues(
        self,
        prompts: list[str],
        force_regenerate: bool = False,
        max_concurrent: int = 5,
    ) -> list[GeneratedDialogue]:
        """
        Get dialogues for multiple prompts.

        Uses cache where available, generates rest in parallel.
        """
        results = [None] * len(prompts)
        to_generate = []
        generate_indices = []

        # Check cache for each prompt
        for i, prompt in enumerate(prompts):
            if not force_regenerate:
                cached = self._load_from_cache(prompt)
                if cached is not None:
                    results[i] = cached
                    continue

            to_generate.append(prompt)
            generate_indices.append(i)

        # Generate missing dialogues
        if to_generate:
            generated = await self.generator.generate_batch(
                to_generate,
                max_concurrent=max_concurrent,
            )

            # Fill in results and cache
            for idx, dialogue in zip(generate_indices, generated):
                results[idx] = dialogue
                self._save_to_cache(dialogue)

        return results

    def iterate_cached(self) -> Iterator[GeneratedDialogue]:
        """Iterate over all cached dialogues."""
        if not self.cache_dir.exists():
            return

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    data = json.load(f)

                # Reconstruct (simplified)
                history = DialogueHistory(
                    prompt=data["prompt"],
                    initial_response=data["initial_response"],
                )

                final_eval = TeacherEvaluation(
                    overall_score=data["final_evaluation"]["overall_score"],
                    dimension_scores=[],
                    reasoning=data["final_evaluation"]["reasoning"],
                    improved_response=data["final_evaluation"].get("improved_response"),
                )

                yield GeneratedDialogue(
                    prompt=data["prompt"],
                    initial_response=data["initial_response"],
                    history=history,
                    final_evaluation=final_eval,
                    turn_evaluations=[],
                    metadata=data.get("metadata", {}),
                )
            except (json.JSONDecodeError, KeyError):
                continue

    def clear_cache(self) -> int:
        """Clear all cached dialogues. Returns number cleared."""
        count = 0
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                count += 1
        return count

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        if not self.cache_dir.exists():
            return {"count": 0, "size_bytes": 0}

        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            "count": len(files),
            "size_bytes": total_size,
            "size_mb": total_size / (1024 * 1024),
        }
