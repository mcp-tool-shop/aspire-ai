"""
Teacher registry - discover and instantiate teachers by name.

Enables dynamic teacher selection and custom teacher registration.
"""

from typing import Type

from aspire.teachers.base import BaseTeacher


class TeacherRegistry:
    """
    Registry for teacher classes.

    Allows registering custom teachers and retrieving them by name.
    """

    _teachers: dict[str, Type[BaseTeacher]] = {}

    @classmethod
    def register(cls, name: str, teacher_class: Type[BaseTeacher]) -> None:
        """Register a teacher class."""
        cls._teachers[name.lower()] = teacher_class

    @classmethod
    def get(cls, name: str) -> Type[BaseTeacher] | None:
        """Get a teacher class by name."""
        return cls._teachers.get(name.lower())

    @classmethod
    def list(cls) -> list[str]:
        """List all registered teacher names."""
        return list(cls._teachers.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseTeacher:
        """Create a teacher instance by name."""
        teacher_class = cls.get(name)
        if teacher_class is None:
            raise ValueError(f"Unknown teacher: {name}. Available: {cls.list()}")
        return teacher_class(**kwargs)


def register_teacher(name: str):
    """Decorator to register a teacher class."""

    def decorator(cls: Type[BaseTeacher]) -> Type[BaseTeacher]:
        TeacherRegistry.register(name, cls)
        return cls

    return decorator


def get_teacher(name: str, **kwargs) -> BaseTeacher:
    """Get a teacher instance by name."""
    return TeacherRegistry.create(name, **kwargs)


# Register built-in teachers
def _register_builtin_teachers():
    """Register all built-in teachers."""
    from aspire.teachers.claude import ClaudeTeacher
    from aspire.teachers.openai import OpenAITeacher
    from aspire.teachers.local import LocalTeacher
    from aspire.teachers.personas import (
        SocraticTeacher,
        ScientificTeacher,
        CreativeTeacher,
        AdversarialTeacher,
        CompassionateTeacher,
    )

    TeacherRegistry.register("claude", ClaudeTeacher)
    TeacherRegistry.register("openai", OpenAITeacher)
    TeacherRegistry.register("gpt4", OpenAITeacher)
    TeacherRegistry.register("local", LocalTeacher)

    # Personas
    TeacherRegistry.register("socratic", SocraticTeacher)
    TeacherRegistry.register("scientific", ScientificTeacher)
    TeacherRegistry.register("creative", CreativeTeacher)
    TeacherRegistry.register("adversarial", AdversarialTeacher)
    TeacherRegistry.register("compassionate", CompassionateTeacher)

    # Aliases
    TeacherRegistry.register("socrates", SocraticTeacher)
    TeacherRegistry.register("scientist", ScientificTeacher)
    TeacherRegistry.register("innovator", CreativeTeacher)
    TeacherRegistry.register("challenger", AdversarialTeacher)
    TeacherRegistry.register("guide", CompassionateTeacher)


# Auto-register on import
_register_builtin_teachers()
