"""
Teacher models for ASPIRE training.

Teachers provide wisdom, judgment, and adversarial challenge to the student.
Different teachers produce different learning outcomes - a Socratic philosopher
teaches differently than a rigorous scientist or a creative artist.
"""

from aspire.teachers.base import (
    BaseTeacher,
    TeacherEvaluation,
    TeacherChallenge,
    DialogueTurn,
)
from aspire.teachers.claude import ClaudeTeacher
from aspire.teachers.openai import OpenAITeacher
from aspire.teachers.local import LocalTeacher
from aspire.teachers.composite import CompositeTeacher
from aspire.teachers.personas import (
    SocraticTeacher,
    ScientificTeacher,
    CreativeTeacher,
    AdversarialTeacher,
    CompassionateTeacher,
)
from aspire.teachers.registry import TeacherRegistry, get_teacher, register_teacher

__all__ = [
    # Base
    "BaseTeacher",
    "TeacherEvaluation",
    "TeacherChallenge",
    "DialogueTurn",
    # Implementations
    "ClaudeTeacher",
    "OpenAITeacher",
    "LocalTeacher",
    "CompositeTeacher",
    # Personas
    "SocraticTeacher",
    "ScientificTeacher",
    "CreativeTeacher",
    "AdversarialTeacher",
    "CompassionateTeacher",
    # Registry
    "TeacherRegistry",
    "get_teacher",
    "register_teacher",
]
