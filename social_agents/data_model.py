from typing_extensions import  List, Literal
from pydantic import BaseModel, Field


class CriticalQuestion(BaseModel):
    """A critical question that challenges an argument, with the ranking id and a very brief reasoning"""
    id: int = Field(
        description="The numeric rank of the generated question, starting with 0. The values reflect the usefulness of the question, where 0 is the most useful."
    )
    critical_question: str = Field(
        description="A critical question to challenge and argument's *acceptability* or *fallaciousness*."
    )
    reason: str = Field(description="Very brief reasoning for the critical question")


class CriticalQuestionList(BaseModel):
    """The list of the three critical questions generated"""
    critical_questions: List[CriticalQuestion] = Field(
        description="The list of all the critical questions."
    )


class Confirmation(BaseModel):
    """Confirmation that the model confirm their role"""

    confirmation: str = Field(
        description="The LLM confirms the role assigned to it by saying 'ok'."
    )


class SocialAgentAnswer(BaseModel):
    critical_question_list: CriticalQuestionList = Field(
        description="The list of all the critical questions and their ranks to criticize and reveal the weaknesses of an argument."
    )
    question_type: Literal["debate", "reflect", "question", "validate"]
    prompt: dict
    trait: str
