import operator
from typing import Annotated, List, Literal, TypedDict
from pydantic import BaseModel, Field

from social_agents import utils


class CriticalQuestion(BaseModel):
    id: int = Field(
        description="The numeric rank of the generated question, starting with 0. The values reflect the usefulness of the question, where 0 is the most useful."
    )
    critical_question: str = Field(
        description="A critical question that reflects if an argument or a statement is *acceptable* or *fallacious*. The question unmasks the assumptions held by the premises of the given argument and attacks its inference."
    )
    reason: str = Field(description="The reason behind asking the critical question")


class CriticalQuestionList(BaseModel):
    critical_questions: List[CriticalQuestion] = Field(
        description="The list of all the critical questions and their ranks to criticize and reveal the weaknesses of an argument."
    )


class Confirmation(BaseModel):
    confirmation: str = Field(
        description="The LLM confirms the role assigned to it by simply saying ok."
    )


class SocialAgentAnswer(BaseModel):
    critical_question_list: CriticalQuestionList = Field(
        description="The list of all the critical questions and their ranks to criticize and reveal the weaknesses of an argument."
    )
    question_type: Literal["debate", "reflect", "question"]
    prompt: str


class SocialAgentState(TypedDict):
    # INPUT
    input_arg: str
    collaborative_strategy: list

    # Runtime vars
    roles_confirmed: Annotated[list, operator.add]
    round_answer_dict: Annotated[dict[str, list[SocialAgentAnswer]],
                                 utils.dict_reducer]
    current_round: int = -2

    # OUTPUT
    final_cq: CriticalQuestionList
