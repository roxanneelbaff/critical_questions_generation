
import operator
from typing import TypedDict
from typing_extensions import Annotated

from social_agents import utils
from .data_model import CriticalQuestionList, SocialAgentAnswer


# state
class BasicState(TypedDict):
    input_arg: str

    # OutPut
    final_c: CriticalQuestionList


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
    validation_instruction: str
