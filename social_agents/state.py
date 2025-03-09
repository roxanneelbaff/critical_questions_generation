
from typing import TypedDict
from .objects import CriticalQuestionList


# state
class BasicState(TypedDict):
    input_arg: str

    # OutPut
    final_cq: CriticalQuestionList
