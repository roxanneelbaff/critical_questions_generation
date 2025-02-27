
from typing import TypedDict
from .objects import CriticalQuestionList


# state
class BasicState(TypedDict):
    input_arg: str

    # OutPut
    critical_question_list: CriticalQuestionList
