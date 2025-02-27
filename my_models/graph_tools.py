from typing import List, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage

from my_models.objects import CriticalQuestionList
from my_models.state import BasicState



def build_zero_shot_graph(llm):
    def generate_critical_questions(state: BasicState):
        structured_llm = llm.with_structured_output(CriticalQuestionList)
        with open("prompts/system.txt", "r") as f:
            system_prompt = f.read()
        with open("prompts/basic.txt", "r") as file:
            instructions = file.read()

        instructions = instructions.format(
            input_arg=state["input_arg"],
        )

        response = structured_llm.invoke(
            [SystemMessage(content=system_prompt)]
            + [HumanMessage(content=instructions)]
        )

        return {"critical_question_list": response}

    builder = StateGraph(BasicState)
    builder.add_node("generate_critical_questions", generate_critical_questions)
    builder.add_edge(START, "generate_critical_questions")
    builder.add_edge("generate_critical_questions", END)

    graph = builder.compile()
    return graph

def evaluate(data_split: str = "validation"): 