from abc import ABC
import dataclasses
import json
from typing import Optional
from langchain_openai import ChatOpenAI
import pandas as pd
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from pyparsing import abstractmethod
import tqdm

from .objects import CriticalQuestionList
from .state import BasicState
from .utils import get_st_data, timer


# BUILDING GRAPH - 1 class per AGENTS #

@dataclasses.dataclass
class CQSTAbstractAgent(ABC):
    llm_name: str
    experiment_name: Optional[str] = None
    temperature: Optional[float] = None

    # DO NOT SET
    out: json = None
    graph: StateGraph = None
    time_log: pd.DataFrame = None

    def __post_init__(self):
        print(f"Initializing LLM {self.llm_name}")
        self.llm = ChatOpenAI(model=self.llm_name,
                              temperature=self.temperature)
        print("Building Agentic Graph")
        self.graph = self.build_agent()
        print("run `display(Image(graph.get_graph(xray=1).draw_mermaid_png()))` to see your Agentic Graph")
        self.experiment_name = f"{self.llm_name}_temperature{self.temperature}_{self.__class__.__name__.lower()}"
        print("experiment name: ", self.experiment_name)

    @staticmethod
    def _init_llm(llm_name: str, temperature: int = 0):
        if llm_name == "o3-mini-2025-01-31":
            return ChatOpenAI(model=llm_name)
        elif llm_name.startswith("gpt"):
            return ChatOpenAI(model=llm_name, temperature=temperature)

    @abstractmethod
    def build_agent(self) -> StateGraph:
        pass

    def _invoke_graph(self, params, thread= None):
        questions = []
        for event in self.graph.stream(params, thread, stream_mode="values"):
            # Review
            _labels = event.get('critical_question_list', '')
            if _labels:
                questions.append(_labels)
        assert len(questions) == 1
        return questions[0]

    def run_experiment(self, data_type: str = "validation", save: bool = True):

        out = {}
        time_in_seconds_arr = []
        for _, line in tqdm.tqdm(get_st_data(data_type).items()):
            with timer(f"Iteration {line['intervention_id']}",
                       time_in_seconds_arr):
                input_arg = line['intervention']
                cqs = self._invoke_graph({"input_arg": input_arg}).model_dump()['critical_questions']        

                # postprocessing data: replacing critical_question with cq to match the ST format
                for e in cqs:
                    e["cq"] = e.pop("critical_question")
                line['cqs'] = cqs

                out[line['intervention_id']] = line
        if save:
            with open(f'output/output_{self.experiment_name}.json', 'w') as o:
                json.dump(out, o, indent=4)
        # Log TIME
        time_log_df = pd.read_csv("output/time_log.csv")
        time_log_df[f"time_{self.experiment_name}"] = time_in_seconds_arr
        time_log_df.to_csv("output/time_log.csv", index=False)
        self.time_log = time_log_df

        self.out = out
        return out


# ZERO SHOT MODEL
@dataclasses.dataclass
class BasicCQModel(CQSTAbstractAgent):
    def build_agent(self):
        # Define all the logic of the Graph here
        def generate_critical_questions(state: BasicState):
            structured_llm = self.llm.with_structured_output(CriticalQuestionList)
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
        builder.add_node("generate_critical_questions",
                         generate_critical_questions)
        builder.add_edge(START, "generate_critical_questions")
        builder.add_edge("generate_critical_questions", END)

        graph = builder.compile()
        return graph




































