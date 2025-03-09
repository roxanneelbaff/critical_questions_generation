from abc import ABC
import dataclasses
import json
from typing import Optional
from langchain_openai import ChatOpenAI
import pandas as pd
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from pyparsing import abstractmethod
import tqdm

from .objects import (
    Confirmation,
    CriticalQuestionList,
    SocialAgentAnswer,
    SocialAgentState,
)
from .state import BasicState
from .utils import get_st_data, timer


# BUILDING GRAPH - 1 class per AGENTS #


@dataclasses.dataclass
class CQSTAbstractAgent(ABC):
    llm_name: str
    llm_num: int = 1
    experiment_name: Optional[str] = None
    temperature: Optional[float] = None

    # DO NOT SET
    out: json = None
    graph: StateGraph = None
    time_log: pd.DataFrame = None
    llm_lst: list = None

    def __post_init__(self):
        print(f"Initializing LLM {self.llm_name}")
        self.llm_lst = [
            CQSTAbstractAgent._init_llm(self.llm_name, self.temperature)
            for _ in range(self.llm_num)
        ]
        print("Building Agentic Graph")
        self.graph = self.build_agent()
        print(
            "run `display(Image(graph.get_graph(xray=1).draw_mermaid_png()))` to see your Agentic Graph"
        )
        self.experiment_name = (
            self.experiment_name
            if self.experiment_name is not None
            else f"{self.llm_name}_temperature{self.temperature}_{self.__class__.__name__.lower()}"
        )
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

    def _invoke_graph(self, params, thread=None):
        questions = []
        for event in self.graph.stream(params, thread, stream_mode="values"):
            # Review
            _labels = event.get("final_cq", "")
            if _labels:
                questions.append(_labels)
        assert len(questions) == 1
        return questions[0]

    def run_experiment(self, data_type: str = "validation", save: bool = True):

        out = {}
        time_in_seconds_arr = []
        for _, line in tqdm.tqdm(get_st_data(data_type).items()):
            with timer(f"Iteration {line['intervention_id']}", time_in_seconds_arr):
                input_arg = line["intervention"]
                cqs = self._invoke_graph({"input_arg": input_arg}).model_dump()[
                    "critical_questions"
                ]

                # postprocessing data: replacing critical_question with cq to match the ST format
                for e in cqs:
                    e["cq"] = e.pop("critical_question")
                line["cqs"] = cqs

                out[line["intervention_id"]] = line
        if save:
            with open(f"output/output_{self.experiment_name}.json", "w") as o:
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
            structured_llm = self.llm_lst[0].with_structured_output(
                CriticalQuestionList
            )
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


class SocialAgentBuilder(CQSTAbstractAgent):
    collaborative_strategy: Optional[list[str]] = []  # ["debate", "reflect"]
    agent_trait_lst: list[str] = ["easy_going"]  # , "overconfident", "easy_going"]

    def __init__(
        self,
        llm_name: str,
        llm_num: int = 1,
        experiment_name: Optional[str] = None,
        temperature: Optional[float] = None,
        collaborative_strategy: Optional[list[str]] = [],
        agent_trait_lst: list[str] = ["easy_going"]
    ):
        self.collaborative_strategy = collaborative_strategy if collaborative_strategy is not None else []
        self.agent_trait_lst = agent_trait_lst if agent_trait_lst is not None else []
        super().__init__(
            llm_name=llm_name,
            llm_num=llm_num,
            experiment_name=experiment_name,
            temperature=temperature,
        )

    def build_agent(self) -> StateGraph:
        # llm_num = len(self.agent_trait_lst)
        print("BUILDING")
        validator_llm = CQSTAbstractAgent._init_llm(self.llm_name, self.temperature)

        def llm_role_node(state: SocialAgentState, node_id: int, trait: str):
            with open(f"prompts/trait_{trait}.txt", "r") as f:
                trait_prompt = f.read()

            response = (
                self.llm_lst[node_id]
                .with_structured_output(Confirmation)
                .invoke(
                    [SystemMessage(trait_prompt)],
                    [HumanMessage("If you understand your role, please say ok only.")],
                )
            )
            if response.confirmation.lower() == "ok":
                print(f"role {trait} confirmed")

                return {"roles_confirmed": [response.confirmation.lower() == "ok"]}

        def question_node(state: SocialAgentState, node_id: int, trait: str):
            round_answer_dict = {}

            with open(f"prompts/trait_{trait}.txt", "r") as f:
                system_prompt = f.read()
            with open("prompts/question.txt", "r") as file:
                instructions = file.read()

            instructions = instructions.format(
                input_arg=state["input_arg"],
            )
            response = (
                self.llm_lst[node_id]
                .with_structured_output(CriticalQuestionList)
                .invoke([SystemMessage(system_prompt)] + [HumanMessage(instructions)])
            )

            answer: SocialAgentAnswer = SocialAgentAnswer(
                critical_question_list=response,
                question_type="question",
                prompt=instructions,
            )
            round_answer_dict[f"agent{node_id}"] = [answer]

            return {"round_answer_dict": round_answer_dict}

        def debate_node(state: SocialAgentState, node_id: int, trait: str):
            answer_round = {}
            last_answers = [
                state["round_answer_dict"][f"agent{x}"][state["current_round"]]
                for x in range(self.llm_num)
            ]
            with open(f"prompts/trait_{trait}.txt", "r") as f:
                system_prompt = f.read()
            with open("prompts/strategy_debate.txt", "r") as file:
                instructions = file.read()

            # Get others answer and own answer

            other_agents_response_str = ""
            own_answer_str = ""
            cq_str = "\n- critical question {id}: '{cq}', reasoning: '{reason}'.\n"
            for a_num, other_ in enumerate(last_answers):
                if a_num != node_id:
                    other_agents_response_str += f"## Agent{a_num+1}:"
                    for cq in other_.critical_question_list.critical_questions:
                        if a_num != node_id:
                            other_agents_response_str = (
                                other_agents_response_str
                                + cq_str.format(
                                    id=cq.id, cq=cq.critical_question, reason=cq.reason
                                )
                            )
                        else:
                            own_answer_str = own_answer_str + cq_str.format(
                                id=cq.id, cq=cq.critical_question, reason=cq.reason
                            )
                    if a_num != node_id:
                        other_agents_response_str += "\n"

                instructions = instructions.format(
                    input_arg=state["input_arg"],
                    own_answer=own_answer_str,
                    other_agents_response=other_agents_response_str,
                )

                response = (
                    self.llm_lst[node_id]
                    .with_structured_output(CriticalQuestionList)
                    .invoke(
                        [SystemMessage(system_prompt)] + [HumanMessage(instructions)]
                    )
                )
                answer: SocialAgentAnswer = SocialAgentAnswer(
                    critical_question_list=response,
                    question_type="debate",
                    prompt=instructions,
                )

                answer_round[f"agent{node_id}"] = [answer]
            return {"round_answer_dict": answer_round}

        def moderator_node(state: SocialAgentState):
            if not all(state["roles_confirmed"]):
                print("Role not confirmed")
                return Command(goto=END)
            else:
                print("role confirmed")

            next_round = (
                (state["current_round"] + 1) if "current_round" in state.keys() else -1
            )
            return {"current_round": next_round}

        def reflect_node(state: SocialAgentState, node_id: int, trait: str):
            round_answer_dict = {}
            with open(f"prompts/trait_{trait}.txt", "r") as f:
                system_prompt = f.read()
            with open("prompts/strategy_reflect.txt", "r") as file:
                instructions = file.read()

            # Get others answer
            previous_answer = state["round_answer_dict"][f"agent{node_id}"][-1]
            own_answer_str = ""
            cq_str = "\n- critical question {id}: '{cq}', reasoning: '{reason}'.\n"
            for cq in previous_answer.critical_question_list.critical_questions:
                own_answer_str = own_answer_str + cq_str.format(
                    id=cq.id, cq=cq.critical_question, reason=cq.reason
                )
            instructions = instructions.format(
                input_arg=state["input_arg"], own_answer=own_answer_str
            )

            response = (
                self.llm_lst[node_id]
                .with_structured_output(CriticalQuestionList)
                .invoke([SystemMessage(system_prompt)] + [HumanMessage(instructions)])
            )
            answer: SocialAgentAnswer = SocialAgentAnswer(
                critical_question_list=response,
                question_type="reflect",
                prompt=instructions,
            )

            round_answer_dict[f"agent{i}"] = [answer]
            return {"round_answer_dict": round_answer_dict}

        def validate_node(state: SocialAgentState):
            with open("prompts/validator.txt", "r") as file:
                instructions = file.read()
            # Get others answer
            round_answer_dict = state["round_answer_dict"]
            others_answers = [
                round_answer_dict[f"agent{x}"][-1] for x in range(self.llm_num)
            ]
            other_agents_response_str = ""
            other_cq_str = (
                "\n- critical question {id}: '{cq}'.\n"  # reasoning: '{reason}'
            )
            for a_num, other_ in enumerate(others_answers):
                other_agents_response_str += f"Agent{a_num+1}:"
                for cq in other_.critical_question_list.critical_questions:
                    other_agents_response_str = (
                        other_agents_response_str
                        + other_cq_str.format(
                            id=cq.id, cq=cq.critical_question, reason=cq.reason
                        )
                    )
                other_agents_response_str += "\n\n"
            instructions = instructions.format(
                input_arg=state["input_arg"],
                other_agents_response=other_agents_response_str,
            )
            response = validator_llm.with_structured_output(
                CriticalQuestionList
            ).invoke([HumanMessage(instructions)])
            print(instructions)
            return {"final_cq": response}

        workflow = StateGraph(SocialAgentState)

        for i, trait in enumerate(self.agent_trait_lst):
            workflow.add_node(
                f"llm_role_node{i+1}",
                lambda state, node_id=i, trait=trait: llm_role_node(
                    state, node_id, trait
                ),
            )

            workflow.add_node(
                f"question_node{i+1}",
                lambda state, node_id=i, trait=trait: question_node(
                    state, node_id, trait
                ),
            )
            if "debate" in self.collaborative_strategy:
                workflow.add_node(
                    f"debate_node{i+1}",
                    lambda state, node_id=i, trait=trait: debate_node(
                        state, node_id, trait
                    ),
                )
            if "reflect" in self.collaborative_strategy:
                workflow.add_node(
                    f"reflect_node{i+1}",
                    lambda state, node_id=i, trait=trait: reflect_node(
                        state, node_id, trait
                    ),
                )

        workflow.add_node(validate_node)
        workflow.add_node("moderator_role", moderator_node)

        for i in range(self.llm_num):
            workflow.add_edge(START, f"llm_role_node{i+1}")
            workflow.add_edge("moderator_role", f"question_node{i+1}")

        workflow.add_edge(
            [f"llm_role_node{i+1}" for i in range(self.llm_num)], "moderator_role"
        )

        prev_strategy = "question"
        for i, strategy in enumerate(self.collaborative_strategy):
            str_nodes = [
                f"{strategy}_node{i+1}" for i in range(self.llm_num)
            ]  # if i<(len(collaborative_strategy)-1) else ["validate_node"]
            curr_moderator = "moderator_question" if i == 0 else f"moderator_round{i}"

            workflow.add_node(curr_moderator, moderator_node)
            workflow.add_edge(
                [f"{prev_strategy}_node{i+1}" for i in range(self.llm_num)],
                curr_moderator,
            )
            for e in str_nodes:
                workflow.add_edge(curr_moderator, e)

            prev_strategy = strategy

            if i == len(self.collaborative_strategy) - 1:  # last
                curr_moderator = "moderator_final"
                workflow.add_node(curr_moderator, moderator_node)
                workflow.add_edge(
                    [f"{strategy}_node{i+1}" for i in range(self.llm_num)],
                    curr_moderator,
                )
                workflow.add_edge(curr_moderator, "validate_node")

        workflow.add_edge("validate_node", END)

        memory = MemorySaver()
        print("BUILDING DONE")
        return workflow.compile(checkpointer=memory)
