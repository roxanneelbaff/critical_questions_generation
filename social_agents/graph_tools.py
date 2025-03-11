from abc import ABC
import dataclasses
import itertools
import json
from typing import ClassVar, Counter, Optional, Union
import uuid
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
from social_agents import objects


# BUILDING GRAPH - 1 class per AGENTS #


@dataclasses.dataclass
class CQSTAbstractAgent(ABC):
    model_thread_id: str
    llm_name: str
    llm_num: int = 1
    experiment_name: Optional[str] = None
    temperature: Optional[float] = None

    serialize_func: callable = None

    # DO NOT SET
    out: json = None
    graph: StateGraph = None
    time_log: pd.DataFrame = None
    llm_lst: list = None

    ROOT_FOLDER: ClassVar[str] = "output/elbaff_experiment/"

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

    def _invoke_graph(self, params, id_):
        questions = []
        config = {"configurable": {"thread_id": f"{self.model_thread_id}_{id_}"}}
        for event in self.graph.stream(params, config, stream_mode="values"):
            # Review
            _labels = event.get("final_cq", "")
            if _labels:
                questions.append(_labels)
                # Save the final state
                fname = f"{CQSTAbstractAgent.ROOT_FOLDER}final_states/{self.experiment_name}_arg{id_}.json"
                with open(
                    fname,
                    "w",
                ) as f:
                    try:
                        if self.serialize_func is None:
                            json.dump(dict(event), f, indent=2)
                        else:
                            json.dump(self.serialize_func(event), f, indent=2)
                    except Exception:
                        print("error whiles saving file: ", fname)
    
        assert len(questions) == 1
        return questions[0]

    def run_experiment(self, data_type: str = "validation", save: bool = True):
        out = {}
        time_in_seconds_arr = []
        for _, line in tqdm.tqdm(get_st_data(data_type).items()):
            with timer(f"Iteration {line['intervention_id']}", time_in_seconds_arr):
                input_arg = (
                    line["intervention"].encode("utf-8").decode("unicode-escape")
                )
                cqs = self._invoke_graph(
                    {"input_arg": input_arg}, line["intervention_id"]
                ).model_dump()["critical_questions"]

                # postprocessing data: replacing critical_question with cq to match the ST format
                for e in cqs:
                    e["cq"] = e.pop("critical_question")
                line["cqs"] = cqs

                out[line["intervention_id"]] = line
        if save:
            with open(f"{CQSTAbstractAgent.ROOT_FOLDER}output_{self.experiment_name}.json", "w") as o:
                json.dump(out, o, indent=4)
        # Log TIME
        time_log_df = pd.read_csv(f"{CQSTAbstractAgent.ROOT_FOLDER}time_log.csv")
        time_log_df[f"{self.experiment_name}"] = time_in_seconds_arr
        time_log_df.to_csv(f"{CQSTAbstractAgent.ROOT_FOLDER}time_log.csv", index=False)
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

    collaborative_strategy: Optional[list[str]] = []
    agent_trait_lst: list[str] = ["easy_going"]

    def __init__(
        self,
        model_thread_id: str,
        llm_name: str,
        llm_num: int = 1,
        experiment_name: Optional[str] = None,
        temperature: Optional[float] = None,
        serialize_func: callable = objects.state_to_serializable,
        collaborative_strategy: Optional[list[str]] = [],
        agent_trait_lst: list[str] = ["easy_going"],
    ):
        self.collaborative_strategy = (
            collaborative_strategy if collaborative_strategy is not None else []
        )
        self.agent_trait_lst = agent_trait_lst if agent_trait_lst is not None else []
        assert len(agent_trait_lst) > 0

        super().__init__(
            model_thread_id=model_thread_id,
            llm_name=llm_name,
            llm_num=llm_num,
            experiment_name=experiment_name,
            temperature=temperature,
            serialize_func=serialize_func,
        )

    def build_agent(self) -> StateGraph:
        # llm_num = len(self.agent_trait_lst)
        print("building workflow")
        validator_llm = CQSTAbstractAgent._init_llm(self.llm_name, self.temperature)

        def llm_role_node(state: SocialAgentState, node_id: int, trait: str):
            with open(f"prompts/trait_{trait}.txt", "r") as f:
                trait_prompt = f.read()

            response = (
                self.llm_lst[node_id]
                .with_structured_output(Confirmation)
                .invoke(
                    [
                        SystemMessage(trait_prompt),
                        HumanMessage(
                            "If you understand your role, please say ok only."
                        ),
                    ],
                )
            )
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
                prompt={"system": system_prompt,
                        "human": instructions},
                trait=trait,
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
                    other_agents_response_str += f"AGENT {a_num+1}:"
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
                prompt={"system": system_prompt,
                        "human": instructions},
                trait=trait,
            )
            answer_round[f"agent{node_id}"] = [answer]
            return {"round_answer_dict": answer_round}

        def moderator_node(state: SocialAgentState):
            if not all(state["roles_confirmed"]):
                print("Role not confirmed")
                return Command(goto=END)

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
            cq_str = "\n- critical question {id}: '{cq}', reasoning: '{reason}'."
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
                prompt={"system": system_prompt,
                        "human": instructions},
                trait=trait,
            )

            round_answer_dict[f"agent{node_id}"] = [answer]
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
                "\n- critical question {id}: '{cq}'."  # reasoning: '{reason}'
            )
            for a_num, other_ in enumerate(others_answers):
                for cq in other_.critical_question_list.critical_questions:
                    other_agents_response_str = (
                        other_agents_response_str
                        + other_cq_str.format(
                            id=cq.id, cq=cq.critical_question, reason=cq.reason
                        )
                    )
            instructions = instructions.format(
                input_arg=state["input_arg"],
                other_agents_response=other_agents_response_str,
            )
            response = validator_llm.with_structured_output(
                CriticalQuestionList
            ).invoke([HumanMessage(instructions)])

            answer: SocialAgentAnswer = SocialAgentAnswer(
                critical_question_list=response,
                question_type="validate",
                prompt={"human": instructions},
                trait=trait,
            )

            round_answer_dict["validator"] = [answer]

            return {"final_cq": response,
                    "round_answer_dict": round_answer_dict}
        workflow = StateGraph(SocialAgentState)

        for i, trait in enumerate(self.agent_trait_lst):
            workflow.add_node(
                f"llm_role_node{i}",
                lambda state, node_id=i, trait=trait: llm_role_node(
                    state, node_id, trait
                ),
            )

            workflow.add_node(
                f"question_node{i}",
                lambda state, node_id=i, trait=trait: question_node(
                    state, node_id, trait
                ),
            )

            if "debate" in self.collaborative_strategy:
                for r_ in [
                    j
                    for j, x in enumerate(self.collaborative_strategy)
                    if x == "debate"
                ]:
                    workflow.add_node(
                        f"r{r_}_debate_node{i}",
                        lambda state, node_id=i, trait=trait: debate_node(
                            state, node_id, trait
                        ),
                    )
            if "reflect" in self.collaborative_strategy:
                for r_ in [
                    j
                    for j, x in enumerate(self.collaborative_strategy)
                    if x == "reflect"
                ]:
                    workflow.add_node(
                        f"r{r_}_reflect_node{i}",
                        lambda state, node_id=i, trait=trait: reflect_node(
                            state, node_id, trait
                        ),
                    )

        workflow.add_node(validate_node)
        workflow.add_node("moderator_role", moderator_node)

        for i in range(self.llm_num):
            workflow.add_edge(START, f"llm_role_node{i}")
            workflow.add_edge("moderator_role", f"question_node{i}")

        workflow.add_edge(
            [f"llm_role_node{i}" for i in range(self.llm_num)], "moderator_role"
        )

        prev_strategy = "question"
        last_round = 0
        for round_, strategy in enumerate(self.collaborative_strategy):
            str_nodes = [
                f"r{round_}_{strategy}_node{i}" for i in range(self.llm_num)
            ]  # if i<(len(collaborative_strategy)-1) else ["validate_node"]

            curr_moderator = (
                "moderator_question" if i == 0 else f"moderator_round{round_}"
            )
            workflow.add_node(curr_moderator, moderator_node)
            pre_pend = "" if prev_strategy == "question" else f"r{round_-1}_"

            workflow.add_edge(
                [f"{pre_pend}{prev_strategy}_node{i}" for i in range(self.llm_num)],
                curr_moderator,
            )

            for e in str_nodes:
                workflow.add_edge(curr_moderator, e)

            prev_strategy = strategy
            last_round = round_

        curr_moderator = "moderator_final"
        workflow.add_node(curr_moderator, moderator_node)
        pre_pend = "" if prev_strategy == "question" else f"r{last_round}_"

        workflow.add_edge(
            [f"{pre_pend}{prev_strategy}_node{i}" for i in range(self.llm_num)],
            curr_moderator,
        )
        if len(self.collaborative_strategy) > 0:
            workflow.add_edge(curr_moderator, "validate_node")
            workflow.add_edge("validate_node", END)
        else:
            workflow.add_edge(curr_moderator, END)

        memory = MemorySaver()
        print("Building Completed!")
        return workflow.compile(checkpointer=memory)

    # Static methods
    @staticmethod
    def _get_strategy_permutation(
        n: Union[list[int], int], elements: Optional[list[str]] = ["debate", "reflect"]
    ):
        permutations = []

        def _get_permutation(r: int):
            for p in itertools.product(elements, repeat=r):
                permutations.append(p)

        if isinstance(n, int):
            _get_permutation(n)
        else:
            for r in n:
                _get_permutation(r)
        return permutations

    @staticmethod
    def _get_traits_combos(
        n: Union[list[int], int],
        elements: Optional[list[str]] = ["overconfident", "easy_going"],
    ):
        combos = []

        def _get_combinations(r: int):
            for p in itertools.combinations_with_replacement(elements, r):
                combos.append(p)

        if isinstance(n, int):
            _get_combinations(n)
        else:
            for r in n:
                _get_combinations(r)
        return combos

    @staticmethod
    def _generate_experiment_settings(
        rounds: int = 3,
        number_of_agents: int = 3,
        traits: list = ["overconfident", "easy_going"],
        strategies_: list = ["debate", "reflect"],
    ):
        all_expr = []
        all_traits_combos = SocialAgentBuilder._get_traits_combos(
            range(1, number_of_agents + 1), elements=traits
        )
        all_strategy_permutations = SocialAgentBuilder._get_strategy_permutation(
            range(1, rounds + 1), elements=strategies_
        )
        pairs = list(itertools.product(all_traits_combos, all_strategy_permutations))
        for pair in pairs:
            if len(pair[0]) == 1 and "debate" in pair[1]:
                continue

            trait_str = "".join(sorted([x[0] for x in pair[0]]))
            strategy_str = "".join([x[0] for x in pair[1]])
            all_expr.append(
                {
                    "traits": pair[0],
                    "strategies": pair[1],
                    "rounds": len(pair[1]),
                    "number_of_agents": len(pair[0]),
                    "has_debate": "debate" in pair[1],
                    "thread_id": uuid.uuid4(),
                    "experiment_name": "{llm_name}"
                    + f"social_n{len(traits)}_T{trait_str}_S{strategy_str}",
                }
            )
        for traits in all_traits_combos:
            trait_str = "".join(sorted([x[0] for x in traits]))
            all_expr.append(
                {
                    "traits": traits,
                    "strategies": tuple(),
                    "rounds": 0,
                    "number_of_agents": len(traits),
                    "has_debate": False,
                    "thread_id": uuid.uuid4(),
                    "experiment_name": "{llm_name}"
                    + f"social_n{len(traits)}_T{trait_str}_S",
                }
            )

        all_exp_settings_df = pd.DataFrame(all_expr)

        all_exp_settings_df.sort_values(
            by=["rounds", "number_of_agents"], ascending=True
        )
        return pd.DataFrame(all_expr)
