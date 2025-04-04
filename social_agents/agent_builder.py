from abc import ABC
import dataclasses
import json
import os
import statistics
from typing import ClassVar, Optional
from langchain_openai import ChatOpenAI
import pandas as pd
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from pyparsing import abstractmethod
import tqdm
from langchain.chat_models import init_chat_model

from social_agents import helper

from .data_model import (
    Confirmation,
    CriteriaRank,
    CriteriaScore,
    CriticalQuestion,
    CriticalQuestionList,
    Validator4Agent,
)

from .state import (
    BasicState,
    RankerAgentState,
    SocialAgentAnswer,
    SocialAgentState,
    TwoStageValState,
    ValidatorAgentState,
)
from .utils import get_st_data, timer


# BUILDING GRAPH - 1 class per WORKFLOW #


@dataclasses.dataclass
class CQSTAbstractAgent(ABC):
    model_thread_id: str
    llm_name: str
    llm_num: int = 1
    experiment_name: Optional[str] = None
    temperature: Optional[float] = None

    serialize_func: callable = helper._basic_state_to_serializable

    # DO NOT SET
    out: json = None
    graph: StateGraph = None
    time_log: pd.DataFrame = None
    llm_lst: list = None

    ROOT_FOLDER: ClassVar[str] = "output/elbaff_experiment/"
    _SUB_FOLDER_: ClassVar[str] = "test_set/" # REMOVE THIS

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
        elif llm_name.startswith("gpt") or llm_name.startswith("openai"):
            return ChatOpenAI(model=llm_name, temperature=temperature)
        elif llm_name.lower().startswith(
            "meta-llama".lower()
        ) or llm_name.lower().startswith("mistralai".lower()):
            return init_chat_model(
                llm_name,
                model_provider="together",
                temperature=temperature,
            )
        elif llm_name.lower().startswith("mistral"):
            return init_chat_model(
                llm_name,
                model_provider="mistralai",
                temperature=temperature,
            )

    @abstractmethod
    def build_agent(self) -> StateGraph:
        pass

    def _invoke_graph(self, params, id_):
        print("invoking")
        questions = []
        config = {"configurable": {"thread_id": f"{self.model_thread_id}_{id_}"}}
        fname = f"{CQSTAbstractAgent.ROOT_FOLDER}final_states/{CQSTAbstractAgent._SUB_FOLDER_}{self.experiment_name}_arg{id_}.json"
        if os.path.exists(fname):
            print("arg already exist and cq generated, loading file")
            with open(fname, "r") as f:
                questions.append(
                    CriticalQuestionList.model_validate(json.load(f)["final_cq"])
                )
        else:
            for event in self.graph.stream(params, config, stream_mode="values"):
                # Review
                _labels = event.get("final_cq", "")
                if _labels:
                    questions.append(_labels)
                    # Save the final state

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
            with timer(f"Iteration {line['intervention_id']}",
                       time_in_seconds_arr):
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
            with open(
                f"{CQSTAbstractAgent.ROOT_FOLDER}output_{self.experiment_name}.json",
                "w",
            ) as o:
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

    _MAX_REPEAT_: ClassVar = 5

    def build_agent(self):
        # Define all the logic of the Graph here
        def generate_critical_questions(state: BasicState):
            structured_llm = self.llm_lst[0].with_structured_output(
                CriticalQuestionList
            )

            with open("prompts/question.txt", "r") as file:
                instructions = file.read()

            instructions = instructions.format(
                input_arg=state["input_arg"],
            )

            response = None

            exception_repeat = BasicCQModel._MAX_REPEAT_
            while exception_repeat > 0 and response is None:
                if exception_repeat < BasicCQModel._MAX_REPEAT_:
                    print(
                        f"Exception repeat {BasicCQModel._MAX_REPEAT_-exception_repeat}"
                    )
                response = structured_llm.invoke([HumanMessage(instructions)])
                exception_repeat = exception_repeat - 1
            if len(response.critical_questions) > 3:
                print("limiting it to 3")
                response.critical_questions = response.critical_questions[:3]
            return {"final_cq": response}

        builder = StateGraph(BasicState)
        builder.add_node("generate_critical_questions", generate_critical_questions)
        builder.add_edge(START, "generate_critical_questions")
        builder.add_edge("generate_critical_questions", END)

        graph = builder.compile()
        return graph


# @dataclasses.dataclass
class SocialAgentBuilder(CQSTAbstractAgent):

    collaborative_strategy: Optional[list[str]] = []
    agent_trait_lst: list[str] = ["easy_going"]
    validator_type: str = "DEFAULT"  # AGGREGATED_VALIDATOR EXTERNAL_SCORER

    _REPEAT_ON_FAIL_: ClassVar = 10

    def __init__(
        self,
        model_thread_id: str,
        llm_name: str,
        llm_num: int = 1,
        experiment_name: Optional[str] = None,
        temperature: Optional[float] = None,
        serialize_func: callable = helper._state_to_serializable,
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
                            "If you understand your role, please reply with OK only."
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
            exception_repeat = SocialAgentBuilder._REPEAT_ON_FAIL_
            response = None
            while exception_repeat > 0 and response is None:
                if exception_repeat < SocialAgentBuilder._REPEAT_ON_FAIL_:
                    print(
                        f"Q - Exception repeat {SocialAgentBuilder._REPEAT_ON_FAIL_-exception_repeat}"
                    )
                response = (
                    self.llm_lst[node_id]
                    .with_structured_output(CriticalQuestionList)
                    .invoke(
                        [SystemMessage(system_prompt)] + [HumanMessage(instructions)]
                    )
                )
                exception_repeat = exception_repeat - 1

            answer: SocialAgentAnswer = SocialAgentAnswer(
                critical_question_list=response,
                question_type="question",
                prompt={"system": system_prompt, "human": instructions},
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

            exception_repeat = SocialAgentBuilder._REPEAT_ON_FAIL_
            response = None
            while exception_repeat > 0 and response is None:
                if exception_repeat < SocialAgentBuilder._REPEAT_ON_FAIL_:
                    print(
                        f"Debate - Exception repeat {SocialAgentBuilder._REPEAT_ON_FAIL_-exception_repeat}"
                    )
                response = (
                    self.llm_lst[node_id]
                    .with_structured_output(CriticalQuestionList)
                    .invoke(
                        [SystemMessage(system_prompt)] + [HumanMessage(instructions)]
                    )
                )
                exception_repeat = exception_repeat - 1
            answer: SocialAgentAnswer = SocialAgentAnswer(
                critical_question_list=response,
                question_type="debate",
                prompt={"system": system_prompt, "human": instructions},
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
            exception_repeat = SocialAgentBuilder._REPEAT_ON_FAIL_
            response = None
            while exception_repeat > 0 and response is None:
                if exception_repeat < SocialAgentBuilder._REPEAT_ON_FAIL_:
                    print(
                        f"Reflect - Exception repeat {SocialAgentBuilder._REPEAT_ON_FAIL_-exception_repeat}"
                    )
                response = (
                    self.llm_lst[node_id]
                    .with_structured_output(CriticalQuestionList)
                    .invoke(
                        [SystemMessage(system_prompt)] + [HumanMessage(instructions)]
                    )
                )
                exception_repeat = exception_repeat - 1
            answer: SocialAgentAnswer = SocialAgentAnswer(
                critical_question_list=response,
                question_type="reflect",
                prompt={"system": system_prompt, "human": instructions},
                trait=trait,
            )

            round_answer_dict[f"agent{node_id}"] = [answer]
            return {"round_answer_dict": round_answer_dict}

        def validate_node(state: SocialAgentState):
            round_answer_dict = state["round_answer_dict"]

            instructions = ""
            if self.llm_num == 1:
                response = round_answer_dict["agent0"][-1].critical_question_list
            else:
                with open("prompts/validator.txt", "r") as file:
                    instructions = file.read()
                # Get others answer

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
                exception_repeat = SocialAgentBuilder._REPEAT_ON_FAIL_
                response = None
                while exception_repeat > 0 and response is None:
                    if exception_repeat < SocialAgentBuilder._REPEAT_ON_FAIL_:
                        print(
                            f"Validate - Exception repeat {SocialAgentBuilder._REPEAT_ON_FAIL_-exception_repeat}"
                        )
                    response = validator_llm.with_structured_output(
                        CriticalQuestionList
                    ).invoke([HumanMessage(instructions)])
                    exception_repeat = exception_repeat - 1

            if len(response.critical_questions) > 3:
                print("limiting it to 3")
                response.critical_questions = response.critical_questions[:3]

            answer: SocialAgentAnswer = SocialAgentAnswer(
                critical_question_list=response,
                question_type="validate",
                prompt={"human": instructions},
                trait=trait,
            )

            round_answer_dict["validator"] = [answer]

            return {"final_cq": response, "round_answer_dict": round_answer_dict}

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
                "moderator_question" if round_ == 0 else f"moderator_round{round_}"
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
        # if len(self.collaborative_strategy) > 0:
        workflow.add_node(validate_node)
        workflow.add_edge(curr_moderator, "validate_node")
        workflow.add_edge("validate_node", END)
        # else:
        # workflow.add_edge(curr_moderator, END)

        memory = MemorySaver()
        print("Building Completed!")
        return workflow.compile(checkpointer=memory)


class ValidatorAgentBuilder(CQSTAbstractAgent):
    cqs: list[str]
    weights = {"depth": 1, "relevance": 1, "specificity": 1, "reasoning": 1}

    def __init__(
        self,
        model_thread_id: str,
        llm_name: str,
        llm_num: int = 1,
        experiment_name: Optional[str] = None,
        temperature: Optional[float] = None,
        serialize_func: callable = helper._basic_state_to_serializable,
        cqs: list[str] = [],
        weights: Optional[list[str]] = {
            "depth": 1,
            "relevance": 1,
            "specificity": 1,
            "reasoning": 1,
        },
    ):
        self.cqs = cqs if cqs is not None else []
        self.weights = (
            weights
            if weights is not None
            else {"depth": 1, "relevance": 1, "specificity": 1, "reasoning": 1}
        )
        assert len(cqs) > 3
        super().__init__(
            model_thread_id=model_thread_id,
            llm_name=llm_name,
            llm_num=llm_num,
            experiment_name=experiment_name,
            temperature=temperature,
            serialize_func=serialize_func,
        )

    def calculate_score(self, criteria: Validator4Agent):
        score = (
            criteria.depth * self.weights["depth"]
            + criteria.reasoning * self.weights["reasoning"]
            + criteria.relevance * self.weights["relevance"]
            + criteria.specificity * self.weights["specificity"]
        ) / sum(self.weights.values())
        return score

    def build_agent(self) -> StateGraph:
        # llm_num = len(self.agent_trait_lst)
        print("building workflow")
        validator_llm = CQSTAbstractAgent._init_llm(self.llm_name, self.temperature)
        repeat = 1 if self.temperature == 0 else 3

        def _score_cq_node(state: ValidatorAgentState, cq: str):
            with open("prompts/validators/four_scores.txt", "r") as f:
                prompt = f.read()
            scores = []
            while len(scores) < repeat:
                response = validator_llm.with_structured_output(Validator4Agent).invoke(
                    [
                        SystemMessage(
                            "You are a critical thinker, you trust the power of sound reasoning."
                        ),
                        HumanMessage(
                            prompt.format(input_arg=state["input_arg"], cq=cq)
                        ),
                    ],
                )
                if response is not None:
                    scores.append(self.calculate_score(response))
            return {"cq_scores_dict": {cq: round(statistics.mean(scores), 2)}}

        def aggregator(state: ValidatorAgentState):
            print(state["cq_scores_dict"])
            print(
                type(
                    sorted(
                        state["cq_scores_dict"],
                        key=state["cq_scores_dict"].get,
                        reverse=True,
                    )[:3]
                )
            )
            cqs = sorted(
                state["cq_scores_dict"], key=state["cq_scores_dict"].get, reverse=True
            )[:3]
            return {
                "final_cq": CriticalQuestionList(
                    critical_questions=[
                        CriticalQuestion(id=i, critical_question=cq, reason="")
                        for i, cq in enumerate(cqs)
                    ]
                )
            }

        workflow = StateGraph(ValidatorAgentState)
        workflow.add_node("aggregator", aggregator)
        for i, cq in enumerate(self.cqs):
            workflow.add_node(
                f"{i}_score_cq_node", lambda state, cq=cq: _score_cq_node(state, cq)
            )
            workflow.add_edge(START, f"{i}_score_cq_node")

            workflow.add_edge(f"{i}_score_cq_node", "aggregator")
        workflow.add_edge("aggregator", END)
        memory = MemorySaver()
        print("Building Completed!")
        return workflow.compile(checkpointer=memory)


class RankerAgentBuilder(CQSTAbstractAgent):
    # cqs: list[str]
    weights = {"depth": 1, "relevance": 1, "specificity": 1, "reasoning": 1}

    # do not set
    criteria_desc: dict = None
    
    _CRITERIA_DESC_JSON_FILE_: ClassVar = "prompts/validators/criteria_ranker_desc.json"

    def __init__(
        self,
        model_thread_id: str,
        llm_name: str,
        llm_num: int = 1,
        experiment_name: Optional[str] = None,
        temperature: Optional[float] = None,
        serialize_func: callable = helper._state_to_serializable,
        weights: Optional[list[str]] = {
            "depth": 1,
            "relevance": 1,
            "specificity": 1,
            "reasoning": 1,
        },
    ):
        self.weights = (
            weights
            if weights is not None
            else {"depth": 1, "relevance": 1, "specificity": 1, "reasoning": 1}
        )
        with open("prompts/validators/criteria_ranker_desc.json", "r") as f:
            self.criteria_desc = json.load(f)

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
        validator_llm = CQSTAbstractAgent._init_llm(self.llm_name,
                                                    self.temperature)

        def rank_criteria_node(state: RankerAgentState, criterion: str):
            criterion_desc = self.criteria_desc[criterion]
            with open("prompts/validators/criteria_ranker.txt", "r") as f:
                prompt = f.read().format(
                    criteria_name=criterion_desc["name"],
                    criteria_adj=criterion_desc["adj"],
                    criteria_desc=criterion_desc["desc"],
                    input_arg=state["input_arg"],
                    cqs=state["cqs"]
                )
                print(prompt)
            response = None
            while response is None:
                response = validator_llm.with_structured_output(CriteriaRank).invoke(
                    [
                        SystemMessage("You are an assistant that evaluates critical questions based on a specific quality."),
                        HumanMessage(
                            prompt
                        ),
                    ],
                )
            assert response is not None

            #print("RESPONSE", response)
            return {"criteria_cqs_rank_dict": {criterion: response}}

        def aggregator(state: RankerAgentState):
            cqs_ranks = state["criteria_cqs_rank_dict"]
            #print(state["criteria_cqs_rank_dict"])
            scores = {}

            for _, items in cqs_ranks.items():
                for item in items.cq_ranking_lst:
                    if item.cq not in scores.keys(): scores[item.cq] = []
                    scores[item.cq].append(item.rank)

            # Compute mean rank for each question
            mean_scores = {cq: sum(ranks) / len(ranks) for cq, ranks in scores.items()}

            # Sort by mean score (lower is better)
            cqs_items = sorted(mean_scores.items(), key=lambda x: x[1])
            cqs = [x[0] for x in cqs_items][:3]

            return {
                "final_cq": CriticalQuestionList(
                    critical_questions=[
                        CriticalQuestion(id=i, critical_question=cq, reason="")
                        for i, cq in enumerate(cqs)
                    ]
                )
            }

        workflow = StateGraph(RankerAgentState)
        workflow.add_node("aggregator", aggregator)
        for criterion in self.criteria_desc.keys():
            workflow.add_node(
                f"{criterion}_validation_node",
                lambda state, criterion=criterion:
                rank_criteria_node(state, criterion),
            )
            workflow.add_edge(START, f"{criterion}_validation_node")

            workflow.add_edge(f"{criterion}_validation_node", "aggregator")
        workflow.add_edge("aggregator", END)
        memory = MemorySaver()
        print("Building Completed!")
        return workflow.compile(checkpointer=memory)


class TwoStepsCriteriaScorer(CQSTAbstractAgent):
    # cqs: list[str]
    weights = {"depth": 1, "relevance": 1, "specificity": 1, "reasoning": 1}    

    def __init__(
        self,
        model_thread_id: str,
        llm_name: str,
        llm_num: int = 1,
        experiment_name: Optional[str] = None,
        temperature: Optional[float] = None,
        serialize_func: callable = helper._state_to_serializable,
        weights: Optional[list[str]] = {
            "depth": 1,
            "relevance": 1,
            "specificity": 1,
            "reasoning": 1,
        },
    ):
        self.weights = (
            weights
            if weights is not None
            else {"depth": 1, "relevance": 1, "specificity": 1, "reasoning": 1}
        )
       
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
        validator_llm = CQSTAbstractAgent._init_llm(self.llm_name,
                                                    self.temperature)

        def score_criteria_node(state: TwoStageValState, criterion: str):
            with open(f"prompts/validators/system_2step_{criterion}.txt", "r") as f:
                system = f.read()
            with open("prompts/validators/step1.txt", "r") as f:
                prompt_1 = f.read()
            with open("prompts/validators/step2.txt", "r") as f:
                prompt_2 = f.read()
            response = None
            cqs = state["cqs"]
            results: dict[str:dict] = {i: {} for i in range(len(cqs))}
            for i, cq in enumerate(cqs):
                response = None
                while response is None:
                    try:
                        messages: list = [
                                SystemMessage(system),
                                HumanMessage(
                                    prompt_1
                                )
                            ]
                        response1 = validator_llm.invoke(
                            messages
                        )
                        #print(response1)
                        messages = messages + [{"role": "assistant",
                                                "content": response1.content},
                                               HumanMessage(prompt_2.format(
                                                   input_arg=state["input_arg"],
                                                   cq=cq))]
                        response = validator_llm.with_structured_output(CriteriaScore).invoke(
                            messages
                        )
                        results[i][criterion] = response.score

                    except Exception as e:
                        print(f"Error: {e}")
                        response = None
            return {"cq_scores_dict": results}

        def aggregator(state: TwoStageValState):
            cq_score_dict = state["cq_scores_dict"]

            # Compute mean rank for each question
            mean_scores = {cq: sum(scores.values()) / len(scores.values()) for cq, scores in cq_score_dict.items()}

            # Sort by mean score (lower is better)
            cqs_items = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
            print(cqs_items)
            cqs_idx = list([x[0] for x in cqs_items])
            print("indices", cqs_idx)
            cqs = [state["cqs"][i] for i in cqs_idx][:3]
            print(cqs)
            return {
                "final_cq": CriticalQuestionList(
                    critical_questions=[
                        CriticalQuestion(id=i, critical_question=cq, reason="")
                        for i, cq in enumerate(cqs)
                    ]
                )
            }

        workflow = StateGraph(TwoStageValState)
        workflow.add_node("aggregator", aggregator)
        for criterion in ["depth", "reasoning", "specificity"]:
            workflow.add_node(
                f"{criterion}_validation_node",
                lambda state, criterion=criterion:
                score_criteria_node(state, criterion),
            )
            workflow.add_edge(START, f"{criterion}_validation_node")

            workflow.add_edge(f"{criterion}_validation_node", "aggregator")
        workflow.add_edge("aggregator", END)
        memory = MemorySaver()
        print("Building Completed!")
        return workflow.compile(checkpointer=memory)
