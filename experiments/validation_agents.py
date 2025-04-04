
import json
from .utils import llm_dict
from social_agents.agent_builder import (
    RankerAgentBuilder,
    TwoStepsCriteriaScorer,
    ValidatorAgentBuilder,
)

# EVALUATION STRATEGIES
VALIDATION_4SCORES_VAL = "4SCORES_VAL"  ## Scores each CQ with 1 prompt and 5 criteria - Langgraph has len(cqs) nodes
VALIDATION_4RANKS_VAL = (
    "4RANKING_VAL"  ## Ranks all cqs per criteria - Langgraph has len(criteria) nodes
)
VALIDATION_2STAGE = "2STAGE_VAL"


def load_last_state(arg):
    """Reloads the last state in a Langgraph workflow (pre-evaluation) and returns the CQs"""
    agents = [x for x in arg["round_answer_dict"].keys() if x.startswith("agent")]
    arg_cqs = []
    for agent in agents:
        last_cq_list = arg["round_answer_dict"][agent][-1]["critical_question_list"][
            "critical_questions"
        ]
        agent_cq = list(set([cq["critical_question"] for cq in last_cq_list]))
        arg_cqs.extend(agent_cq)

    arg_cqs = list(set(arg_cqs))
    original_arg = arg["input_arg"]

    return original_arg, arg_cqs


def revalidate_cqs(val_type, original_arg, arg_cqs, arg_id_, llm_full, exp_name):
    """Revalidates the CQs with the given validation strategy and returns the results"""
    thread_id = f"{val_type}_{exp_name}"
    if len(arg_cqs) <= 3:
        new_cqs = [{"cq": x, "id": i} for i, x in enumerate(arg_cqs)]

    elif val_type == VALIDATION_4SCORES_VAL:
        # init workflow
        validator_workflow = ValidatorAgentBuilder(
            model_thread_id=thread_id,
            llm_name=llm_full,
            llm_num=1,
            experiment_name=thread_id,
            temperature=0.7,
            cqs=arg_cqs,
        )
        # display(Image(validator_workflow.graph.get_graph(xray=1).draw_mermaid_png()))
        print(validator_workflow.graph.get_graph().draw_ascii())
        sorted_questions = validator_workflow._invoke_graph(
            {"input_arg": original_arg}, arg_id_
        ).model_dump()["critical_questions"]
        new_cqs = [
            {"cq": x["critical_question"], "id": x["id"]} for x in sorted_questions
        ]
    elif val_type == VALIDATION_2STAGE:
        # init workflow
        twostage = TwoStepsCriteriaScorer(
            model_thread_id=thread_id,
            llm_name=llm_full,
            llm_num=1,
            experiment_name=thread_id,
            temperature=0.7,
        )
        print(twostage.graph.get_graph().draw_ascii())
        sorted_questions = twostage._invoke_graph(
            {"input_arg": original_arg, "cqs": arg_cqs}, arg_id_
        ).model_dump()["critical_questions"]
        new_cqs = [
            {"cq": x["critical_question"], "id": x["id"]} for x in sorted_questions
        ]
    elif val_type == VALIDATION_4RANKS_VAL:
        # init workflow
        
        ranker_flow = RankerAgentBuilder(
            model_thread_id=thread_id,
            llm_name=llm_full,
            llm_num=1,
            experiment_name=thread_id,
            temperature=0.7,
        )
        # try:
        print(ranker_flow.graph.get_graph().draw_ascii())
        arg_cqs_str = "-" + "\n-".join(arg_cqs)
        sorted_questions = ranker_flow._invoke_graph(
            {"input_arg": original_arg, "cqs": arg_cqs_str}, arg_id_
        ).model_dump()["critical_questions"]
        new_cqs = [
            {"cq": x["critical_question"], "id": x["id"]} for x in sorted_questions
        ]
    return new_cqs


def reevaluate_experiments(social_experiments_lst:list, val_type: str, OUTPUT_FILES = "output/elbaff_experiment", is_test=False):
    for exp in social_experiments_lst: # mistral24b_social_n3_Teeo_Sddr
        llm_key = exp.split("_")[0]
        llm_full = llm_dict[f"{llm_key}_"]
         
        test_name_str = "TEST_" if is_test else ""
        out_f = f"{OUTPUT_FILES}/output_{test_name_str}{exp}.json"

        new_out_f = f"{OUTPUT_FILES}/output_{val_type}_{test_name_str}{exp}.json"

        test_str = "test_set/TEST_" if is_test else ""
        arg_path = f"{OUTPUT_FILES}/final_states/{test_str}{exp}_arg{{}}.json" # REMOVE TESTSET

        with open(out_f, "r")  as file:
            new_out = {}
            json_ = json.load(file)
            print(json_.keys())
            for arg_id_ in json_.keys():
                new_out[arg_id_] = json_[arg_id_] 
                new_out[arg_id_]['cqs'] = []
                # each arg
                with open(arg_path.format(arg_id_), "r")  as file:
                    arg = json.load(file)
                    original_arg, cqs = load_last_state(arg)
                    not_done = True
                    retries = 0
                    while not_done and retries < 10:
                        try:
                            retries = retries + 1
                            new_out[arg_id_]['cqs'] = revalidate_cqs(val_type, original_arg, cqs, arg_id_, llm_full, exp)
                            not_done = False
                        except Exception as e:
                            print(e)
                            not_done = True
                        

            # SAVE NEW 
            print(f"saving to {new_out_f}")
            with open(new_out_f, "w") as nof:
                json.dump(new_out, nof, indent=4)  # pretty print the json
