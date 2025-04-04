import ast
import pandas as pd
import os

import tqdm

from social_agents import helper
from social_agents.agent_builder import SocialAgentBuilder
from .utils import llm_dict

EXPERIMENT_SETTINGS_FILE_PATH = "output/elbaff_experiment/experiment_settings.csv"


def get_all_experiment_settings(root_path: str = "", override=False):

    if not os.path.exists(f"{root_path}_{EXPERIMENT_SETTINGS_FILE_PATH}") or override:
        print("generating exp settings file")
        exps_df = helper.generate_experiment_settings()
        exps_df.to_csv(EXPERIMENT_SETTINGS_FILE_PATH, index=False)

    experiment_settings = pd.read_csv(EXPERIMENT_SETTINGS_FILE_PATH)
    experiment_settings = experiment_settings.sort_values(
        by=["rounds", "number_of_agents"]
    )
    return experiment_settings


def init_and_run_experiment(
    llm_key,
    llm_name,
    experiment_name,
    thread_id,
    number_of_agents,
    strategies,
    traits,
    temperature: float = 0.7,
    data_type="validation",
):
    exp_name = experiment_name.format(llm_name=llm_key)
    if os.path.exists(f"{SocialAgentBuilder.ROOT_FOLDER}output_{exp_name}.json"):
        print("Experiment ", exp_name, "already done!")
        return True
    print("EXPERIMENT NAME: ", exp_name)
    social_agent = SocialAgentBuilder(
        model_thread_id=thread_id,
        llm_name=llm_name,
        llm_num=number_of_agents,
        experiment_name=exp_name,
        temperature=temperature,
        collaborative_strategy=list(ast.literal_eval(strategies)),
        agent_trait_lst=list(ast.literal_eval(traits)),
    )
    #display(Image(social_agent.graph.get_graph(xray=1).draw_mermaid_png()))
    print(social_agent.graph.get_graph().draw_ascii())
    
    social_agent.run_experiment(data_type=data_type, save=True)
    print(f"finished {exp_name}")
    return True


def run_all_exp_settings_per_model(llm_key, root_path: str = "", data_set="validation"):
    experiment_settings = get_all_experiment_settings(root_path=root_path)
    all_done = False
    if llm_key not in llm_dict.keys():
        print("LLM not found, choose one of these:", llm_dict.keys())

    for _, row in tqdm(experiment_settings.iterrows(), total=len(experiment_settings)):

        _ = init_and_run_experiment(
            llm_key,
            llm_dict[llm_key],
            row["experiment_name"],
            row["thread_id"],
            row["number_of_agents"],
            row["strategies"],
            row["traits"],
            data_type=data_set,
        )
    all_done = True
    return all_done
