from social_agents.agent_builder import BasicCQModel
from .utils import llm_dict


def run_baselines():
    """ Runs the baseline agent with all models """
    for llm_key, llm_full in llm_dict.items():
        print(f"Running {llm_key}")

        basic_agent = BasicCQModel(
            model_thread_id=f"{llm_key}_zero-shot_",
            llm_name=llm_full,
            llm_num=1,
            experiment_name=f"{llm_key}_zero-shot_",
            temperature=0.0,
        )
        basic_agent.run_experiment()
