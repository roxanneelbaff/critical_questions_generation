from social_agents.agent_builder import SocialAgentBuilder
from social_agents.utils import eval_experiment
import pandas as pd
from glob import glob
import os


def evaluate_existing_exp():
    all_results_path = SocialAgentBuilder.ROOT_FOLDER + "experiment_results.csv"
    path_ = SocialAgentBuilder.ROOT_FOLDER + "output_*.json"
    time_log_path = SocialAgentBuilder.ROOT_FOLDER + "time_log.csv"

    out_files = [x for x in glob(path_) if x.find("_eval_") == -1]
    evaluated_files = [x for x in glob(path_) if x.find("_eval_") > -1]
    data_split = "validation"
    threshold = 0.6
    metric = "similarity"

    all_results = []
    for out_ in out_files:
        eval_name = out_.replace(
            "json", f"_eval_{metric}_{str(threshold).replace('.', '')}.json"
        )
        if eval_name in evaluated_files:
            print("already evaluated")
            continue

        exp_name = os.path.basename(out_).replace("output_", "").replace(".json", "")
        eval_dict = {"experiment_name": exp_name}
        eval_dict = eval_dict | eval_experiment(
            submission_path=out_, data_split=data_split, threshold=threshold
        )

        time_log_df = pd.read_csv(time_log_path)
        if exp_name not in time_log_df.columns:
            print(f"time not logged for {exp_name}")
        else:
            eval_dict["time_mean"] = time_log_df[exp_name].mean()
            eval_dict["time_std"] = time_log_df[exp_name].std()
        try:
            all_experiments_results_df = pd.read_csv(all_results_path)
            all_experiments_results_df = all_experiments_results_df.drop(
                columns=[
                    col
                    for col in all_experiments_results_df.columns
                    if col.startswith("Unnamed")
                ]
            )
        except FileNotFoundError:
            all_experiments_results_df = pd.DataFrame()

        all_experiments_results_df = pd.concat(
            [all_experiments_results_df, pd.DataFrame([eval_dict])], ignore_index=True
        )
        all_experiments_results_df.to_csv(all_results_path, index=False)
        all_results.append(eval_dict)

    new_results_df = pd.DataFrame(all_results)
    all_experiments_results_df = pd.read_csv(all_results_path)
    summary_df = all_experiments_results_df[
        [
            "experiment_name",
            "Useful_ratio",
            "3/3_ratio",
            "overall_punctuation",
            "time_mean",
            "time_std",
        ]
    ]
    summary_df.to_csv(all_results_path.replace(".csv", "_summary.csv"), index=False)

    summary_extended_df = all_experiments_results_df.apply(
        _apply_extend_experiment_attribute, axis=1
    )
    summary_extended_df["overall_punctuation"] = summary_extended_df[
        "overall_punctuation"
    ].round(2)

    summary_extended_df["+/-"] = (
        summary_extended_df["not_able_to_evaluate"] / (186 * 3)
    ).round(2)
    summary_extended_df["upper_bound"] = (
        (summary_extended_df["+/-"] / 2) + summary_extended_df["overall_punctuation"]
    ).round(2)
    summary_extended_df = summary_extended_df.sort_values(
        by=[
            "overall_punctuation",
            "3/3_ratio",
        ],
        ascending=False,
    )
    return new_results_df, all_experiments_results_df, summary_extended_df


def _apply_extend_experiment_attribute(row):
    settings = row["experiment_name"].split("_")
    if len(settings) < 5:
        return row
    offset = 2 if settings[0].startswith("4") or settings[0].startswith("2") else 0
    row["model_name"] = settings[
        offset
    ]  # if (settings[0].startswith("4") or settings[0].startswith("2"))
    row["number_of_agents"] = int(settings[offset + 2].replace("n", "").strip())
    row["traits"] = settings[offset + 3].replace("T", "").strip()
    row["strategies"] = settings[offset + 4].replace("S", "").strip()
    row["round"] = int(len(row["strategies"]))
    return row
