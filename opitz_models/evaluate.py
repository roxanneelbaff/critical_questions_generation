import subprocess
import json
import ast
import re
from collections import Counter

### Loading data
# Validation Set & Generated Questions
evaluation_file_path = '/localdata1/opit_do/critical_question_generation/st_critical_questions/data_splits/validation.json'
generated_questions_path = '/localdata1/opit_do/critical_question_generation/st_critical_questions/output/93849_output.json'
#generated_questions_path = evaluation_file_path


# ### Run evaluation script
# # Define arguments to pass to the evaluate.py script
# args = [
#     'python3',
#     '/localdata1/opit_do/critical_question_generation/st_critical_questions/eval_scripts/evaluation.py',
#     '--metric', 'similarity',
#     '--input_path', evaluation_file_path, # Path of the test set.
#     '--submission_path', generated_questions_path, # Path where the generated questions have been saved.
#     '--threshold', '0.6'
# ]

# # Run the script
# subprocess.run(args)




def _extract_eval_from_str(stdout_text: str):
    """
    From o3-mini
    """
    # --- Extract the labels counter ---
    label_dstr = r"Distribution of the labels:\s*Counter\((\{.*?\})\)"
    punc_dstr = r"Distribution of the intervention punctuation:\s*Counter\((\{.*?\})\)"

    def _match_labels(pattern_labels):
        labels_counter = None
        match_labels = re.search(pattern_labels, stdout_text)
        if match_labels:
            # match_labels.group(1) will be the dictionary string
            labels_dict_str = match_labels.group(1)
            # Safely evaluate the string to a Python dict
            labels_dict = ast.literal_eval(labels_dict_str)
            # Optionally wrap it in a Counter (if you want Counter behavior)
            labels_counter = Counter(labels_dict)
        return labels_counter

    labels_counter = _match_labels(label_dstr)
    punc_counter = _match_labels(punc_dstr)
    # --- Extract the overall punctuation value ---
    # Look for a line starting with "Overall punctuation" followed by a number
    pattern_overall = r"Overall punctuation\s+([0-9\.eE+-]+)"
    match_overall = re.search(pattern_overall, stdout_text)
    if match_overall:
        overall_punctuation = float(match_overall.group(1))
    else:
        overall_punctuation = None

    # Now you have your variables:
    result = {
        "labels_counter": labels_counter,
        "punctuation_counter": punc_counter,
        "overall_punctuation": overall_punctuation,
    }
    return result

def eval_experiment(
    submission_path: str,
    data_split: str = "validation",
    threshold: float = 0.6,
    metric: str = "similarity",
):
    # Build the command as a list of strings
    command = [
        "python",
        "/localdata1/opit_do/critical_question_generation/st_critical_questions/eval_scripts/evaluation.py",
        "--metric",
        metric,
        "--input_path",
        evaluation_file_path,
        "--submission_path",
        submission_path,
        "--threshold",
        str(threshold),
    ]
    print("Running command:", " ".join(command))
    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    # Print the standard output and error (if any)
    eval_res = _extract_eval_from_str(result.stdout)

    # generate result
    labels_dstr = dict(eval_res["labels_counter"])
    sum_ = sum(labels_dstr.values())
    labels_dstr_ratio = {
        f"{k}_ratio": round(v / sum_, 2) for k, v in labels_dstr.items()
    }

    punctuation_counter = dict(eval_res["punctuation_counter"])

    rename_mapping = {"1.0": "3/3", "0.6": "2/3", "0.3": "1/3", "0": "0/3"}

    punctuation_counter_proc = {
        rename_mapping[str(k)[: min(3, len(str(k)))]]: v
        for k, v in punctuation_counter.items()
    }
    punctuation_counter_ratio = {
        f"{k}_ratio": round(v / sum(punctuation_counter.values()), 2)
        for k, v in punctuation_counter_proc.items()
    }
    merged = (
        labels_dstr
        | labels_dstr_ratio
        | punctuation_counter_proc
        | punctuation_counter_ratio
    )
    merged["overall_punctuation"] = eval_res["overall_punctuation"]
    return merged


merged = eval_experiment(generated_questions_path)
print(merged)