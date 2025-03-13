import random
import time
import json
from agents.GeneratorAgent import GeneratorAgent
from agents.ValidatorAgent import ValidatorAgent
from agents.AggregatorAgent import AggregatorAgent
from utils import print_text_bold, print_text_cyan, print_text_orange, print_text_pink,  format_duration, get_llm_calls, log_message

DEFAULT_LLM = "llama3.2:latest" # llama3.2:latest, llama3.2:3b-instruct-q5_K_M

# Logs
prefix = random.randint(10000, 99999)
log_file_path = f"/localdata1/opit_do/critical_question_generation/st_critical_questions/output/{prefix}_log.txt"
print(log_file_path)

# Create an empty log file
with open(log_file_path, "w"):
    log_message(log_file_path, f"Run Id: {prefix}")

### Validation Set ###
arguments_path = '/localdata1/opit_do/critical_question_generation/st_critical_questions/data_splits/validation.json'
print("Set: ", arguments_path)
with open(arguments_path, 'r') as file:
    arguments_data = json.load(file)
output_data = {}

### Generator Agent ###
model_name = DEFAULT_LLM  # llama3.2:1b, llama3:latest
generator = GeneratorAgent(model_name)
log_message(log_file_path, f"Generator Agent: {model_name}")

### Validator Agents ###
# Validator 1
model_name = DEFAULT_LLM  # llama3.2:1b, llama3:latest
subrole = """Specifically assess whether the critical question does not introduce any new concepts or topics, that were not covered in the argument. The introduction of an unmentioned topic or concept would render the question invalid.

Do this in three steps:
1. First, identify the (one or multiple) broad topic(s) of the argument.
2. Second, identify the (one or multiple) core topic(s) of the critical question.
3. Finally, assess whether or not the core topic of the critical question is part of the topics of the argument. If it is, the question can remain valid. If the core topic is different to the topics mentioned and implied in the argument, then the question should be invalid.

Keep your answer concise overall. Don't be too strict in your assessment.
"""
validator_1 = ValidatorAgent(subrole, model_name)
log_message(log_file_path, f"Validator Agent: {model_name}, Subtask: {subrole}")

# Validator 2
model_name = DEFAULT_LLM  # llama3.2:1b, llama3:latest
subrole = """Specifically assess whether the critical question does not contain any bad reasoning, namely, questions critical positions or claims **that the speaker does not hold**. If the question contains such bad reasoning, it would render the question invalid.

Do this in three steps:
1. First, identify the speakers overall position on the topic and the claims they make.
2. Second, determine the position and claims that the question suggests the speaker holds.
3. Finally, evaluate whether there is any significant mismatch between the speaker's actual position and claims and those implied by the question. If such a mismatch exists, assess whether its significance remains acceptable, considering that this is a critical question for that argument. If the mismatch is too inacceptable, the question is invalid. If the mismatch is not too impactfull, the question can remain valid. Give a final suggestion on the validity of the question.

Keep your answer concise overall. Don't be too strict in your assessment.
"""
validator_2 = ValidatorAgent(subrole, model_name)
log_message(log_file_path, f"Validator Agent: {model_name}, Subtask: {subrole}")

# Validator 3
model_name = DEFAULT_LLM  # llama3.2:1b, llama3:latest
subrole = """Specifically assess whether the critical question adresses the argument. Evaluate whether the question picks up on parts of the argument at all. If the question is specific to the argument, it can be considered as valid.

Do this in two steps:
1. First, identify the core topic(s) of the argument.
2. Second, assess whether the question adressess any of the core topic(s) of the argument.
3. Finally, give a suggestion on the validity of the question. If it adresses any of the arguments core topics, it should be valid. If it adresses none of the core topics, it should be invalid.

Keep your answer concise overall. Don't be too strict in your assessment.
"""
validator_3 = ValidatorAgent(subrole, model_name)
log_message(log_file_path, f"Validator Agent: {model_name}, Subtask: {subrole}")

validators = [validator_1, validator_2, validator_3]
log_message(log_file_path, f"This run uses (!) {len(validators)} (!) validator agents.")

### Aggregator Agent ###
model_name = DEFAULT_LLM  # llama3.2:1b, llama3:latest
aggregator = AggregatorAgent(model_name)


### Generation ###
number_of_arguments = 200
a_count = 0
start_time = time.time()
for key, value in arguments_data.items():
    # Extract argument and list of critical questions, which will be overwritten by the generated ones downstream
    intervention_id = value["intervention_id"]
    argument = value["intervention"]
    cqs = value["cqs"]
    log_message(log_file_path, "\n\n")
    log_message(log_file_path, f"{a_count}. Argument (id: {intervention_id})")

    # Log the argument to console
    print_text_bold(f"{a_count}. Argument:\n{argument}")

    # Start generation process
    generated_questions = []
    max_questions = 3
    count = 0
    feedback_summary = None
    while len(generated_questions) < max_questions:
        log_message(log_file_path, f" > Generating question no. {count}")
        q_start_time = time.time()
        question_valid = True
        feedback = []
        #if not feedback_summary:
        critical_question = generator.generate_critical_question(argument)
        #else:
        #    critical_question = generator.refine_critical_question(feedback_summary)
        log_message(log_file_path, f"   -> '{critical_question}'")
        print_text_cyan(f"Generator Agent:\n{critical_question}")

        # Evaluate question
        evaluations = []
        for validator in validators:
            evaluation = validator.evaluate_critical_question(critical_question, argument)
            evaluations.append(evaluation)
            print_text_orange(
                f"\n\n##########\nValidator Agent: Decision: {'valid' if evaluation['is_valid'] else 'invalid'}\n{evaluation['feedback']}\n"
            )

        # Ask Aggregator Agent to provide final feedback
        reasoning, is_valid = aggregator.aggregate_feedback(argument, critical_question, evaluations)
        q_end_time = time.time()
        d = q_end_time - q_start_time
        minutes = int((d % 3600) // 60)
        seconds = int(d % 60)
        print_text_pink(f"Aggregator Agent:\n\n{reasoning}\n\n")
        print_text_bold(f"Final Decision: Question is {'valid' if is_valid else 'invalid'} ({format_duration(q_start_time, q_end_time)})\n\n")
        
        if is_valid:
            suffix = random.randint(10000, 99999)
            generated_questions.append({"id": f"{intervention_id}_{suffix}", "cq": critical_question})
            feedback_summary = None
            print_text_bold(f">>> Number of questions for current argument: {len(generated_questions)}/{max_questions}\n\n\n\n")
            log_message(log_file_path, f"   -> Question valid")
        else:
            feedback_summary = reasoning
            log_message(log_file_path, f"   -> Question invalid")
        count += 1
    
    # Save to output
    v = value
    v["cqs"] = generated_questions
    output_data[key] = v

    # Save output to file
    filename = f"/localdata1/opit_do/critical_question_generation/st_critical_questions/output/{prefix}_output.json"
    with open(filename, 'w') as file:
        json.dump(output_data, file, indent=4)
    print(f"Saved batch of questions to {prefix}_output.json")
    log_message(log_file_path, f"Saved batch of questions to {prefix}_output.json")

    if a_count >= number_of_arguments:
        break
    a_count += 1


end_time = time.time()
print(f"Total duration: {format_duration(start_time, end_time)}")
print(f"Generated questions saved to {filename}")
print(f"Used {get_llm_calls()} llm calls in total. This is an average of {get_llm_calls()/(a_count*3)} calls per valid question.")

log_message(log_file_path, f"Total duration: {format_duration(start_time, end_time)}")
log_message(log_file_path, f"Generated questions saved to {filename}")
log_message(log_file_path, f"Used {get_llm_calls()} llm calls in total. This is an average of {get_llm_calls()/(a_count*3)} calls per valid question.")
