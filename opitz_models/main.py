import random
import time
import json
from agents.GeneratorAgent import GeneratorAgent
from agents.ValidatorAgent import ValidatorAgent
from agents.AggregatorAgent import AggregatorAgent
from utils import print_text_bold, print_text_cyan, print_text_orange, print_text_pink,  format_duration, get_llm_calls

DEFAULT_LLM = "llama3.2:latest" # llama3.2:latest, llama3.2:3b-instruct-q5_K_M

### Validation Set ###
arguments_path = '/localdata1/opit_do/critical_question_generation/st_critical_questions/data_splits/validation.json'
with open(arguments_path, 'r') as file:
    arguments_data = json.load(file)
output_data = {}

### Generator Agent ###
model_name = DEFAULT_LLM  # llama3.2:1b, llama3:latest
generator = GeneratorAgent(model_name)

### Validator Agents ###
# Validator 1
model_name = DEFAULT_LLM  # llama3.2:1b, llama3:latest
subrole = "Specifically assess whether the critical question does not introduce any new concepts or topics, that were not covered in the argument. The introduction of an unmentioned topic or concept would render the question invalid."
validator_1 = ValidatorAgent(subrole, model_name)

# Validator 2
model_name = DEFAULT_LLM  # llama3.2:1b, llama3:latest
subrole = "Specifically assess whether the critical question does not contain any bad reasoning, namely, questions critical positions or claims **that the speaker does not hold**. If the question contains any bad reasoning, it would render the question invalid. Be cautious here, this can be tricky. Only evaluate the question as invalid if you are absolutely sure."
validator_2 = ValidatorAgent(subrole, model_name)

# Validator 3
model_name = DEFAULT_LLM  # llama3.2:1b, llama3:latest
subrole = "Specifically assess whether the critical question is focused to the argument, meaning that it could not just be asked on any argument. If the question is specific to the argument, it can be considered as valid."
validator_3 = ValidatorAgent(subrole, model_name)

validators = [validator_1, validator_2]

### Aggregator Agent ###
model_name = DEFAULT_LLM  # llama3.2:1b, llama3:latest
aggregator = AggregatorAgent(model_name)


### Generation ###
number_of_arguments = 6
a_count = 0
start_time = time.time()
prefix = random.randint(10000, 99999)
for key, value in arguments_data.items():
    # Extract argument and list of critical questions, which will be overwritten by the generated ones downstream
    intervention_id = value["intervention_id"]
    argument = value["intervention"]
    cqs = value["cqs"]

    # Log the argument to console
    print_text_bold(f"{a_count}. Argument:\n{argument}")

    # Start generation process
    generated_questions = []
    max_questions = 3
    count = 0
    while len(generated_questions) < max_questions:
        q_start_time = time.time()
        question_valid = True
        feedback = []
        critical_question = generator.generate_critical_question(argument)
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
            print_text_bold(f">>> Number of questions for current argument: {len(generated_questions)}/{max_questions}\n\n\n\n")
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

    if a_count >= number_of_arguments:
        break
    a_count += 1


end_time = time.time()
print(f"Total duration: {format_duration(start_time, end_time)}")

print(f"Generated questions saved to {filename}")
print(f"Used {get_llm_calls()} llm calls in total. This is an average of {get_llm_calls()/(a_count*3)} calls per valid question.")
