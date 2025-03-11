import random
import json
from agents.GeneratorAgent import GeneratorAgent
from agents.ValidatorAgent import ValidatorAgent
from agents.AggregatorAgent import AggregatorAgent
from utils import print_text_bold, print_text_cyan, print_text_orange, print_text_pink


### Validation Set ###
arguments_path = '/localdata1/opit_do/critical_question_generation/st_critical_questions/data_splits/sample_copy.json'
with open(arguments_path, 'r') as file:
    arguments_data = json.load(file)
output_data = {}

### Generator Agent ###
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
generator = GeneratorAgent(model_name)

### Validator Agents ###
# Validator 1
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
subrole = "Specifically assess whether the critical question does not introduce any new concepts or topics, that were not covered in the argument. An introduction of an unmentioned topic or concept would render the question invalid."
validator_1 = ValidatorAgent(subrole, model_name)

# Validator 2
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
subrole = "Specifically assess whether the critical question does not contain any bad reasoning, namely, questions critical positions or claims **that the speaker does not hold**. If the question contains any bad reasoning, it would render the question invalid. Be cautious here, this can be tricky. Only evaluate the question as invalid if you are absolutely sure."
validator_2 = ValidatorAgent(subrole, model_name)

# Validator 3
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
subrole = "Specifically assess whether the critical question is focused to the argument, meaning that it could not just be asked on any argument. If the question is specific to the argument, it can be considered as valid."
validator_3 = ValidatorAgent(subrole, model_name)

validators = [validator_1, validator_2, validator_3]

### Aggregator Agent ###
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
aggregator = AggregatorAgent(model_name)


### Generation ###
number_of_arguments = 3
a_count = 0
for key, value in arguments_data.items():
    # Extract argument and list of critical questions, which will be overwritten by the generated ones downstream
    intervention_id = value["intervention_id"]
    argument = value["intervention"]
    cqs = value["cqs"]

    # Log the argument to console
    print_text_bold(argument)

    # Start generation process
    generated_questions = []
    count = 0
    while len(generated_questions) < 3:
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
                f"\n\n##########\nValidator Agent: Decision: {'valid' if evaluation['is_valid'] else 'invalid'}\n{evaluation['feedback']}"
            )

        # Ask Aggregator Agent to provide final feedback
        reasoning, is_valid = aggregator.aggregate_feedback(argument, critical_question, evaluations)
        print_text_pink(f"Aggregator Agent:\n\n{reasoning}\n\n\t>>> Final Answer: Question is {'valid' if is_valid else 'invalid'}.")
        if is_valid:
            suffix = random.randint(10000, 99999)
            generated_questions.append({"id": f"{intervention_id}_{suffix}", "cq": critical_question})

        count += 1
    
    # Save to output
    v = value
    v["cqs"] = generated_questions
    output_data[key] = v

    # Save output to file
    prefix = random.randint(10000, 99999)
    prefix = "67450"
    filename = f"/localdata1/opit_do/critical_question_generation/st_critical_questions/output/{prefix}_output.json"
    with open(filename, 'a') as file:
        json.dump(output_data, file, indent=4)

    if a_count >= number_of_arguments:
        break
    a_count += 1



print(f"Generated questions saved to {filename}")
