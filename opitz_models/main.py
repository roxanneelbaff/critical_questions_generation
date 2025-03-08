import random
import json
from agents.GeneratorAgent import GeneratorAgent
from agents.ValidatorAgent import ValidatorAgent
from utils import print_text_bold, print_text_cyan, print_text_orange


### Validation Set ###
arguments_path = '/localdata1/opit_do/critical_question_generation/st_critical_questions/data_splits/sample.json'
with open(arguments_path, 'r') as file:
    arguments_data = json.load(file)
output_data = {}

### Generator Agent ###
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
generator = GeneratorAgent(model_name)

### Validator Agents ###
# Validator 1
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
subrole = "Specifically pay attention to whether the critical question introduces any new concepts or topics, that were not explicitly mentioned in the argument. An introduction of an unmentioned topic or concept would render the question invalid."
validator_1 = ValidatorAgent(subrole, model_name)

# Validator 2
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
subrole = "Specifically evaluate whether the critical question contains bad reasoning, namely, questions critical positions or claims that the speaker does not hold. If that is the case, it would render the question invalid."
validator_2 = ValidatorAgent(subrole, model_name)

# Validator 3
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
subrole = "Specifically assess whether the critical question is non-specific, meaning that it could be asked on any argument and is not focused on the argument at hand. If the question is too unspecific, it would render it invalid."
validator_3 = ValidatorAgent(subrole, model_name)

validators = [validator_3]


### Generation ###
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
            if evaluation:
                if not evaluation["is_valid"]:
                    question_valid = False
                evaluations.append(evaluation)
                print_text_orange(
                    f"\n\nValidator Agent: Critical Question is useful: {evaluation['is_valid']}\n{evaluation['feedback']}"
                )
        if question_valid:
            generated_questions.append({"id": f"{intervention_id}_{id}", "cq": critical_question})

        count += 1
    
    # Save to output
    v = value
    v["cqs"] = generated_questions
    output_data[key] = v
    break

# Save output to file
prefix = random.randint(10000, 99999)
filename = f"/localdata1/opit_do/critical_question_generation/st_critical_questions/output/{prefix}_output.json"
with open(filename, 'w') as file:
    json.dump(output_data, file, indent=4)

print(f"Generated questions saved to {output_data}")
