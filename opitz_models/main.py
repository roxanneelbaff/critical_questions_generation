from agents.GeneratorAgent import GeneratorAgent
from agents.ValidatorAgent import ValidatorAgent
from utils import print_text_bold, print_text_cyan, print_text_orange

argument = """
CLINTON: "The central question in this election is really what kind of country we want to be and what kind of future we 'll build together
Today is my granddaughter 's second birthday
I think about this a lot
we have to build an economy that works for everyone , not just those at the top
we need new jobs , good jobs , with rising incomes
I want us to invest in you
I want us to invest in your future
jobs in infrastructure , in advanced manufacturing , innovation and technology , clean , renewable energy , and small business
most of the new jobs will come from small business
We also have to make the economy fairer
That starts with raising the national minimum wage and also guarantee , finally , equal pay for women 's work
I also want to see more companies do profit-sharing"
"""

### Generator Agent ###
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
generator = GeneratorAgent(argument, model_name)

### Validator Agents ###
# Validator 1
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
subrole = "Specifically pay attention to whether the critical question introduces any new concepts or topics, that were not explicitly mentioned in the argument. An introduction of an unmentioned topic or concept would render the question invalid."
validator_1 = ValidatorAgent(argument, subrole, model_name)

# Validator 2
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
subrole = "Specifically assess whether any invalid relations between claims and the premises provided to support those claims are drawn. Specifically check whether the question uncovers any blind spots in argumentation. If invalid relations between claims and premises are drawn, this would render the question invalid."
validator_2 = ValidatorAgent(argument, subrole, model_name)

# Validator 3
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
subrole = "Specifically evaluate whether the critical question contains bad reasoning, namely, questions critical positions or claims that the speaker does not hold. If that is the case, it would render the question invalid."
validator_3 = ValidatorAgent(argument, subrole, model_name)

# Validator 4
model_name = "llama3.2:latest"  # llama3.2:1b, llama3:latest
subrole = "Specifically assess whether the critical question is non-specific, meaning that it could be asked on any argument and is not focused on the argument at hand. If the question is too unspecific, it would render it invalid."
validator_4 = ValidatorAgent(argument, subrole, model_name)

validators = [validator_1, validator_2, validator_3, validator_4]


### Generation ###
print_text_bold(argument)
question_valid = False
feedback = []

# Generate initial question
critical_question = generator.generate_critical_question()
print_text_cyan("Generator Agent:")
print_text_cyan(critical_question)

# Evaluate question
evaluations = []
for validator in validators:
    evaluation = validator.evaluate_critical_question(critical_question)
    if evaluation == None:
        continue
    else:
        evaluations.append(evaluation)
        print()
        print("################################################")
        print()
        print_text_orange(
            f"Validator Agent: Critical Question is useful: {evaluation['is_valid']}"
        )
        print_text_orange(evaluation["feedback"])

# Check if all evaluations are positive
if all(evaluation["is_valid"] for evaluation in evaluations):
    question_valid = True
else:
    # Gather and concatenate feedbacks where 'is_valid' is False
    feedbacks = [
        f"### Feedback:\n{evaluation['feedback']}\n\n\n"
        for evaluation in evaluations
        if not evaluation["is_valid"]
    ]

    # Concatenate the feedbacks into a single string
    feedback = "".join(feedbacks)

print()
print()
print("###### Final Feedback:")
print()
print(feedbacks)

print("\n\nDone.")
