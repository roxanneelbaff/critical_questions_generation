from agents.GeneratorAgent import GeneratorAgent
from agents.ValidatorAgent import ValidatorAgent
from utils import print_text_bold, print_text_cyan, print_text_orange

intervention = """
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
model_name = "llama3.2:3b-instruct-q5_K_M"  # llama3.2:1b, llama3:latest
generator = GeneratorAgent(intervention, model_name)

### Validator Agents ###
model_name = "llama3.2:3b-instruct-q5_K_M"  # llama3.2:1b, llama3:latest
role = "Your role is to assess whether any invalid relations between claims and the premises provided to support those claims are drawn. Specifically check whether the question uncovers the blind spots in argumentation"
validator_1 = ValidatorAgent(intervention, role, model_name)
validators = [validator_1]

### Generation Loop ###
print_text_bold(intervention)

question_valid = False
feedback = []
while not question_valid:
    # Generate initial question
    critical_question = generator.generate_critical_question(feedback)
    print_text_cyan("Generator Agent:")
    print_text_cyan(critical_question)

    # Evaluate question
    evaluations = []
    for validator in validators:
        evaluation = validator.evaluate_critical_question(critical_question)
        evaluations.append(evaluation)
        print()
        print_text_orange(f"Validator Agent: Critical Question is useful: {evaluation['is_useful']}")
        print_text_orange(evaluation["feedback"])

    # Check if all evaluations are positive
    if all(evaluation["is_useful"] for evaluation in evaluations):
        question_valid = True
    else:
        # Gather and concatenate feedbacks where 'is_useful' is False
        feedbacks = [
            f"### Feedback:\n{evaluation['feedback']}\n\n\n"
            for evaluation in evaluations
            if not evaluation["is_useful"]
        ]

        # Concatenate the feedbacks into a single string
        feedback = "".join(feedbacks)

print("\n\nDone.")
