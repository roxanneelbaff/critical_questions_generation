## Validators and Aggregator
Validator and Aggregator class are located in the `agents` subdirectory.
# Libraries
`pip install json-repair`

#### Imports
```python
from AggregatorAgent import AggregatorAgent
from agents.ValidatorAgent import ValidatorAgent
```
#### Instantiating Three Validators
```python
DEFAULT_LLM = "llama3.2:3b-instruct-q8_0"

# Validator 1: Depth
subrole = """
Specifically assess whether the question delves into the argument's core, challenging the speaker's reasoning or assumptions.

Do this in two steps:
1. First, identify the speaker's corepoints, claims, arguments and assumptions.
2. Second, assess whether the question challenges any of those.
3. Finally, give a suggestion on the validity of the question. If it adresses any of the arguments core topics, it should be valid. If it adresses none of the core topics, it should be invalid.
"""
validator_1 = ValidatorAgent(subrole, DEFAULT_LLM)


# Validator 2: Relevance
subrole = """
Specifically assess whether the critical question does not contain any bad reasoning, namely, questions critical positions or claims **that the speaker does not hold**. If the question contains such bad reasoning, it would render the question invalid.

Do this in three steps:
1. First, identify the speakers overall position on the topic and the claims they make.
2. Second, determine the position and claims that the question suggests the speaker holds.
3. Finally, evaluate whether there is any significant mismatch between the speaker's actual position and claims and those implied by the question. If such a mismatch exists, assess whether its significance remains acceptable, considering that this is a critical question for that argument. If the mismatch is too inacceptable, the question is invalid. If the mismatch is not too impactfull, the question can remain valid. Give a final suggestion on the validity of the question.

Keep your answer concise overall. Don't be too strict in your assessment.
"""
validator_2 = ValidatorAgent(subrole, DEFAULT_LLM)


# Validator 3: Specificity
subrole  =  """
Specifically assess whether the critical question is specific to the argument and not generic. Give a suggestion on the validity of the question based on this. if the question is generic, then it should be invalid. If it is (broadly) adressing the topic of the argument, then it should be valid.

"""
validator_3 = ValidatorAgent(subrole, DEFAULT_LLM)


# Placing all validators into a list
validators  = [validator_1, validator_2, validator_3]
```

#### Instantiating One Aggregator
```python
aggregator = AggregatorAgent(DEFAULT_LLM)
```

#### Validating a Generated Question
The value of `is_valid` will hold the decision on whether or not the critical question is considered valid.
The value of `reasoning` can be used to feedback to the generating LLM, if desired.
```python
argument = "<some argument>"
critical_question = "<some generated question>"

# Step 1: Evaluate question using Validators
evaluations = []
for validator in validators:
    evaluation = validator.evaluate_critical_question(critical_question, argument)
    evaluations.append(evaluation)

# Step 2: Let Aggregator Agent make final decision on questions validity
reasoning:str, is_valid:bool = aggregator.aggregate_feedback(argument, critical_question, evaluations)
```
