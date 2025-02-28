from typing import Optional
from agents.Agent import Agent
from utils import extract_json_from_string

SYSTEM_PROMPT = """
You are part of a larger system that generates critical questions to to interventions of political debates.
Other agents of this system have already created possible critical questions for the intervention.
Your task in this system is to assess the quality of the generated critical questions with respect to the original intervention. Avoid evaluating the intervention itself, this is not the task.

The user will provide you with both the original intervention and a corresponding critical question.
Assess the validity and usefulness of the question with regards to the intervention based on your role: {role}.
"""

EVALUATION_PROMPT = """
The intervention is this:

```
{intervention}
```

The corresponding critical question for you to evaluate is this:
```
{critical_question}
```

Your role is: {role}.
"""

ASSESSMENT_PROMPT = """
Based on your reasoning, do you assess the critical question to be useful?
Strictly return your answer in the following JSON format:  
```json
{{
  'is_useful': <True/False>,
}}
```
"""


class ValidatorAgent(Agent):
    def __init__(
        self,
        intervention: str,
        role: str,
        model_name: str,
        model_parameters: dict = {},
    ):
        super().__init__(model_name, model_parameters)

        # Agent specific parameters
        self.intervention = intervention
        self.role = role
        self.model_name = model_name

    def _initialize(self):
        # Initialize system prompt
        message = SYSTEM_PROMPT.format(role=self.role)
        self._add_to_chat(role="system", message=message)

    def evaluate_critical_question(self, critical_question: str) -> dict[str, str]:
        """Evaluates the provided critical question and assesses whether it is qualitative enough.

        Args:
            critical_question (str): The critical question to evaluate.

        Returns:
            dict[str, str]: The evaluation result. Dictionary that contains the key 'is_useful' and 'feedback'.
        """
        # Step 1: Evaluation: LLM evaluates quality of the question
        evaluation_prompt = EVALUATION_PROMPT.format(
            intervention=self.intervention, critical_question=critical_question, role=self.role
        )
        self._add_to_chat(role="user", message=evaluation_prompt)
        feedback = self.single_response(messages=self.chat)
        self._add_to_chat(role="assistant", message=feedback)

        # Step 2: Assessment: LLM assesses whether question is "good enough"
        assessment_prompt = ASSESSMENT_PROMPT
        self._add_to_chat(role="user", message=assessment_prompt)
        response = self.single_response(messages=self.chat)
        self._add_to_chat(role="assistant", message=response)

        # Process response
        response_dict: dict = extract_json_from_string(response)
        is_useful = response_dict.get("is_useful", None)
        if not is_useful:
            print("Error: LLM returned an invalid response. Expected JSON with key 'is_useful', but no JSON with that key was found.")
            # TODO Implement some retries here.

        evaluation = {"is_useful": is_useful, "feedback": feedback}

        # TODO: If not 'useful', then prompt again to summarize the feedback.

        return evaluation