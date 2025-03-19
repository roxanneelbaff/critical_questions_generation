from typing import Optional
from agents.Agent import Agent
from utils import extract_json_from_string

SYSTEM_PROMPT = """
You are an expert in critical reasoning and logic.
The user will provide you with two things:
A) An argument, which is taken from a larger piece of a debate
B) A corresponding critical question. A critical question is an inquiry that should be asked in order to judge if an argument is acceptable or fallacious.

Your task is to assess the usefulness of the critical question. As part of a larger validator system, you have the following sub-task:

{subtask}
"""

EVALUATION_PROMPT = """
Argument:
```
{argument}
```

Corresponding critical question to evaluate:
```
{critical_question}
```

Your specific subtask is: {subtask}.
"""

ASSESSMENT_PROMPT = """
Return your final answer as a JSON.
If you evaluated the questions to be valid, return True.
If you evaluated the questions to be invalid, return False.

Here is the JSON structure for you to return.

```json
{{
  'is_valid': <True/False>,
}}
```
"""


class ValidatorAgent(Agent):
    def __init__(
        self,
        subtask: str,
        model_name: str,
        model_parameters: dict = {},
    ):
        super().__init__(model_name, model_parameters)

        # Agent specific parameters
        self.subtask = subtask
        self.model_name = model_name

        super()._initialize(SYSTEM_PROMPT.format(subtask=self.subtask))

    def evaluate_critical_question(
        self, critical_question: str, argument: str
    ) -> Optional[dict[str, str]]:
        """Evaluates the provided critical question and assesses whether it is qualitative enough.

        Args:
            critical_question (str): The critical question to evaluate.
            argument (str): The argument to which the question belongs.

        Returns:
            dict[str, str]: The evaluation result. Dictionary that contains the key 'is_valid' and 'feedback'.
        """
        self._initialize(SYSTEM_PROMPT.format(subtask=self.subtask))

        is_response_valid = False
        retry = 0
        max_retries = 3
        while not is_response_valid:
            # Step 1: Evaluation: LLM evaluates quality of the question
            evaluation_prompt = EVALUATION_PROMPT.format(
                argument=argument,
                critical_question=critical_question,
                subtask=self.subtask,
            )
            self._add_to_chat(role="user", message=evaluation_prompt)
            feedback = self.single_response(messages=self.chat)
            self._add_to_chat(role="assistant", message=feedback)

            # Step 2: Assessment: LLM assesses whether question is "good enough"
            assessment_prompt = ASSESSMENT_PROMPT
            self._add_to_chat(role="user", message=assessment_prompt)
            response = self.single_response(messages=self.chat)
            self._add_to_chat(role="assistant", message=response)

            # Process response which is supposed to be a JSON with the key "is_valid*"
            response_dict: dict = extract_json_from_string(response)
            is_valid = response_dict.get("is_valid", None)

            # If no valid JSON was extracted, retry
            if is_valid == None:
                retry += 1
                if retry >= max_retries:
                    evaluation = None
                    print("LLM did not output a valid response. Discarding result.")
                    break
                self._initialize(SYSTEM_PROMPT.format(subtask=self.subtask))
                continue
            is_response_valid = True

            evaluation = {"is_valid": is_valid, "feedback": feedback}

        return evaluation
