from typing import Optional
from agents.Agent import Agent
from utils import extract_json_from_string

SYSTEM_PROMPT = """
You are a professor for critical reasoning and logic.
Given a critical question along with a corresponding argument, your task is to judge the critical question for its validity. As part of a larger validator system, you have the following sub-role in this task: {subrole}
"""

EVALUATION_PROMPT = """
The argument is this:
```
{argument}
```

The corresponding critical question for you to evaluate is this:
```
{critical_question}
```

Your subrole for this task is: {subrole}.
End your response with a brief sentence about the questions validity.
"""

ASSESSMENT_PROMPT = """
Return your final answer as a JSON.
If you evaluated the questions validity to remain intact based on your evaluation, return True.
If you evaluated the questions validity to break based on your evaluation, return False.

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
        subrole: str,
        model_name: str,
        model_parameters: dict = {},
    ):
        super().__init__(model_name, model_parameters)

        # Agent specific parameters
        self.subrole = subrole
        self.model_name = model_name

        self._initialize()

    def _initialize(self):
        self.chat.clear()
        
        # Initialize system prompt
        message = SYSTEM_PROMPT.format(subrole=self.subrole)
        self._add_to_chat(role="system", message=message)

    def _clear_chat(self):
        self.chat = []

    def evaluate_critical_question(self, critical_question: str, argument: str) -> Optional[dict[str, str]]:
        """Evaluates the provided critical question and assesses whether it is qualitative enough.

        Args:
            critical_question (str): The critical question to evaluate.
            argument (str): The argument to which the question belongs.

        Returns:
            dict[str, str]: The evaluation result. Dictionary that contains the key 'is_useful' and 'feedback'.
        """
        self._initialize()
        
        is_response_valid = False
        retry = 0
        max_retries = 3
        while not is_response_valid:
            # Step 1: Evaluation: LLM evaluates quality of the question
            evaluation_prompt = EVALUATION_PROMPT.format(
                argument=argument, critical_question=critical_question, subrole=self.subrole
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
            is_valid = response_dict.get("is_valid", None)
            if is_valid == None:
                retry += 1
                if retry >= max_retries:
                    evaluation = None
                    print("LLM did not output a valid response. Discarding result.")
                    break
                self._clear_chat()
                self._initialize()
                continue
            is_response_valid = True

            evaluation = {"is_valid": is_valid, "feedback": feedback}

        return evaluation