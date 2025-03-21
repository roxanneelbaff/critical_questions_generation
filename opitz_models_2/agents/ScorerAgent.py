from typing import Optional
from agents.Agent import Agent
from utils import extract_json_from_string

SYSTEM_PROMPT = """
You are an expert in critical reasoning and logic.
The user will provide you with two items:
A) An argument, which is taken from a larger piece of a debate,
B) A corresponding critical question. A critical question is an inquiry that should be asked in order to judge if an argument is acceptable or fallacious.

Your task will be assess the usefulness of the critical question solely based on the following subtask:
{subtask}
"""

PROMPT_STEP_1 = """
Argument:
```
{argument}
```

Corresponding critical question to evaluate:
```
{critical_question}
```
"""

PROMPT_STEP_2 = """
Now, based on your reasoning, rate the usefulness of the question on a scale from 0 to 10. Assign a low score between 0 and 3 if the question is not helpful at all. Assign a medium score between 4 and 6 if the question is somewhat helpful. Assign a high score between 8 and 10 if you find the question to be very helpfull.

Strictly return your score in a JSON in the following structure:
```json
{{
  'score': <score>
}}
```
"""


class ScorerAgent(Agent):
    def __init__(
        self,
        subtask: str,
        model_name: str,
        weight:float=1.0,
        model_parameters: dict = {},
    ):
        super().__init__(model_name, model_parameters)

        # Agent specific parameters
        self.subtask = subtask
        self.model_name = model_name
        self.weight = weight

        self._initialize_chat()

    def _initialize_chat(self):
        super()._initialize(SYSTEM_PROMPT.format(subtask=self.subtask))

    def score_question(self, argument: str, question: str, max_retries:int = 5) -> float:
        # Prepare: Erase chat history
        self._initialize_chat()

        # Step 1: Let LLM reason about the question
        prompt_1 = PROMPT_STEP_1.format(argument=argument, critical_question=question)
        self._add_to_chat(role="user", message=prompt_1)
        response = self.single_response(messages=self.chat)
        self._add_to_chat(role="assistant", message=response)

        # Step 2: Let LLM score question
        for _ in range(max_retries):
            prompt_2 = PROMPT_STEP_2
            self._add_to_chat(role="user", message=prompt_2)
            response = self.single_response(messages=self.chat)

            # Step 3: Extract the score from LLM's response
            response_dict: dict = extract_json_from_string(response)
            score = response_dict.get("score", None)
            if score and isinstance(score, int):
                score = score
                return score
            else:
                # Remove last prompt so that LLM can be re-prompted
                self.chat.pop()
            
        # If question could not be scores due to invalid LLM responses, return -1
        return -1