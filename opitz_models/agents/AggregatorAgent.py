from typing import Tuple
from agents.Agent import Agent
from utils import extract_json_from_string

SYSTEM_PROMPT = """
You are an expert in critical reasoning and logic that excells at validating critical questions to an argument. For a given argument, multiple Validator Agents have already provided an assessment on the validity of the critical question. Your role will be to provide a final decision on the validity of the generated question. Definition: A critical question is an inquiry that should be asked in order to judge if an argument is acceptable or fallacious.

Original argument:
```
{argument}
```

Generated critical question: ```{critical_question}```

The user will guide you through this process step by step, so please follow their instructions precisely.
"""

PROMPT_STEP_1 = """
First, provide a concise summary the following {number_of_feedbacks} feedbacks given by the Validator Agents:

{concatenated_feedback}
"""

PROMPT_STEP_2 = """
Very good. Next, based on your summary, decide on the validity of the critical question.
Do this in two steps:
1. First, based on your summary, concisely reason about the feedback in your own words. Weigh each feedback according to their significance.
2. Second, make your decision about the validity of the critical question.
"""

PROMPT_STEP_3 = """
Finally, formulate your decision as s JSON structure like this:
```json
{{
  'is_valid': <True/False>,
}}
```
If you evaluated the questions to be valid, return True.
If you evaluated the questions to be invalid, return False.
"""


class AggregatorAgent(Agent):
    def __init__(self, model_name: str, model_parameters: dict = {}):
        super().__init__(model_name, model_parameters)

    def _concatenate_feedbacks(self, feedbacks: list[str]) -> str:
        """Helper Function to concatenate a list of feedbacks into a single string.

        Args:
            feedbacks (list[str]): List of feedbacks for the question by the LLM

        Returns:
            str: Concatenated Feedbacks
        """
        result = ""
        for i, feedback_text in enumerate(feedbacks, 1):
            result += f"Feedback {i}:\n{feedback_text}\n\n\n"
        return result

    def aggregate_feedback(
        self, argument: str, critical_question: str, feedbacks: list[str]
    ) -> Tuple[str, bool]:
        """Aggregates the provided list of feedbacks for the critical question.
        Summarizes the feedback in a single string and provides a single answer on the validity of the question.

        Args:
            argument (str): Original argument
            critical_question (str): Generated critical question
            feedbacks (list[str]): List of individual feedback from the Validator LLMs

        Returns:
            Tuple[str,bool]: Feedback and final validation
        """

        system_prompt = SYSTEM_PROMPT.format(
            argument=argument, critical_question=critical_question
        )
        self._initialize(system_prompt)

        ### Step 1: Let Agent summarize the feedback
        concatenated_feedback = self._concatenate_feedbacks(feedbacks)
        prompt_1 = PROMPT_STEP_1.format(
            number_of_feedbacks=len(feedbacks),
            concatenated_feedback=concatenated_feedback,
        )
        self._add_to_chat(role="user", message=prompt_1)
        response = self.single_response(messages=self.chat)
        self._add_to_chat(role="assistant", message=response)

        ### Step 2: Let Agent reason about the feedback
        prompt_2 = PROMPT_STEP_2
        self._add_to_chat(role="user", message=prompt_2)
        reasoning = self.single_response(messages=self.chat)
        self._add_to_chat(role="assistant", message=reasoning)

        ### Step 3: Provide final decision in structured JSON
        prompt_3 = PROMPT_STEP_3
        self._add_to_chat(role="user", message=prompt_3)
        decision = self.single_response(messages=self.chat)
        self._add_to_chat(role="assistant", message=decision)

        # Extract response from JSON
        response_dict: dict = extract_json_from_string(decision)
        is_valid = response_dict.get("is_valid", None)

        return (reasoning, is_valid)