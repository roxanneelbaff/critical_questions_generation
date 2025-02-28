from abc import abstractmethod
from ollama import ChatResponse
from agents.LLM import LLM


class Agent(LLM):
    def __init__(
        self,
        model_name: str,
        model_parameters: dict = {},
    ):
        # Instantiate LLM for this Agent
        super().__init__(model_parameters)

        self.model_name = model_name
        self.chat: list[dict[str, str]] = []

    @abstractmethod
    def _initialize(self) -> None:
        """Initializes the chat history for the agent by adding the system prompt."""
        pass

    def _add_to_chat(self, role: str, message: str) -> None:
        """Appends a message to the chat history.

        Args:
            role (str): The role of the message being added. Must be either "system", "assistant" or "user".
            message (str): The message being added.
        """
        self.chat.append({"role": role, "content": message})

    def single_response(self, messages: list[dict[str, str]]) -> str:
        """Prompts the LLM with the given chat messages and returns the complete response as a string.

        Args:
            messages (list[dict[str,str]]): Chat messages.

        Returns:
            str: Complete response of the LLM.
        """
        # Prompt LLM
        response: ChatResponse = self.client.chat(
            model=self.model_name,
            messages=messages,
            stream=False,
            options=self.model_parameters,
        )
        response = response["message"]["content"]
        return response
