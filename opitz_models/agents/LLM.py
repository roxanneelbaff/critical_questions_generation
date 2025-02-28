import os
from dotenv import load_dotenv
from abc import ABC
from ollama import Client

load_dotenv()
MODEL_HOST = os.getenv("MODEL_HOST")
OLLAMA_API_TOKEN = os.getenv("OLLAMA_API_TOKEN")


class LLM(ABC):
    def __init__(self, model_parameters: dict):
        self.model_parameters = {"cache_prompt": False}
        self.model_parameters.update(model_parameters)
        self.client = Client(
            host=MODEL_HOST,
            headers={"Authorization": f"Bearer {OLLAMA_API_TOKEN}"},
        )
