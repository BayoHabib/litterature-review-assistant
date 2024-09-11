# src/llm_integration/api_client.py

import requests
from typing import Dict, Any, Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def generate(self, prompt: str, model: str = "llama2", params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response using the Ollama API.

        :param prompt: The input prompt for the model
        :param model: The name of the model to use (default: "llama2")
        :param params: Additional parameters for the API call
        :return: The generated response as a string
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
        }
        if params:
            payload.update(params)

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama API: {str(e)}")

    def get_models(self) -> Dict[str, Any]:
        """
        Get the list of available models from Ollama.

        :return: A dictionary containing information about available models
        """
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch models from Ollama API: {str(e)}")