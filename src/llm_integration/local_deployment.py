# src/llm_integration/local_deployment.py

import subprocess
import time

class OllamaDeployment:
    @staticmethod
    def pull_model(model_name: str) -> None:
        """
        Pull a model from Ollama.

        :param model_name: The name of the model to pull
        """
        try:
            subprocess.run(["ollama", "pull", model_name], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to pull model {model_name}: {str(e)}")

    @staticmethod
    def list_models() -> list:
        """
        List all locally available models.

        :return: A list of model names
        """
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            # Parse the output to extract model names
            lines = result.stdout.strip().split('\n')[1:]  # Skip the header line
            return [line.split()[0] for line in lines]
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to list models: {str(e)}")

    @staticmethod
    def start_ollama(wait: int = 5) -> None:
        """
        Start the Ollama service.

        :param wait: Number of seconds to wait for the service to start
        """
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(wait)
        except Exception as e:
            raise RuntimeError(f"Failed to start Ollama service: {str(e)}")

    @staticmethod
    def stop_ollama() -> None:
        """
        Stop the Ollama service.
        """
        try:
            subprocess.run(["pkill", "ollama"], check=True)
        except subprocess.CalledProcessError:
            # If the process is not running, pkill will return a non-zero exit code
            # We're ignoring this as per the test specification
            pass