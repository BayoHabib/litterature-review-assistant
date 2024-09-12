# tests/unit/test_local_deployment.py

import unittest
from unittest.mock import patch, MagicMock
import subprocess  # Add this import
from src.llm_integration.local_deployment import OllamaDeployment

class TestOllamaDeployment(unittest.TestCase):
    @patch('subprocess.run')
    def test_pull_model(self, mock_run):
        OllamaDeployment.pull_model("test_model")
        mock_run.assert_called_once_with(["ollama", "pull", "test_model"], check=True)

    @patch('subprocess.run')
    def test_pull_model_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ollama pull")
        with self.assertRaises(RuntimeError):
            OllamaDeployment.pull_model("test_model")

    @patch('subprocess.run')
    def test_list_models(self, mock_run):
        mock_result = MagicMock()
        mock_result.stdout = "NAME      ID    SIZE   MODIFIED\nmodel1    1     5GB    2023-01-01\nmodel2    2     3GB    2023-01-02\n"
        mock_run.return_value = mock_result

        models = OllamaDeployment.list_models()
        self.assertEqual(models, ["model1", "model2"])
        mock_run.assert_called_once_with(["ollama", "list"], capture_output=True, text=True, check=True)

    @patch('subprocess.run')
    def test_list_models_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ollama list")
        with self.assertRaises(RuntimeError):
            OllamaDeployment.list_models()

    @patch('subprocess.Popen')
    @patch('time.sleep')
    def test_start_ollama(self, mock_sleep, mock_popen):
        OllamaDeployment.start_ollama()
        mock_popen.assert_called_once_with(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mock_sleep.assert_called_once_with(5)

    @patch('subprocess.Popen')
    def test_start_ollama_failure(self, mock_popen):
        mock_popen.side_effect = Exception("Failed to start")
        with self.assertRaises(RuntimeError):
            OllamaDeployment.start_ollama()

    @patch('subprocess.run')
    def test_stop_ollama(self, mock_run):
        OllamaDeployment.stop_ollama()
        mock_run.assert_called_once_with(["pkill", "ollama"], check=True)

    @patch('subprocess.run')
    def test_stop_ollama_not_running(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "pkill ollama")
        OllamaDeployment.stop_ollama()  # This should not raise an exception

if __name__ == '__main__':
    unittest.main()