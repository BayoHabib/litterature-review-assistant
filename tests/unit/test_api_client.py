# tests/unit/test_api_client.py

import unittest
from unittest.mock import patch, MagicMock
from src.llm_integration.api_client import OllamaClient
class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()

    @patch('requests.post')
    def test_generate(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': 'Generated text'}
        mock_post.return_value = mock_response

        response = self.client.generate("Test prompt")
        
        self.assertEqual(response, 'Generated text')
        mock_post.assert_called_once_with(
            'http://localhost:11434/api/generate',
            json={'model': 'llama2', 'prompt': 'Test prompt'}
        )

    @patch('requests.post')
    def test_generate_with_params(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': 'Generated text'}
        mock_post.return_value = mock_response

        response = self.client.generate("Test prompt", model="gpt4", params={"temperature": 0.7})
        
        self.assertEqual(response, 'Generated text')
        mock_post.assert_called_once_with(
            'http://localhost:11434/api/generate',
            json={'model': 'gpt4', 'prompt': 'Test prompt', 'temperature': 0.7}
        )

    @patch('requests.post')
    def test_generate_connection_error(self, mock_post):
        mock_post.side_effect = ConnectionError("Connection failed")

        with self.assertRaises(ConnectionError):
            self.client.generate("Test prompt")

    @patch('requests.get')
    def test_get_models(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {'models': [{'name': 'model1'}, {'name': 'model2'}]}
        mock_get.return_value = mock_response

        models = self.client.get_models()
        
        self.assertEqual(models, {'models': [{'name': 'model1'}, {'name': 'model2'}]})
        mock_get.assert_called_once_with('http://localhost:11434/api/tags')

    @patch('requests.get')
    def test_get_models_connection_error(self, mock_get):
        mock_get.side_effect = ConnectionError("Connection failed")

        with self.assertRaises(ConnectionError):
            self.client.get_models()

if __name__ == '__main__':
    unittest.main()