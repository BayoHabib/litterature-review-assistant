import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
from src.vector_store.index_manager import IndexManager

class TestIndexManager(unittest.TestCase):
    def setUp(self):
        self.index_manager = IndexManager(index_path='test_index')

    @patch('sentence_transformers.SentenceTransformer')
    @patch('faiss.IndexFlatL2')
    def test_create_index(self, mock_faiss_index, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1, 2, 3], [4, 5, 6]])
        self.index_manager.model = mock_model  # Replace the model with our mock

        documents = ["doc1", "doc2"]
        metadata = [{"id": 1}, {"id": 2}]
        self.index_manager.create_index(documents, metadata)

        mock_model.encode.assert_called_once_with(documents)
        mock_faiss_index.assert_called_once_with(3)  # dimension of mock embeddings
        self.assertEqual(self.index_manager.metadata, metadata)

    @patch('faiss.write_index')
    @patch('pickle.dump')
    def test_save_index(self, mock_pickle_dump, mock_faiss_write):
        self.index_manager.index = MagicMock()
        self.index_manager.metadata = [{"id": 1}]
        self.index_manager.save_index()

        mock_faiss_write.assert_called_once()
        mock_pickle_dump.assert_called_once()

    @patch('os.path.exists')
    @patch('faiss.read_index')
    @patch('pickle.load')
    def test_load_index(self, mock_pickle_load, mock_faiss_read, mock_path_exists):
        mock_path_exists.return_value = True  # Simulate that the index file exists
        mock_faiss_read.return_value = MagicMock()
        mock_pickle_load.return_value = [{"id": 1}]
        self.index_manager.load_index()

        self.assertIsNotNone(self.index_manager.index)
        self.assertEqual(self.index_manager.metadata, [{"id": 1}])

    @patch('sentence_transformers.SentenceTransformer')
    def test_search(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1, 2, 3]])
        self.index_manager.model = mock_model  # Replace the model with our mock

        self.index_manager.index = MagicMock()
        self.index_manager.index.search.return_value = (np.array([[0.5]]), np.array([[0]]))
        self.index_manager.metadata = [{"id": 1, "title": "Test"}]

        results = self.index_manager.search("test query")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], 1)
        self.assertEqual(results[0]['title'], "Test")
        self.assertEqual(results[0]['distance'], 0.5)

if __name__ == '__main__':
    unittest.main()