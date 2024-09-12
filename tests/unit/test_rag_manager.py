# tests/unit/test_rag_manager.py

import pytest
from unittest.mock import patch, Mock
from src.rag.rag_manager import RAGManager

@pytest.fixture
def mock_index_manager():
    with patch('src.vector_store.index_manager.IndexManager') as mock:
        mock.return_value.search.return_value = [
            {'content': 'Relevant context 1'},
            {'content': 'Relevant context 2'}
        ]
        yield mock.return_value

def test_rag_manager_get_relevant_context(mock_index_manager):
    rag_manager = RAGManager(mock_index_manager)
    context = rag_manager.get_relevant_context("test query")
    
    assert "Relevant context 1" in context
    assert "Relevant context 2" in context
    mock_index_manager.search.assert_called_once_with("test query", 3)

def test_rag_manager_enhance_prompt(mock_index_manager):
    rag_manager = RAGManager(mock_index_manager)
    base_prompt = "Summarize this:"
    query = "Test query"

    enhanced_prompt = rag_manager.enhance_prompt(base_prompt, query)

    assert "Relevant context 1" in enhanced_prompt
    assert "Relevant context 2" in enhanced_prompt
    assert base_prompt in enhanced_prompt
    mock_index_manager.search.assert_called_once_with(query, 3)

def test_rag_manager_no_relevant_context(mock_index_manager):
    mock_index_manager.search.return_value = []
    rag_manager = RAGManager(mock_index_manager)
    
    context = rag_manager.get_relevant_context("test query")
    
    assert context == ""
    
    enhanced_prompt = rag_manager.enhance_prompt("Base prompt", "test query")
    assert "Base prompt" in enhanced_prompt
    assert "Context information" not in enhanced_prompt

def test_rag_manager_error_handling(mock_index_manager):
    mock_index_manager.search.side_effect = Exception("Search error")
    rag_manager = RAGManager(mock_index_manager)
    
    context = rag_manager.get_relevant_context("test query")
    assert context == ""
    
    enhanced_prompt = rag_manager.enhance_prompt("Base prompt", "test query")
    assert "Base prompt" in enhanced_prompt
    assert "Context information" not in enhanced_prompt