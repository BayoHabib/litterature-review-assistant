# tests/unit/test_summarizer.py

import pytest
from unittest.mock import Mock, patch
from src.output_processing.summarizer import Summarizer

@pytest.fixture
def mock_ollama_client():
    with patch('src.llm_integration.api_client.OllamaClient') as mock:
        client = mock.return_value
        client.generate.return_value = "Mocked LLM response"
        yield client

@pytest.fixture
def mock_index_manager():
    with patch('src.vector_store.index_manager.IndexManager') as mock:
        manager = mock.return_value
        manager.search.return_value = [
            {'content': 'Relevant context 1'},
            {'content': 'Relevant context 2'}
        ]
        yield manager

@pytest.fixture
def summarizer(mock_ollama_client, mock_index_manager):
    with patch('src.rag.rag_manager.RAGManager') as mock_rag_manager:
        rag_manager = mock_rag_manager.return_value
        rag_manager.enhance_prompt.return_value = "Enhanced prompt"
        return Summarizer()

def test_summarize_paper_with_rag(summarizer, mock_ollama_client):
    paper = {
        'title': 'Test Paper',
        'authors': 'Test Author',
        'abstract': 'This is a test abstract'
    }
    mock_ollama_client.generate.return_value = "This is a summary."

    result = summarizer.summarize_paper(paper)

    assert "This is a summary." in result
    mock_ollama_client.generate.assert_called_once()

def test_extract_key_points_with_rag(summarizer, mock_ollama_client):
    paper = {
        'title': 'Test Paper',
        'abstract': 'This is a test abstract'
    }
    mock_ollama_client.generate.return_value = "Key point 1. Key point 2."

    result = summarizer.extract_key_points(paper)

    assert "Key point" in result
    mock_ollama_client.generate.assert_called_once()

def test_compare_papers_with_rag(summarizer, mock_ollama_client):
    paper1 = {
        'title': 'Test Paper 1',
        'authors': 'Author 1',
        'abstract': 'Abstract 1'
    }
    paper2 = {
        'title': 'Test Paper 2',
        'authors': 'Author 2',
        'abstract': 'Abstract 2'
    }
    mock_ollama_client.generate.return_value = "Comparison result."

    result = summarizer.compare_papers(paper1, paper2)

    assert "Comparison result." in result
    mock_ollama_client.generate.assert_called_once()

def test_summarize_paper_error_handling(summarizer, mock_ollama_client):
    paper = {
        'title': 'Test Paper',
        'authors': 'Test Author',
        'abstract': 'This is a test abstract'
    }
    mock_ollama_client.generate.side_effect = Exception("LLM Error")

    result = summarizer.summarize_paper(paper)

    assert "Error: Unable to summarize the paper." in result

def test_summarizer_with_no_rag_results(summarizer, mock_index_manager):
    mock_index_manager.search.return_value = []
    paper = {
        'title': 'Test Paper',
        'authors': 'Test Author',
        'abstract': 'This is a test abstract'
    }

    result = summarizer.summarize_paper(paper)

    assert result is not None  # The summarizer should still produce a result even without RAG context

# Add more tests as needed...