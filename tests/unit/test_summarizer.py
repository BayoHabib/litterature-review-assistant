# tests/unit/test_summarizer.py

import pytest
from unittest.mock import Mock, patch
from src.output_processing.summarizer import Summarizer

@pytest.fixture
def mock_ollama_client():
    return Mock()

@pytest.fixture
def mock_rag_manager():
    return Mock()

@pytest.fixture
def summarizer(mock_ollama_client, mock_rag_manager):
    with patch('src.output_processing.summarizer.OllamaClient', return_value=mock_ollama_client):
        with patch('src.output_processing.summarizer.RAGManager', return_value=mock_rag_manager):
            with patch('src.output_processing.summarizer.IndexManager'):
                return Summarizer()

def test_summarize_paper_with_rag(summarizer, mock_ollama_client, mock_rag_manager):
    paper = {
        'title': 'Test Paper',
        'authors': 'Test Author',
        'abstract': 'This is a test abstract'
    }
    mock_ollama_client.generate.return_value = "This is a summary."
    mock_rag_manager.enhance_prompt.return_value = "Enhanced prompt"

    result = summarizer.summarize_paper(paper)

    assert "This is a summary." in result
    mock_ollama_client.generate.assert_called_once()
    mock_rag_manager.enhance_prompt.assert_called_once()

def test_extract_key_points_with_rag(summarizer, mock_ollama_client, mock_rag_manager):
    paper = {
        'title': 'Test Paper',
        'abstract': 'This is a test abstract'
    }
    mock_ollama_client.generate.return_value = "Key point 1. Key point 2."
    mock_rag_manager.enhance_prompt.return_value = "Enhanced prompt"

    result = summarizer.extract_key_points(paper)

    assert "Key point" in result
    mock_ollama_client.generate.assert_called_once()
    mock_rag_manager.enhance_prompt.assert_called_once()

def test_compare_papers_with_rag(summarizer, mock_ollama_client, mock_rag_manager):
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
    mock_rag_manager.enhance_prompt.return_value = "Enhanced prompt"

    result = summarizer.compare_papers(paper1, paper2)

    assert "Comparison result." in result
    mock_ollama_client.generate.assert_called_once()
    mock_rag_manager.enhance_prompt.assert_called_once()

def test_summarize_paper_error_handling(summarizer, mock_ollama_client):
    paper = {
        'title': 'Test Paper',
        'authors': 'Test Author',
        'abstract': 'This is a test abstract'
    }
    mock_ollama_client.generate.side_effect = Exception("LLM Error")

    result = summarizer.summarize_paper(paper)

    assert "Error: Unable to summarize the paper." in result

def test_summarizer_with_no_rag_results(summarizer, mock_rag_manager, mock_ollama_client):
    mock_rag_manager.enhance_prompt.return_value = "Prompt without RAG context"
    paper = {
        'title': 'Test Paper',
        'authors': 'Test Author',
        'abstract': 'This is a test abstract'
    }
    mock_ollama_client.generate.return_value = "Summary without RAG"

    result = summarizer.summarize_paper(paper)

    assert result == "Summary without RAG"
    mock_ollama_client.generate.assert_called_once()
    mock_rag_manager.enhance_prompt.assert_called_once()