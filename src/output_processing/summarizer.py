# src/output_processing/summarizer.py

from typing import Dict
from src.llm_integration.api_client import OllamaClient
from src.prompt_engineering.prompt_templates import PromptTemplates
from src.rag.rag_manager import RAGManager
from src.vector_store.index_manager import IndexManager

class Summarizer:
    def __init__(self, model: str = "phi3"):
        self.client = OllamaClient()
        self.model = model
        self.index_manager = IndexManager()  # Assume index is already created and loaded
        self.rag_manager = RAGManager(self.index_manager)

    def summarize_paper(self, paper: Dict[str, str]) -> str:
        """
        Summarize a research paper based on its sections using RAG-enhanced LLM.

        :param paper: A dictionary containing paper sections (title, authors, abstract)
        :return: A summarized version of the paper
        """
        base_prompt = PromptTemplates.format_prompt('SUMMARIZE_PAPER', **paper)
        
        # Use the paper's title and abstract as the query for relevant context
        query = f"{paper['title']} {paper['abstract']}"
        enhanced_prompt = self.rag_manager.enhance_prompt(base_prompt, query)
        
        try:
            summary = self.client.generate(enhanced_prompt, model=self.model)
            return summary
        except Exception as e:
            print(f"Error in paper summarization: {str(e)}")
            return f"Error: Unable to summarize the paper."

    def extract_key_points(self, paper: Dict[str, str]) -> str:
        """
        Extract key points from a paper's abstract using RAG-enhanced LLM.

        :param paper: A dictionary containing paper title and abstract
        :return: Extracted key points
        """
        base_prompt = PromptTemplates.format_prompt('EXTRACT_KEY_POINTS', **paper)
        
        query = f"{paper['title']} {paper['abstract']}"
        enhanced_prompt = self.rag_manager.enhance_prompt(base_prompt, query)
        
        try:
            key_points = self.client.generate(enhanced_prompt, model=self.model)
            return key_points
        except Exception as e:
            print(f"Error in extracting key points: {str(e)}")
            return f"Error: Unable to extract key points."

    def compare_papers(self, paper1: Dict[str, str], paper2: Dict[str, str]) -> str:
        """
        Compare two papers using RAG-enhanced LLM.

        :param paper1: A dictionary containing details of the first paper
        :param paper2: A dictionary containing details of the second paper
        :return: Comparison of the two papers
        """
        base_prompt = PromptTemplates.format_prompt('COMPARE_PAPERS', 
                                                    title1=paper1['title'], 
                                                    authors1=paper1['authors'], 
                                                    abstract1=paper1['abstract'], 
                                                    title2=paper2['title'], 
                                                    authors2=paper2['authors'], 
                                                    abstract2=paper2['abstract'])
        
        query = f"{paper1['title']} {paper2['title']} {paper1['abstract']} {paper2['abstract']}"
        enhanced_prompt = self.rag_manager.enhance_prompt(base_prompt, query)
        
        try:
            comparison = self.client.generate(enhanced_prompt, model=self.model)
            return comparison
        except Exception as e:
            print(f"Error in comparing papers: {str(e)}")
            return f"Error: Unable to compare the papers."