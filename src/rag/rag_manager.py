# src/rag/rag_manager.py

from src.vector_store.index_manager import IndexManager
from typing import List, Dict, Any

class RAGManager:
    def __init__(self, index_manager):
        self.index_manager = index_manager

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        try:
            results = self.index_manager.search(query, k)
            context = "\n".join([result.get('content', '') for result in results])
            return context
        except Exception as e:
            print(f"Error in retrieving context: {str(e)}")
            return ""

    def enhance_prompt(self, base_prompt: str, query: str) -> str:
        context = self.get_relevant_context(query)
        if context:
            enhanced_prompt = f"""
            Context information:
            {context}

            Based on the above context and your knowledge, {base_prompt}
            """
        else:
            enhanced_prompt = f"""
            {base_prompt}
            """
        return enhanced_prompt