import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pickle
import os

class IndexManager:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_path: str = 'faiss_index'):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.index = None
        self.metadata = []

    def create_index(self, documents: List[str], metadata: List[Dict[str, Any]]):
        embeddings = self.model.encode(documents)  # Ensure this line is present
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        self.metadata = metadata

    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, f"{self.index_path}.faiss")
            with open(f"{self.index_path}_metadata.pkl", 'wb') as f:
                pickle.dump(self.metadata, f)
        else:
            raise ValueError("Index has not been created yet.")

    def load_index(self):
        if os.path.exists(f"{self.index_path}.faiss"):
            self.index = faiss.read_index(f"{self.index_path}.faiss")
            with open(f"{self.index_path}_metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            raise FileNotFoundError(f"Index file not found: {self.index_path}.faiss")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 indicates no match found
                result = self.metadata[idx].copy()
                result['distance'] = distances[0][i]
                results.append(result)
        
        return results

    def add_to_index(self, documents: List[str], metadata: List[Dict[str, Any]]):
        if self.index is None:
            raise ValueError("Index has not been created yet.")
        
        new_embeddings = self.model.encode(documents)
        self.index.add(new_embeddings.astype('float32'))
        self.metadata.extend(metadata)

# Example usage
if __name__ == "__main__":
    index_manager = IndexManager()

    # Create and save index
    documents = [
        "This is a sample document about AI.",
        "Vector stores are useful for similarity search.",
        "FAISS is an efficient similarity search library."
    ]
    metadata = [
        {"id": 1, "title": "AI Document"},
        {"id": 2, "title": "Vector Stores"},
        {"id": 3, "title": "FAISS Library"}
    ]
    index_manager.create_index(documents, metadata)
    index_manager.save_index()

    # Load index and perform search
    index_manager.load_index()
    results = index_manager.search("similarity search", k=2)
    print(results)