from typing import List, Dict
import chromadb

class RAGQueryEngine:
    def __init__(self, vector_db_path: str = "vector_store"):
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.chroma_client.get_collection("web_content")

    async def query(self, question: str, k: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_texts=[question],
            n_results=k
        )
        
        return [{
            'content': doc,
            'metadata': meta
        } for doc, meta in zip(results['documents'][0], results['metadatas'][0])] 