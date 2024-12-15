from typing import Optional, List, Dict
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.indexes import SQLRecordManager, index
from langchain_community.vectorstores import Chroma
import logging

logger = logging.getLogger(__name__)

class VectorStoreEngine:
    def __init__(
        self,
        vector_db_path: str
    ):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)
        self.vector_store = Chroma(
            persist_directory=vector_db_path,
        )
        
    async def add_documents(self, documents: List[Document]):
        """Add documents to vector store"""
        try:
            self.vector_store.add_documents(documents)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise