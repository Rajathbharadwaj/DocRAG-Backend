from typing import List, Dict, Optional
from langchain_core.documents import Document
from .web_indexer import WebIndexer
import logging

logger = logging.getLogger(__name__)

class DocumentStore:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.web_indexer = WebIndexer(
            max_depth=2,
            backlink_threshold=0.3,
            rag_engine=self  # WebIndexer will call our add_documents
        )
    
    async def initialize(self):
        """Initialize the web indexer"""
        await self.web_indexer.initialize_crawler()
        return self

    async def add_url(
        self, 
        url: str,
        max_depth: Optional[int] = None,
        backlink_threshold: Optional[float] = None,
        doc_name: Optional[str] = None
    ) -> List[Document]:
        """Process URL using WebIndexer"""
        try:
            page_content = await self.web_indexer.process_initial_url(
                url=url,
                max_depth=max_depth,
                backlink_threshold=backlink_threshold,
                doc_name=doc_name
            )
            
            if not page_content:
                raise ValueError(f"Failed to process URL: {url}")
            
            return [page_content.document]
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            raise

    async def add_documents(self, documents: List[Document]):
        """Interface for WebIndexer to add documents to vector store"""
        try:
            self.vector_store.add_documents(documents)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise 