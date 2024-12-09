from typing import Optional, List, Dict
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from ..indexer.content_processor import ContentProcessor  # Updated import
from ..model.types import ContentType
import logging

logger = logging.getLogger(__name__)

class RAGQueryEngine:
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        vector_db_path: str
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_db_path = vector_db_path
        self.content_processor = ContentProcessor()  # Use the consolidated processor
        
    async def add_content(self, url: str, content_type: Optional[ContentType] = None) -> List[Document]:
        """Process and add content to the vector store"""
        try:
            # Use consolidated content processor
            content = await self.content_processor.process_url(
                url=url,
                content_type=content_type
            )
            
            if content and content.documents:
                # Add to vector store
                await self.vector_store.add_documents(content.documents)
                return content.documents
                
            return []
            
        except Exception as e:
            logger.error(f"Error adding content from {url}: {str(e)}")
            return []