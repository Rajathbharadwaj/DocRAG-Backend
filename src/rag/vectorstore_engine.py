from langchain.indexes import SQLRecordManager
from langchain_experimental.text_splitter import SemanticChunker
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ProcessPoolExecutor
from langchain_core.documents import Document
import pinecone
import asyncio
from typing import List
import logging
from dotenv import load_dotenv  
import os
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, PodSpec
import time

logger = logging.getLogger(__name__)

load_dotenv(override=True, dotenv_path='../../.env')

class VectorStoreEngine:
    def __init__(
        self,
        doc_name: str,
        use_serverless: bool = True,
        region: str = "us-east-1",
        breakpoint_type: str = "percentile",
        max_workers: int = 4,
        batch_size: int = 10,
        cleanup_mode: str = "incremental"
    ):
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Create index name from doc_name (sanitize the name)
        sanitized_name = doc_name.lower().replace("_", "-").replace(" ", "-")
        self.index_name = f"docrag-{sanitized_name}"
        
        # Initialize or get index
        self._initialize_index(use_serverless, region)
        
        # Initialize embeddings and other components
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            chunk_size=200
        )
        
        # Initialize record manager for deduplication
        self.record_manager = SQLRecordManager(
            f"pinecone/{doc_name}",
            db_url="sqlite:///record_manager_cache.sql"
        )
        self.record_manager.create_schema()
        
        # # Create or get index
        # if index_name not in self.pc.list_indexes():
        #     self.pc.create_index(
        #         name=index_name,
        #         dimension=1536,
        #         metric="cosine",
        #         spec=ServerlessSpec(
        #             cloud='aws',
        #             region=region
        #         )
        #     )
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            embedding=self.embeddings,
            index_name=self.index_name,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        self.cleanup_mode = cleanup_mode
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.text_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=breakpoint_type
        )

    def _initialize_index(self, use_serverless: bool, region: str):
        """Initialize or get Pinecone index"""
        try:
            # Check if index exists
            if not self.pc.has_index(self.index_name):
                logger.info(f"Creating new index: {self.index_name}")
                
                # Configure index spec
                if use_serverless:
                    spec = ServerlessSpec(
                        cloud='aws',
                        region=region
                    )
                else:
                    spec = PodSpec(
                        environment='gcp-starter'  # or your preferred pod type
                    )
                
                # Create index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric='cosine',
                    spec=spec
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                    
                logger.info(f"Index {self.index_name} created and ready")
            else:
                logger.info(f"Using existing index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error initializing index: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Don't delete the index, just cleanup client
            self.pc.deinit()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def _process_batch(self, batch: List[Document]) -> List[Document]:
        """Process a single batch of documents"""
        try:
            # Split documents using semantic chunker
            split_docs = self.text_splitter.create_documents(
                [doc.page_content for doc in batch]
            )
            
            # Create a mapping of original metadata
            metadata_map = {i: doc.metadata for i, doc in enumerate(batch)}
            
            # Preserve metadata for each chunk based on its origin
            for doc in split_docs:
                # Get the index from the chunk's metadata (added by semantic chunker)
                original_idx = doc.metadata.get('original_index', 0)
                # Update with original document's metadata
                if original_idx in metadata_map:
                    doc.metadata.update(metadata_map[original_idx])
            
            return split_docs
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise

    async def _process_batches_parallel(self, documents: List[Document]) -> List[Document]:
        """Process documents in parallel batches"""
        try:
            # Process in smaller sequential batches instead of full parallel
            batch_size = 5
            all_chunks = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                chunks = self._process_batch(batch)  # Note: Not async since _process_batch is sync
                all_chunks.extend(chunks)
                
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            raise


    async def add_documents(self, documents: List[Document]):
        """Add documents with deduplication and cleanup"""
        try:
            # Always process documents with semantic chunking
            chunked_docs = await self._process_batches_parallel(documents)
                
            # Use LangChain's indexing API for deduplication and cleanup
            from langchain.indexes import index
            result = index(
                chunked_docs,
                self.record_manager,
                self.vector_store,
                cleanup=self.cleanup_mode,
                source_id_key="source"
            )
            
            logger.info(
                f"Indexing complete: added={result['num_added']}, "
                f"updated={result['num_updated']}, "
                f"skipped={result['num_skipped']}, "
                f"deleted={result['num_deleted']}"
            )
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    async def add_documents_batch(self, documents: List[Document], batch_size: int = 100):
        """Add documents in batches to avoid memory issues"""
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                await self.add_documents(batch)
                logger.info(f"Processed batch {i//batch_size + 1} of {(len(documents) + batch_size - 1)//batch_size}")
                
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise