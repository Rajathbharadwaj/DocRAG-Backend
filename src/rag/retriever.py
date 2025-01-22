import os
from contextlib import contextmanager
from typing import Iterator, Optional, List
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC
import logging
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

# Add logging to check if .env is loaded
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get path to .env file
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), '.env')

logger.info(f"Loading .env from: {env_path}")
load_dotenv(dotenv_path=env_path, override=True)

# Add debug logging to verify API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if pinecone_api_key:
    logger.info("Pinecone API key loaded successfully")
    logger.info(f"API key starts with: {pinecone_api_key[:5]}... ends with: {pinecone_api_key[-5:]}")
else:
    logger.error("Failed to load Pinecone API key from environment variables")

def make_embeddings(model: str = "text-embedding-3-small") -> Embeddings:
    """Initialize embeddings model"""
    from langchain_openai import OpenAIEmbeddings
    
    return OpenAIEmbeddings(
        model=model,
        chunk_size=200
    )

@contextmanager
def make_pinecone_retriever(
    index_name: str,
    embedding_model: Optional[Embeddings] = None,
    top_k: int = 4,
    **kwargs
) -> Iterator[BaseRetriever]:
    """Create a Pinecone retriever with the given configuration"""
    try:
        # Initialize Pinecone client
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
            
        pc = PineconeGRPC(api_key=pinecone_api_key)
        
        # Get embeddings model if not provided
        if embedding_model is None:
            embedding_model = make_embeddings()
        
        # Initialize vector store
        vector_store = PineconeVectorStore(
            embedding=embedding_model,
            index_name=index_name,
            pinecone_api_key=pinecone_api_key
        )
        
        # Create retriever with search parameters
        search_kwargs = {
            "k": top_k,
            **kwargs
        }
        
        yield vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        
    except Exception as e:
        logger.error(f"Error creating Pinecone retriever: {str(e)}")
        raise

# Usage example
@contextmanager
def make_retriever(config: RunnableConfig) -> Iterator[BaseRetriever]:
    """Create a retriever based on the current configuration"""
    from rag.chat_agent.backend.retrieval_graph.configuration import AgentConfiguration
    
    configuration = AgentConfiguration.from_runnable_config(config)
    embedding_model = make_embeddings()
    
    logger.info(f"Creating retriever for doc: {configuration.index_name}")
    
    with make_pinecone_retriever(
        index_name=configuration.index_name,
        embedding_model=embedding_model
    ) as retriever:
        yield retriever