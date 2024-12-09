from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict
from indexer.content_processor import ContentProcessor
from model.types import ContentType
from model.config import ProcessingConfig
from rag.query_engine import RAGQueryEngine
from config import settings, LLMProvider
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
import logging

logger = logging.getLogger(__name__)

# Global variables
content_processor: Optional[ContentProcessor] = None
rag_engine: Optional[RAGQueryEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global content_processor, rag_engine
    
    # Initialize LLM based on default provider
    if settings.default_provider == LLMProvider.ANTHROPIC:
        llm = ChatAnthropic(
            model=settings.anthropic_model,
            temperature=settings.temperature,
            anthropic_api_key=settings.anthropic_api_key
        )
    else:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key
    )
    
    # Initialize content processor
    content_processor = ContentProcessor()
    
    # Initialize RAG engine
    rag_engine = RAGQueryEngine(
        llm=llm,
        embeddings=embeddings,
        vector_db_path=settings.vector_store_path
    ).initialize()
    
    yield
    
    # Cleanup
    if rag_engine:
        await rag_engine.cleanup()

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

class URLInput(BaseModel):
    url: HttpUrl
    content_type: Optional[ContentType] = None
    max_depth: int = Field(default=2, ge=1)
    backlink_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    doc_name: str

class RAGRequest(BaseModel):
    question: str
    thread_id: str = "default"
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    return_sources: bool = False
    content_filter: Optional[List[ContentType]] = None

def get_llm(provider: LLMProvider):
    """Get LLM based on provider choice"""
    if provider == LLMProvider.OPENAI:
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
    elif provider == LLMProvider.ANTHROPIC:
        if not settings.anthropic_api_key:
            raise HTTPException(400, "Anthropic API key not configured")
        return ChatAnthropic(
            model=settings.anthropic_model,
            temperature=settings.temperature,
            anthropic_api_key=settings.anthropic_api_key
        )
    else:
        raise HTTPException(400, f"Unsupported LLM provider: {provider}")

@app.post("/index_url")
async def index_url(url_input: URLInput, background_tasks: BackgroundTasks):
    """Process URL with content type detection and custom parameters"""
    try:
        # Process initial URL
        content = await content_processor.process_url(
            url=str(url_input.url),
            content_type=url_input.content_type
        )
        
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Failed to process URL. Please check if the URL is accessible."
            )
        
        # Add to RAG engine
        await rag_engine.add_content(
            content=content,
            doc_name=url_input.doc_name
        )
        
        # Schedule background crawling if depth > 1
        if url_input.max_depth > 1:
            background_tasks.add_task(
                rag_engine.crawl_links,
                start_url=str(url_input.url),
                max_depth=url_input.max_depth,
                backlink_threshold=url_input.backlink_threshold,
                doc_name=url_input.doc_name
            )
        
        return {
            "status": "success",
            "message": "Initial page processed, background indexing started",
            "content_type": content.content_type.value,
            "doc_name": url_input.doc_name,
            "crawl_config": {
                "max_depth": url_input.max_depth,
                "backlink_threshold": url_input.backlink_threshold
            }
        }
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing URL: {str(e)}"
        )

@app.get("/indexing_status/{doc_name}")
async def get_status(doc_name: str):
    """Get the current status of background indexing for a document"""
    return await rag_engine.get_indexing_status(doc_name)

@app.post("/query")
async def query(request: RAGRequest):
    """Query endpoint with content type filtering"""
    try:
        llm = get_llm(request.llm_provider)
        
        response = await rag_engine.query(
            question=request.question,
            thread_id=request.thread_id,
            return_sources=request.return_sources,
            content_filter=request.content_filter
        )
        
        if request.return_sources:
            return {
                "provider": request.llm_provider,
                **response
            }
        else:
            return {
                "provider": request.llm_provider,
                "answer": response["answer"]
            }
            
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)