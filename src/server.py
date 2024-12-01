from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field, confloat, AnyHttpUrl
from typing import Annotated, Optional  
from indexer.web_indexer import WebIndexer
from rag.query_engine import RAGQueryEngine
from config import settings, LLMProvider
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

# Global variables
indexer: Optional[WebIndexer] = None
rag_engine: Optional[RAGQueryEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global indexer, rag_engine
    
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
    
    # Initialize RAG engine
    rag_engine = RAGQueryEngine(
        llm=llm,
        embeddings=embeddings,
        vector_db_path=settings.vector_store_path
    )
    
    # Initialize indexer with RAG engine
    indexer = await WebIndexer(
        max_depth=settings.max_depth,
        backlink_threshold=settings.backlink_threshold,
        rag_engine=rag_engine
    ).initialize_crawler()
    
    yield
    
    # Shutdown
    if indexer:
        await indexer.cleanup()

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

class URLInput(BaseModel):
    url: AnyHttpUrl
    max_depth: int = Field(default=2, ge=1)
    backlink_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    doc_name: str

class QueryInput(BaseModel):
    question: str

class RAGRequest(BaseModel):
    question: str
    thread_id: str = "default"
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    return_sources: bool = False

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
async def index_url(url_input: URLInput):
    """Process URL with custom crawling parameters."""
    try:
        page_content = await indexer.process_initial_url(
            url=str(url_input.url),
            max_depth=url_input.max_depth,
            backlink_threshold=url_input.backlink_threshold,
            doc_name=url_input.doc_name
        )
        
        if not page_content:
            raise HTTPException(
                status_code=400,
                detail="Failed to process URL. Please check if the URL is accessible."
            )
        
        return {
            "status": "success",
            "message": "Initial page processed, background indexing started",
            "doc_name": url_input.doc_name,
            "crawl_config": {
                "max_depth": url_input.max_depth,
                "backlink_threshold": url_input.backlink_threshold
            },
            "page_content": {
                "title": page_content.metadata.get('title', 'No title'),
                "word_count": page_content.metadata.get('word_count', 0),
                "initial_links_found": len(page_content.links)
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing URL: {str(e)}"
        )

@app.get("/indexing_status")
async def get_status():
    """Get the current status of background indexing."""
    return indexer.get_indexing_status()

@app.post("/query")
async def query(request: RAGRequest):
    """Query endpoint that accepts LLM provider choice"""
    try:
        llm = get_llm(request.llm_provider)
        embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
        
        rag_engine = RAGQueryEngine(
            llm=llm,
            embeddings=embeddings,
            vector_db_path=settings.vector_store_path
        )
        
        response = await rag_engine.query(
            question=request.question,
            thread_id=request.thread_id,
            return_sources=request.return_sources
        )
        
        return {
            "provider": request.llm_provider,
            **response
        }
    except Exception as e:
        raise HTTPException(500, str(e))

class DocumentURL(BaseModel):
    url: HttpUrl
    doc_name: str

@app.post("/api/documents")
async def add_document(document: DocumentURL):
    try:
        # Here you would add logic to:
        # 1. Download the document from the URL
        # 2. Process it
        # 3. Store it with the given doc_name
        return {
            "status": "success",
            "message": f"Document '{document.doc_name}' successfully added",
            "url": str(document.url)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 