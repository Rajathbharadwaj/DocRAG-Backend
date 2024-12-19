from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict, cast
from model.types import ContentType
from rag import retrieve_documents
from config import settings, LLMProvider
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from indexer.web_indexer import WebIndexer
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import json

logger = logging.getLogger(__name__)


# Initialize FastAPI with lifespan
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add state management
class AppState:
    def __init__(self):
        self.web_indexer = None

app.state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    # Initialize app state
    app.state = AppState()
    
    try:
        # Initialize any resources
        yield
    finally:
        # Cleanup
        if app.state.web_indexer:
            await app.state.web_indexer.cleanup()

app = FastAPI(lifespan=lifespan)
class URLInput(BaseModel):
    url: HttpUrl
    content_type: Optional[ContentType] = None
    max_depth: int = Field(default=2, ge=0)  # Changed to ge=0
    max_links: Optional[int] = Field(default=None)
    backlink_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    doc_name: str

class RAGRequest(BaseModel):
    question: str
    index_name: str
    top_k: int = 4
    # thread_id: str = "default"
    # llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    # return_sources: bool = False
    # content_filter: Optional[List[ContentType]] = None

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
        # Initialize web indexer with descriptive doc_name
        app.state.web_indexer = WebIndexer(
            doc_name=url_input.doc_name,
            max_depth=url_input.max_depth,
            max_links=url_input.max_links,
            backlink_threshold=url_input.backlink_threshold
        )
        
        # Initialize crawler
        await app.state.web_indexer.initialize_crawler()
        
        # Convert string to ContentType enum
        content_type = ContentType(url_input.content_type) if url_input.content_type else ContentType.DOCUMENTATION
        
        # Set processing flag before adding to background tasks
        app.state.web_indexer.is_processing = True
        
        # Add both initial URL processing and backlink processing to background tasks
        background_tasks.add_task(
            app.state.web_indexer.process_initial_url,
            url=str(url_input.url),
            content_type=content_type
        )
        
        return {
            "status": "processing",
            "message": "URL indexing started. Subscribe to /index_status_stream/{doc_name} for real-time updates.",
            "doc_name": url_input.doc_name
        }
    except ValueError as e:
        logger.error(f"Invalid content type: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type. Valid types are: {[t.value for t in ContentType]}"
        )
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing URL: {str(e)}"
        )

@app.get("/index_status_stream/{doc_name}")
async def index_status_stream(doc_name: str):
    """Stream indexing status updates using SSE"""
    async def event_generator():
        retry_count = 0
        max_retries = 3  # Number of empty status retries before giving up
        
        while True:
            if not app.state.web_indexer:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "error": "No indexer initialized",
                        "doc_name": doc_name
                    })
                }
                break

            try:
                status = await app.state.web_indexer.get_indexing_status()
                
                # Reset retry counter on successful status
                retry_count = 0
                
                # Always send an update
                yield {
                    "event": "update",
                    "data": json.dumps(status)
                }
                
                # Only break if we're actually complete
                if status["is_complete"]:
                    yield {
                        "event": "complete",
                        "data": json.dumps(status)
                    }
                    break
                
                await asyncio.sleep(3)  # Update frequency
                
            except Exception as e:
                logger.error(f"Error getting status: {str(e)}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "error": f"Failed to get status after {max_retries} retries",
                            "doc_name": doc_name
                        })
                    }
                    break
                
                await asyncio.sleep(2)  # Wait longer between retries

    return EventSourceResponse(event_generator())

@app.post("/query")
async def query(request: RAGRequest):
    """Query endpoint with content type filtering"""
    try:
        response = await retrieve_documents(
            query=request.question,
            index_name=request.index_name,
            top_k=request.top_k
        )
        return response
            
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/indexing_status/{doc_name}")
async def get_indexing_status(doc_name: str):
    """Stream indexing status until complete"""
    async def status_generator():
        while True:
            if not app.state.web_indexer:
                yield f"event: error\ndata: {json.dumps({'error': 'No indexer initialized', 'doc_name': doc_name})}\n\n"
                break

            try:
                status = await app.state.web_indexer.get_indexing_status()
                
                if status["is_complete"]:
                    yield f"event: complete\ndata: {json.dumps(status)}\n\n"
                    break
                else:
                    yield f"event: update\ndata: {json.dumps(status)}\n\n"
                
                await asyncio.sleep(1)  # Update frequency
                
            except Exception as e:
                logger.error(f"Error getting status: {str(e)}")
                yield f"event: error\ndata: {json.dumps({'error': str(e), 'doc_name': doc_name})}\n\n"
                break

    return StreamingResponse(
        status_generator(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
