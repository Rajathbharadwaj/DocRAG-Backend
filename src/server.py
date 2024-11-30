from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel
from indexer.web_indexer import WebIndexer
from rag.query_engine import RAGQueryEngine
from pydantic import BaseModel, HttpUrl


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global indexer, rag_engine
    indexer = await WebIndexer().initialize_crawler()
    rag_engine = RAGQueryEngine()
    yield
    # Shutdown
    if indexer:
        await indexer.cleanup()

app = FastAPI(lifespan=lifespan)
indexer: Optional[WebIndexer] = None
rag_engine: Optional[RAGQueryEngine] = None

class URLInput(BaseModel):
    url: str

class QueryInput(BaseModel):
    question: str

@app.post("/index_url")
async def index_url(url_input: URLInput):
    """Immediately process the main URL and start background indexing."""
    # Process main URL immediately
    page_content = await indexer.process_initial_url(url_input.url)
    
    if not page_content:
        return {"status": "error", "message": "Failed to process URL"}
    
    return {
        "status": "success",
        "message": "Initial page processed, background indexing started",
        "page_content": {
            "title": page_content.metadata['title'],
            "word_count": page_content.metadata['word_count'],
            "initial_links_found": len(page_content.links)
        }
    }

@app.get("/indexing_status")
async def get_status():
    """Get the current status of background indexing."""
    return indexer.get_indexing_status()

@app.post("/query")
async def query(query_input: QueryInput):
    """Query the indexed content."""
    results = await rag_engine.query(query_input.question)
    return {
        "results": results,
        "indexing_status": indexer.get_indexing_status()
    }

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
    uvicorn.run(app, host="0.0.0.0", port=8000) 