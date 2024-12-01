# DocRAG

A production-ready instant RAG system for documentation. Efficiently crawls, processes, and indexes documentation for LLM-optimized information retrieval in seconds.

## Installation

```bash
pip install docrag
```

## Quick Start

### Python Usage
```python
from docrag import DocumentIndexer
from docrag.config import IndexingConfig

# Initialize with custom configuration
config = IndexingConfig(
    max_depth=2,
    backlink_threshold=0.1,
    max_urls_per_depth=20,
    chunk_size=1000,
    openai_api_key="your-api-key" or anthropic_api_key="your-api-key"  # Or use environment variable: OPENAI_API_KEY or ANTHROPIC_API_KEY       
)

indexer = DocumentIndexer(config)

# Index documentation
await indexer.index(
    url="https://python.langchain.com/docs/get_started/introduction",
    doc_name="langchain_docs"
)

# Query the indexed content
response = await indexer.query(
    "What are the main components of LangChain?",
    doc_name="langchain_docs"
)
```

### API Usage

Start the server:
```bash
docrag start --host 0.0.0.0 --port 8000
```

Or with environment file:
```bash
docrag start --env-file .env
```

#### API Endpoints

Index documentation:
```bash
curl -X POST "http://localhost:8000/v1/index" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://python.langchain.com/docs/get_started/introduction",
    "doc_name": "langchain_docs",
    "config": {
      "max_depth": 2,
      "backlink_threshold": 0.1,
      "max_urls_per_depth": 20
    }
  }'
```

Check indexing status:
```bash
curl "http://localhost:8000/v1/status/langchain_docs" \
  -H "Authorization: Bearer ${API_KEY}"
```

Query indexed content:
```bash
curl -X POST "http://localhost:8000/v1/query" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main components of LangChain?",
    "doc_name": "langchain_docs",
    "options": {
      "temperature": 0.7,
      "max_tokens": 500
    }
  }'
```

## Features

- **Instant Processing**: Immediately indexes main URL content for quick RAG availability
- **Background Processing**: Asynchronously crawls and indexes backlinks
- **Comprehensive Content Extraction**: 
  - Text content
  - Code blocks
  - Headers and sections
  - Backlinks analysis
- **Production Ready**:
  - API key authentication
  - Rate limiting
  - Error handling
  - Logging
  - Health checks
  - Docker support

## Configuration

Environment variables:
```env
OPENAI_API_KEY=your-openai-api-key
DOCRAG_API_KEY=your-api-key
DOCRAG_MAX_REQUESTS=100
DOCRAG_RATE_LIMIT=60
```

## Docker

```bash
docker pull docrag/docrag
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-openai-api-key \
  -e DOCRAG_API_KEY=your-api-key \
  docrag/docrag
```

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest tests/
```

## License

MIT License

