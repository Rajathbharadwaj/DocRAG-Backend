# DocRAG AI

A high-performance multi-document RAG system. Your one stop place for all your document understanding needs. Don't waste your time reading through documents. Let DocRAG do it for you while you focus on the important stuff.

## Features

### Content Type Support
- `ContentType.DOCUMENTATION`: Code docs, tutorials, and technical guides
- `ContentType.API`: API references, endpoints, and requests
- `ContentType.ACADEMIC`: Research papers and technical publications
- `ContentType.GITHUB`: READMEs, Issues, PRs, and Discussions
- `ContentType.STACKOVERFLOW`: Questions, answers, and technical discussions

### Performance Optimizations
- Parallel processing with batched operations
- Fast LLM model (gpt-4o-mini)
- Efficient content chunking
- Optimized backlink processing
- Concurrent URL crawling

## Installation

```bash
pip install crawl4ai
```

## Quick Start

```python
from indexer import WebIndexer, ContentType

# Initialize indexer
indexer = await WebIndexer().initialize_crawler()

# Process a URL
result = await indexer.process_initial_url(
    url="https://example.com/docs",
    content_type=ContentType.DOCUMENTATION,
    max_depth=2
)
```

## Configuration

### Processing Options
- `url`: Target URL to process
- `content_type`: Specific ContentType enum value
- `max_depth`: Maximum crawling depth (default: 2)
- `backlink_threshold`: Minimum backlink ratio (default: 0.3)

### Content Types
```python
from indexer import ContentType

# Available content types
ContentType.DOCUMENTATION  # Code docs and guides
ContentType.API           # API documentation
ContentType.ACADEMIC      # Research papers
ContentType.GITHUB        # GitHub content
ContentType.STACKOVERFLOW # StackOverflow content
```

## Processing Pipeline
1. URL submission with content type
2. Specialized content extraction
3. Batch processing of related URLs
4. Vector storage
5. Backlink management

## Performance

- Parallel URL processing
- Batched content extraction
- Optimized LLM calls
- Efficient chunking
- Concurrent operations

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.