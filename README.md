# DocRAG AI

A high-performance multi-document RAG system. Your one stop place for all your document understanding needs. Don't waste your time reading through documents. Let DocRAG do it for you while you focus on the important stuff.

## Features

### Specialized Content Extractors
- **Code Documentation**: Processes code docs, tutorials, and technical guides
- **API Documentation**: Handles API references, endpoints, and requests
- **Academic Papers**: Processes research papers and technical publications
- **GitHub Content**: Specialized for READMEs, Issues, PRs, and Discussions
- **StackOverflow**: Optimized for questions, answers, and technical discussions

### Performance Optimizations
- Parallel processing with batched operations
- Fast LLM model (gpt-4o-mini)
- Efficient content chunking
- Optimized backlink processing
- Concurrent URL crawling

### Key Components
- **WebIndexer**: Main indexing engine
- **ContentProcessor**: Smart content type detection and routing
- **Specialized Extractors**: Type-specific content processing
- **RAG Integration**: Vector database storage for processed content

## Installation

```bash
pip install crawl4ai
```

## Quick Start

```python
from indexer import WebIndexer

# Initialize indexer
indexer = await WebIndexer().initialize_crawler()

# Process a URL
result = await indexer.process_initial_url(
    "https://example.com/docs",
    max_depth=2
)
```

## Configuration

### Crawler Settings
```python
crawler_config = {
    'headless': True,
    'browser_type': 'chromium',
    'page_timeout': 60000,
    'word_count_threshold': 10,
    'remove_overlay_elements': True
}
```

### Processing Options
- `max_depth`: Maximum crawling depth (default: 2)
- `backlink_threshold`: Minimum backlink ratio (default: 0.3)
- `batch_size`: URLs processed in parallel (default: 10)

## Content Processing

### Supported Content Types
1. Code Documentation
   - Technical guides
   - Library documentation
   - Implementation examples

2. API Documentation
   - Endpoint descriptions
   - Request/response formats
   - Authentication details

3. Academic Content
   - Research papers
   - Technical publications
   - Mathematical content

4. GitHub Content
   - READMEs
   - Issues and PRs
   - Discussions
   - Repository metadata

5. StackOverflow Content
   - Questions and answers
   - Code examples
   - Technical discussions

### Processing Pipeline
1. Content type detection
2. Specialized extraction
3. Batch processing
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