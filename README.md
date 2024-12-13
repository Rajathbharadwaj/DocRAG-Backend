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
    max_depth=2,
    max_links=5
)
```

## Configuration

### Processing Options
- `url`: Target URL to process
- `content_type`: Specific ContentType enum value
- `max_depth`: Maximum crawling depth (default: 2)
- `max_links`: Maximum number of links to process at each depth
- `backlink_threshold`: Minimum backlink ratio (default: 0.3)

### Depth Control Examples

#### max_depth=0 (Single Page)
```python
# Only processes the target URL, no link crawling
result = await indexer.process_initial_url(
    url="https://python.org/docs/tutorial",
    max_depth=0
)
```
Initial URL: python.org/docs
│
├── Process this page ✅
│
└── Direct links: ✅ Will be crawled
├── python.org/docs/tutorial
├── python.org/docs/library
└── python.org/docs/reference
`
Initial URL: python.org/docs (finds 20 links)
│
├── Only processes first 5 links:
│ ├── python.org/docs/tutorial ✅
│ ├── python.org/docs/library ✅
│ ├── python.org/docs/reference ✅
│ ├── python.org/docs/howto ✅
│ ├── python.org/docs/faq ✅
│ └── python.org/docs/glossary ❌ (dropped)
│
└── Each of those 5 pages also only processes their top 5 links
:
URL: python.org/docs/tutorial
Backlinks: 4 pages link to it
Ratio: 4/10 = 0.4 (40%) ✅ Process
URL: python.org/docs/obscure-page
Backlinks: 1 page links to it
Ratio: 1/10 = 0.1 (10%) ❌ Skip

#### max_depth=0 (Single Page)
```python
# Only processes the target URL, no link crawling
result = await indexer.process_initial_url(
    url="https://python.org/docs/tutorial",
    max_depth=0
)
```

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