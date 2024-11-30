# DocRAG-Backend
# DocRAG-Backend

A dynamic web scraping and RAG (Retrieval-Augmented Generation) system that processes web content for LLM-optimized information retrieval.

## Features

- **Instant Processing**: Immediately indexes main URL content for quick RAG availability
- **Background Processing**: Asynchronously crawls and indexes backlinks
- **Comprehensive Content Extraction**: 
  - Text content
  - Images and media
  - Code blocks
  - Audio content
  - Backlinks analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/docrag-backend.git
```

2. Navigate to the project directory:
```bash
cd DocRAG-Backend


3. Install dependencies:
```bash
pip install -r requirements.txt
playwright install
```


## Usage

### Start the Server


```bash
python src/server.py
```


The server will start on `http://localhost:8000`

### API Endpoints

- **POST `/index_url`**: Start indexing a URL
  ```json
  {
    "url": "https://example.com"
  }
  ```

- **GET `/indexing_status`**: Check indexing progress

- **POST `/query`**: Query indexed content
  ```json
  {
    "question": "What is this website about?"
  }
  ```

## Project Structure
project_root/
├── requirements.txt
├── src/
│ ├── init.py
│ ├── main.py
│ ├── server.py
│ ├── indexer/
│ │ ├── init.py
│ │ ├── web_indexer.py
│ │ └── content_processor.py
│ ├── rag/
│ │ ├── init.py
│ │ └── query_engine.py
│ └── models/
│ ├── init.py
│ └── page_content.py