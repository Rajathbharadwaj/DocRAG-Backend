import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crawl4ai import CrawlResult
from .base import BaseExtractor
import logging

logger = logging.getLogger(__name__)

class APIDocumentationExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        # Use faster model with optimized settings
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_retries=1,
        )
        
        # Optimize chunk settings
        self.chunk_size = 16000
        self.chunk_overlap = 200
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n# ", "\n", " ", ""],
            keep_separator=True
        )
        
        self.preprocess_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an API documentation preprocessing expert. Your task is to parse and transform raw markdown documentation into LLM-friendly format.

PREPROCESSING REQUIREMENTS:

1. ENDPOINT DOCUMENTATION
   - Format HTTP methods prominently (GET, POST, PUT, DELETE)
   - Structure URL paths consistently
   - Highlight path parameters
   - Format query parameters in tables

2. REQUEST/RESPONSE FORMATTING
   - Format JSON examples with proper syntax highlighting
   ```json
   {
     "key": "value"
   }
   ```
   - Include sample requests with curl commands
   - Show response status codes and meanings
   - Format headers consistently

3. PARAMETER DOCUMENTATION
   - Create consistent parameter tables:
   | Parameter | Type | Required | Description |
   |-----------|------|----------|-------------|
   | param1    | string| Yes     | Description |

4. AUTHENTICATION
   - Highlight authentication methods
   - Show token/key placement
   - Include security warnings
   - Format auth headers

5. ERROR HANDLING
   - List possible error codes
   - Show error response formats
   - Include troubleshooting tips
   - Format error examples

6. STANDARDIZATION
   - Convert all headers to markdown syntax
   - Add proper code block language identifiers
   - Format all examples consistently
   - Clean up whitespace
   - Remove HTML formatting

Transform the following API documentation:"""),
            ("human", "{content}")
        ])

    async def extract(self, result: CrawlResult) -> List[Document]:
        try:
            content = result.markdown_v2.raw_markdown
            print(f"Processing API documentation with {len(content)} characters...")
            
            chunks = self.text_splitter.split_text(content)
            print(f"Split into {len(chunks)} chunks for parallel processing")
            
            # Process all chunks in parallel batches
            batch_size = 10
            tasks = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                task = self.llm.abatch([
                    self.preprocess_prompt.format(content=chunk)
                    for chunk in batch
                ])
                tasks.append(task)
                print(f"Queued batch {len(tasks)} for processing")
            
            print(f"Processing {len(tasks)} batches in parallel...")
            all_responses = await asyncio.gather(*tasks)
            
            preprocessed_chunks = [
                response.content 
                for batch_response in all_responses 
                for response in batch_response
            ]
            
            processed_content = "\n\n".join(preprocessed_chunks)
            print(f"Completed processing. Final length: {len(processed_content)}")
            
            return [Document(
                page_content=processed_content,
                metadata={
                    'url': result.url,
                    'type': 'api_documentation',
                    'original_size': len(content),
                    'processed_size': len(processed_content)
                }
            )]
            
        except Exception as e:
            logger.error(f"Error in extraction: {str(e)}", exc_info=True)
            return [] 