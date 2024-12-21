import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crawl4ai import CrawlResult
from .base import BaseExtractor
import logging
from datetime import datetime
import traceback

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
   {{
     "key": "value"
   }}
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

    async def extract(self, result: CrawlResult, batch_callback=None) -> List[Document]:
        try:
            content = result.markdown_v2.raw_markdown
            print(f"Processing API documentation with {len(content)} characters...")
            
            chunks = self.text_splitter.split_text(content)
            print(f"Split into {len(chunks)} chunks for parallel processing")
            
            # Store all documents if no callback provided
            all_documents = []
            
            async def process_api_batch(batch_chunks, batch_idx):
                try:
                    # Process API chunks in parallel
                    responses = await self.llm.abatch([
                        self.preprocess_prompt.format(content=chunk)
                        for chunk in batch_chunks
                    ])
                    
                    # Create documents from responses
                    documents = []
                    for j, response in enumerate(responses):
                        if hasattr(response, 'content'):
                            try:
                                doc = Document(
                                    page_content=response.content,
                                    metadata={
                                        'url': result.url,
                                        'type': 'api_documentation',
                                        'chunk_index': batch_idx * batch_size + j,
                                        'extraction_date': datetime.now().isoformat()
                                    }
                                )
                                documents.append(doc)
                                
                            except Exception as e:
                                logger.error(
                                    f"Error processing API response in batch {batch_idx}, chunk {j}:\n"
                                    f"Error: {str(e)}\n"
                                    f"Traceback:\n{traceback.format_exc()}"
                                )
                                continue
                    
                    # Store batch immediately if callback provided, otherwise collect
                    if documents:
                        if batch_callback:
                            try:
                                await batch_callback(documents)
                                print(f"Processed and stored API batch {batch_idx + 1}")
                            except Exception as e:
                                logger.error(
                                    f"Error in batch_callback for API batch {batch_idx}:\n"
                                    f"Error: {str(e)}\n"
                                    f"Traceback:\n{traceback.format_exc()}"
                                )
                        else:
                            all_documents.extend(documents)
                            print(f"Processed API batch {batch_idx + 1} (storing in memory)")
                        
                except Exception as e:
                    logger.error(
                        f"Error processing API batch {batch_idx}:\n"
                        f"Error: {str(e)}\n"
                        f"Traceback:\n{traceback.format_exc()}"
                    )
            
            # Create tasks for each batch
            tasks = []
            batch_size = 5
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                task = asyncio.create_task(process_api_batch(batch, i // batch_size))
                tasks.append(task)
            
            # Wait for all batches to complete
            if tasks:
                await asyncio.gather(*tasks)
                print(f"Completed processing {len(tasks)} API batches")
            
            # Return collected documents if no callback was provided
            return all_documents if not batch_callback else []
            
        except Exception as e:
            logger.error(
                f"Error in API extraction:\n"
                f"Error: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            return [] 