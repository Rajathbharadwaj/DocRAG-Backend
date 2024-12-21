from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from crawl4ai import CrawlResult
from .base import BaseExtractor
import logging
import json
import re
import os
from dotenv import load_dotenv
import asyncio
from datetime import datetime
import traceback

load_dotenv(override=True, dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

logger = logging.getLogger(__name__)

@dataclass
class CodeExample:
    """Represents a code example with its output"""
    title: Optional[str]
    setup: List[str]
    execution: List[str]
    output: Optional[str]
    description: Optional[str]
    source_file: Optional[str]

@dataclass
class Parameter:
    """Represents a function/class parameter"""
    name: str
    type: str
    description: str
    default: Optional[str] = None

@dataclass
class SourceReference:
    """Represents a reference to source code"""
    file_path: str
    code_content: str
    docstring: Optional[str] = None

@dataclass
class DocSection:
    """Represents a section in the documentation"""
    title: str
    description: str
    parameters: List[Parameter]
    examples: List[CodeExample]
    source_refs: List[SourceReference]
    subsections: List['DocSection']
    level: int

class CodeDocumentationExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.chunk_size = 12000
        self.chunk_overlap = 500
        
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
            ("system", """You are a documentation preprocessing expert. Your task is to parse and transform raw markdown documentation into LLM-friendly format.

PREPROCESSING REQUIREMENTS:

1. HEADER NORMALIZATION
   - Convert all header styles (===, ---, etc.) to # syntax
   - Ensure proper header hierarchy (# → ## → ###)
   - Add missing section headers where content implies them
   - Split run-on headers into proper sections

2. CODE BLOCK STANDARDIZATION
   - Convert indented code blocks to fenced blocks (```)
   - Add missing language identifiers to code blocks
   - Split mixed content code blocks into separate blocks
   - Example:
     FROM:
     ```
     some_code()
     Output: result
     ```
     TO:
     ```python
     some_code()
     ```
     ```plaintext
     Output: result
     ```

3. LIST FORMATTING
   - Convert all list styles to consistent markdown bullets (-)
   - Proper indentation for nested lists
   - Add line breaks between list items for clarity

4. TABLE CLEANUP
   - Convert ASCII/alternative tables to standard markdown tables
   - Add missing headers
   - Align columns properly

5. INLINE ELEMENT STANDARDIZATION
   - Convert HTML tags to markdown syntax
   - Standardize emphasis markers (* or _)
   - Fix broken inline code blocks
   - Normalize link formats

6. WHITESPACE MANAGEMENT
   - Add consistent line breaks between sections
   - Remove excessive blank lines
   - Ensure proper indentation
   - Preserve meaningful whitespace in code blocks

7. CONTENT SECTIONING
   - Group related content under appropriate headers
   - Break up long paragraphs
   - Add missing section titles
   - Ensure logical content flow

YOUR TASK:
0. Give a summary of the following markdown content:
1. Parse the provided markdown
2. Apply all preprocessing rules
3. Maintain the original meaning and structure
4. Output clean, consistent, LLM-friendly markdown
5. Preserve all technical information
6. Remove any markdown syntax that might confuse LLMs

Transform the following markdown:"""
),
            ("human", "{content}")
        ])

    async def extract(self, result: CrawlResult, batch_callback) -> List[Document]:
        try:
            content = result.markdown_v2.raw_markdown
            print(f"Processing code documentation with {len(content)} characters...")
            
            chunks = self.text_splitter.split_text(content)
            print(f"Split into {len(chunks)} chunks for parallel processing")
            
            # Store all documents if no callback provided
            all_documents = []
            
            async def process_code_batch(batch_chunks, batch_idx):
                try:
                    # Process code chunks in parallel
                    responses = await self.llm.abatch([
                        self.preprocess_prompt.format(content=chunk)
                        for chunk in batch_chunks
                    ])
                    
                    # Create documents from responses
                    documents = []
                    for j, response in enumerate(responses):
                        if hasattr(response, 'content'):
                            try:
                                # First try to parse as JSON
                                
                                doc = Document(
                                    page_content= response.content,
                                    metadata={
                                        'url': result.url,
                                        'type': 'code_documentation',
                                        'language': 'unknown',
                                        'chunk_index': batch_idx * batch_size + j,
                                        'extraction_date': datetime.now().isoformat(),
                                        'code_type': 'documentation'
                                    }
                                )
                                documents.append(doc)
                                
                            except Exception as e:
                                logger.error(
                                    f"Error processing response in batch {batch_idx}, chunk {j}:\n"
                                    f"Error: {str(e)}\n"
                                    f"Traceback:\n{traceback.format_exc()}\n"
                                    f"Response: {response if response else 'None'}"
                                )
                                continue
                    
                    # Store batch immediately if callback provided, otherwise collect
                    if documents:
                        if batch_callback:
                            try:
                                await batch_callback(documents)
                                print(f"Processed and stored code batch {batch_idx + 1}")
                            except Exception as e:
                                logger.error(
                                    f"Error in batch_callback for batch {batch_idx}:\n"
                                    f"Error: {str(e)}\n"
                                    f"Traceback:\n{traceback.format_exc()}"
                                )
                        else:
                            all_documents.extend(documents)
                            print(f"Processed code batch {batch_idx + 1} (storing in memory)")
                        
                except Exception as e:
                    logger.error(
                        f"Error processing code batch {batch_idx}:\n"
                        f"Error: {str(e)}\n"
                        f"Traceback:\n{traceback.format_exc()}\n"
                        f"Batch size: {len(batch_chunks)}"
                    )
            
            # Create tasks for each batch
            tasks = []
            batch_size = 5
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                task = asyncio.create_task(process_code_batch(batch, i // batch_size))
                tasks.append(task)
            
            # Wait for all batches to complete
            if tasks:
                try:
                    await asyncio.gather(*tasks)
                    print(f"Completed processing {len(tasks)} code batches")
                except Exception as e:
                    logger.error(
                        f"Error in task gathering:\n"
                        f"Error: {str(e)}\n"
                        f"Traceback:\n{traceback.format_exc()}"
                    )
            
            # Return collected documents if no callback was provided
            return all_documents if not batch_callback else []
            
        except Exception as e:
            logger.error(
                f"Error in code extraction:\n"
                f"Error: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}\n"
                f"URL: {result.url}\n"
                f"Content length: {len(content) if content else 'None'}"
            )
            return []

    def _format_sections(self, content: str) -> str:
        """Format the processed content for final output"""
        # Add any additional formatting if needed
        return content