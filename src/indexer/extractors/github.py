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

class GitHubExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_retries=1,
        )
        
        self.chunk_size = 16000
        self.chunk_overlap = 200
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n# ", "\n", " ", ""],
            keep_separator=True
        )
        
        self.preprocess_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a GitHub content preprocessing expert. Your task is to parse and transform raw markdown documentation into LLM-friendly format.

PREPROCESSING REQUIREMENTS:

1. README FORMATTING
   - Structure project overview clearly
   - Format installation steps
   - Highlight key features
   - Structure usage examples
   - Format badges and shields

2. ISSUE/PR FORMATTING
   - Format issue templates
   - Structure bug reports:
     ```bug-report
     Description: [what happened]
     Expected: [what should happen]
     Steps: [reproduction steps]
     Environment: [relevant details]
     ```
   - Format feature requests
   - Preserve PR descriptions and changes

3. CODE BLOCKS
   - Add language identifiers
   - Format shell commands:
     ```bash
     $ command --flag
     ```
   - Format config files with appropriate tags
   - Show output examples
   - Include comments

4. DISCUSSIONS/COMMENTS
   - Preserve thread structure
   - Format code snippets in comments
   - Handle quoted text properly
   - Maintain @mentions
   - Format links to issues/PRs

5. REPOSITORY METADATA
   - Format version information
   - Structure dependency lists
   - Format compatibility tables
   - Handle license information
   - Format contributor guidelines

6. STANDARDIZATION
   - Convert all headers to markdown syntax
   - Format task lists [ ] correctly
   - Handle emoji shortcodes
   - Preserve GitHub-specific references
   - Clean up whitespace
   - Remove HTML when possible

Transform the following GitHub content:"""),
            ("human", "{content}")
        ])

    async def extract(self, result: CrawlResult) -> List[Document]:
        try:
            content = result.markdown_v2.raw_markdown
            print(f"Processing GitHub content with {len(content)} characters...")
            
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
                    'type': 'github_content',
                    'original_size': len(content),
                    'processed_size': len(processed_content)
                }
            )]
            
        except Exception as e:
            logger.error(f"Error in extraction: {str(e)}", exc_info=True)
            return [] 