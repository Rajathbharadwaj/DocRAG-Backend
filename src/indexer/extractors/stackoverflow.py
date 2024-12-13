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

class StackOverflowExtractor(BaseExtractor):
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
            ("system", """You are a StackOverflow content preprocessing expert. Your task is to parse and transform raw markdown documentation into LLM-friendly format.

PREPROCESSING REQUIREMENTS:

1. QUESTION FORMATTING
   - Format title as main header
   - Structure problem description clearly
   - Format error messages in code blocks
   - Highlight key requirements
   - Structure expected vs actual results
   ```question
   # [Title]
   
   ## Problem Description
   [Clear explanation]
   
   ## Error/Issue
   ```error
   [Error message]
   ```
   
   ## Expected Behavior
   [What should happen]
   
   ## Current Behavior
   [What actually happens]
   ```

2. CODE FORMATTING
   - Add proper language tags
   - Format minimal reproducible examples
   - Separate setup/execution code
   - Show output/errors clearly
   ```python
   # Setup
   import library
   
   # Problem code
   def function():
       problematic_code()
   
   # Error output
   ```error
   Stack trace or error
   ```

3. ANSWER FORMATTING
   - Structure solution steps clearly
   - Format code examples properly
   - Include explanations between code blocks
   - Show expected output
   - Add version-specific notes
   ```answer
   ## Solution
   Explanation of approach...
   
   ```python
   # Fixed code
   def function():
       working_code()
   ```
   
   ## Why it works
   [Explanation]
   ```

4. COMMENTS/DISCUSSION
   - Format clarifying questions
   - Preserve important comment threads
   - Format code snippets in comments
   - Maintain user references
   - Keep relevant debugging steps

5. METADATA
   - Preserve tags
   - Format version information
   - Keep environment details
   - Maintain links to documentation
   - Format related questions

6. STANDARDIZATION
   - Convert HTML to markdown
   - Format lists consistently
   - Structure code blocks properly
   - Clean up whitespace
   - Handle special characters

Transform the following StackOverflow content:"""),
            ("human", "{content}")
        ])

    async def extract(self, result: CrawlResult) -> List[Document]:
        try:
            content = result.markdown_v2.raw_markdown
            print(f"Processing StackOverflow content with {len(content)} characters...")
            
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
                    'type': 'stackoverflow',
                    'original_size': len(content),
                    'processed_size': len(processed_content)
                }
            )]
            
        except Exception as e:
            logger.error(f"Error in extraction: {str(e)}", exc_info=True)
            return [] 