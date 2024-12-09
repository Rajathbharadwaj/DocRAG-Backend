from typing import List, Dict
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from crawl4ai import CrawlResult
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from ...model.content import CodeBlock, APIEndpoint
from .base import BaseExtractor
import logging

logger = logging.getLogger(__name__)

class CodeDocumentationExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.extraction_strategy = LLMExtractionStrategy(
            provider="openai/gpt-4",
            instruction="""
            Extract documentation content with emphasis on:
            1. Code blocks with context and language
            2. API endpoints and specifications
            3. Technical explanations
            4. Configuration examples
            5. Format all the content in a way that is easy to read and understand. Add blocks to identify the content. For example, if there is an image, add a block that says "Image: [image description]"
            NOTE: The content you're reading is a html page, so make sure to take out all the unncessary page elements.
            """
        )

    async def extract(self, result: CrawlResult) -> List[Document]:
        documents = []
        try:
            # Use LLM extraction
            extracted = await self.crawler.arun(
                url=result.url,
                extraction_strategy=self.extraction_strategy,
                chunking_strategy=self.chunking_strategy
            )
            
            # Process code blocks
            for block in extracted.code_blocks:
                doc = Document(
                    page_content=block.content,
                    metadata={
                        'type': 'code',
                        'language': block.language,
                        'context': block.context,
                        'section': block.section,
                        'priority': 'high'
                    }
                )
                documents.append(doc)

            # Process API endpoints
            for endpoint in extracted.api_endpoints:
                doc = Document(
                    page_content=f"{endpoint.method} {endpoint.endpoint}\n{endpoint.description}",
                    metadata={
                        'type': 'api',
                        'method': endpoint.method,
                        'endpoint': endpoint.endpoint,
                        'parameters': endpoint.parameters,
                        'priority': 'high'
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error in code documentation extraction: {str(e)}")
            return [] 