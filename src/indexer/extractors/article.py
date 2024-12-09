from typing import List
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from crawl4ai import CrawlResult
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from ...model.content import ArticleContent, ArticleSection, ArticleMetadata
from .base import BaseExtractor
import logging

logger = logging.getLogger(__name__)

class ArticleExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.extraction_strategy = LLMExtractionStrategy(
            provider="openai/gpt-4",
            instruction="""
            Extract article content with emphasis on:
            1. Main content sections
            2. Article metadata (author, date, etc.)
            3. Key points and summaries
            4. Related content and references
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
            
            # Process article sections
            for section in extracted.sections:
                doc = Document(
                    page_content=section.content,
                    metadata={
                        'type': 'article_section',
                        'title': section.title,
                        'heading_level': section.heading_level,
                        'position': section.position,
                        'priority': 'high'
                    }
                )
                documents.append(doc)

            # Add metadata document
            if extracted.metadata:
                doc = Document(
                    page_content=str(extracted.metadata),
                    metadata={
                        'type': 'article_metadata',
                        'author': extracted.metadata.author,
                        'publish_date': extracted.metadata.publish_date,
                        'priority': 'medium'
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error in article extraction: {str(e)}")
            return [] 