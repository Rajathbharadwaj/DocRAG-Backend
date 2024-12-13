from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from typing import List
from langchain_core.documents import Document
from crawl4ai import CrawlResult, AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.chunking_strategy import TopicSegmentationChunking
import logging


logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    def __init__(self):
        self.crawler = AsyncWebCrawler()
        self.chunking_strategy = TopicSegmentationChunking(
            chunk_size=1000,
            num_keywords=3
        )
    
    @abstractmethod
    async def extract(self, result: CrawlResult) -> List[Document]:
        """Extract content with proper chunking and structure"""
        pass
    
    async def get_base_content(self, url: str) -> CrawlResult:
        """Get cleaned content using Crawl4AI"""
        return await self.crawler.arun(
            url=url,
            word_count_threshold=10,
            remove_overlay_elements=True,
            chunking_strategy=self.chunking_strategy
        )