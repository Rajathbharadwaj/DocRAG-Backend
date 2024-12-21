from typing import List, Optional, Callable, Awaitable
from langchain_core.documents import Document
from crawl4ai import CrawlResult

from model.types import ContentType
from .extractors.code import CodeDocumentationExtractor
from .extractors.api import APIDocumentationExtractor
from .extractors.academic import AcademicExtractor
from .extractors.github import GitHubExtractor
from .extractors.stackoverflow import StackOverflowExtractor
import logging
import re

logger = logging.getLogger(__name__)

class ContentProcessor:
    def __init__(self):
        self.extractors = {
            ContentType.DOCUMENTATION: CodeDocumentationExtractor(),
            ContentType.API: APIDocumentationExtractor(),
            ContentType.ACADEMIC: AcademicExtractor(),
            ContentType.GITHUB: GitHubExtractor(),
            ContentType.STACKOVERFLOW: StackOverflowExtractor()
        }

    def _detect_content_type(self, result: CrawlResult) -> ContentType:
        """Detect the type of content based on URL and content patterns"""
        url = result.url.lower()
        content = result.markdown_v2.raw_markdown.lower()

        # GitHub detection
        if 'github.com' in url:
            return ContentType.GITHUB

        # StackOverflow detection
        if 'stackoverflow.com' in url:
            return ContentType.STACKOVERFLOW

        # API documentation detection
        api_patterns = [
            r'api.*reference',
            r'api.*documentation',
            r'endpoints',
            r'(get|post|put|delete).*requests?',
            r'rest.*api',
            r'graphql.*api'
        ]
        if any(re.search(pattern, content) for pattern in api_patterns):
            return ContentType.API

        # Academic paper detection
        academic_patterns = [
            r'abstract.*introduction.*methodology',
            r'doi:',
            r'cite this paper',
            r'references?\s*\[\d+\]',
            r'arxiv'
        ]
        if any(re.search(pattern, content) for pattern in academic_patterns):
            return ContentType.ACADEMIC

        # Default to code documentation
        return ContentType.DOCUMENTATION

    async def process(
        self, 
        result: CrawlResult, 
        content_type: Optional[ContentType] = None,
        batch_callback: Optional[Callable[[List[Document]], Awaitable[None]]] = None
    ) -> List[Document]:
        """
        Process content using appropriate extractor
        
        Args:
            result: The crawl result to process
            content_type: Optional content type override
            batch_callback: Async callback function to process document batches
        """
        try:
            # Detect content type if not provided
            if not content_type:
                content_type = self._detect_content_type(result)
            logger.info(f"Processing content type: {content_type} for URL: {result.url}")

            # Get appropriate extractor
            extractor = self.extractors.get(content_type)
            if not extractor:
                logger.warning(f"No extractor found for content type: {content_type}")
                return []

            # Extract and process content with batch callback
            documents = await extractor.extract(
                result=result,
                batch_callback=batch_callback
            )
            
            logger.info(f"Processed content from {result.url}")
            return documents

        except Exception as e:
            logger.error(f"Error processing content from {result.url}: {str(e)}", exc_info=True)
            return []

    def get_supported_types(self) -> List[str]:
        """Get list of supported content types"""
        return list(self.extractors.keys())
        