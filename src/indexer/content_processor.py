from typing import List, Optional
from langchain_core.documents import Document
from crawl4ai import CrawlResult
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
            'code': CodeDocumentationExtractor(),
            'api': APIDocumentationExtractor(),
            'academic': AcademicExtractor(),
            'github': GitHubExtractor(),
            'stackoverflow': StackOverflowExtractor()
        }

    def _detect_content_type(self, result: CrawlResult) -> str:
        """Detect the type of content based on URL and content patterns"""
        url = result.url.lower()
        content = result.markdown_v2.raw_markdown.lower()

        # GitHub detection
        if 'github.com' in url:
            return 'github'

        # StackOverflow detection
        if 'stackoverflow.com' in url:
            return 'stackoverflow'

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
            return 'api'

        # Academic paper detection
        academic_patterns = [
            r'abstract.*introduction.*methodology',
            r'doi:',
            r'cite this paper',
            r'references?\s*\[\d+\]',
            r'arxiv'
        ]
        if any(re.search(pattern, content) for pattern in academic_patterns):
            return 'academic'

        # Default to code documentation
        return 'code'

    async def process(self, result: CrawlResult) -> List[Document]:
        """Process content using appropriate extractor"""
        try:
            # Detect content type
            content_type = self._detect_content_type(result)
            logger.info(f"Detected content type: {content_type} for URL: {result.url}")

            # Get appropriate extractor
            extractor = self.extractors.get(content_type)
            if not extractor:
                logger.warning(f"No extractor found for content type: {content_type}")
                return []

            # Extract and process content
            documents = await extractor.extract(result)
            logger.info(f"Processed {len(documents)} documents from {result.url}")
            
            return documents

        except Exception as e:
            logger.error(f"Error processing content from {result.url}: {str(e)}", exc_info=True)
            return []

    def get_supported_types(self) -> List[str]:
        """Get list of supported content types"""
        return list(self.extractors.keys())
        