from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from datetime import datetime, UTC
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from model.types import ContentType
from model.config import ProcessingConfig
from model.content import PageContent
from .extractors.code import CodeDocumentationExtractor
from .extractors.media import MediaRichExtractor
from .extractors.article import ArticleExtractor
from .extractors.academic import AcademicExtractor

logger = logging.getLogger(__name__)

class ContentProcessor:
    def __init__(self):
        self.crawler = AsyncWebCrawler()
        self.extractors = {
            ContentType.DOCUMENTATION: CodeDocumentationExtractor(),
            ContentType.MEDIA: MediaRichExtractor(),
            ContentType.ARTICLE: ArticleExtractor(),
            ContentType.ACADEMIC: AcademicExtractor()
        }
        
        # LLM strategy for content type detection
        self.type_detection_strategy = LLMExtractionStrategy(
            provider="openai/gpt-4",
            instruction="""
            Analyze the content and determine its type based on:
            1. Content structure and formatting
            2. Presence of code blocks or technical documentation
            3. Media elements (images, videos, etc.)
            4. Academic markers (citations, methodology, etc.)
            Return one of: 'article', 'documentation', 'media', 'academic'
            """
        )

    def get_config_for_type(self, content_type: ContentType) -> ProcessingConfig:
        """Get optimal processing config for content type"""
        configs = {
            ContentType.ARTICLE: ProcessingConfig(
                word_count_threshold=20,
                excluded_tags=['nav', 'footer', 'aside', 'form'],
                exclude_social_media=True
            ),
            ContentType.DOCUMENTATION: ProcessingConfig(
                word_count_threshold=10,
                keep_data_attributes=True,
                process_iframes=True,
                exclude_external_links=False
            ),
            ContentType.MEDIA: ProcessingConfig(
                word_count_threshold=5,
                wait_for_lazy_load=True,
                delay_time=3.0,
                min_media_score=4
            ),
            ContentType.ACADEMIC: ProcessingConfig(
                word_count_threshold=30,
                exclude_social_media=True,
                keep_data_attributes=True
            )
        }
        return configs.get(content_type, ProcessingConfig())

    async def analyze_url(self, url: str) -> ContentType:
        """Analyze URL to determine content type using LLM"""
        try:
            # First try URL pattern matching
            if any(marker in url.lower() for marker in ['docs', 'documentation', 'api', 'reference']):
                return ContentType.DOCUMENTATION
            elif any(marker in url.lower() for marker in ['research', 'paper', 'journal', 'study']):
                return ContentType.ACADEMIC
            
            # Use crawler with LLM strategy for content analysis
            result = await self.crawler.arun(
                url=url,
                extraction_strategy=self.type_detection_strategy,
                clean_content=True
            )

            if not result.success:
                logger.warning(f"Failed to analyze URL: {result.error_message}")
                return ContentType.ARTICLE

            content_type = result.extracted_content.lower()
            return ContentType(content_type)

        except Exception as e:
            logger.error(f"Error analyzing {url}: {str(e)}")
            return ContentType.ARTICLE

    async def process_url(
        self, 
        url: str, 
        content_type: Optional[ContentType] = None,
        backlinks_map: Optional[Dict[str, List[str]]] = None
    ) -> Optional[PageContent]:
        """Process a URL with automatic content type detection if not specified"""
        try:
            logger.info(f"Processing URL: {url}")
            
            # Detect content type if not provided
            if not content_type:
                content_type = await self.analyze_url(url)
            
            # Get appropriate extractor and config
            extractor = self.extractors.get(content_type)
            if not extractor:
                logger.error(f"No extractor found for content type: {content_type}")
                return None
            
            # Extract content
            result = await self.crawler.arun(url=url)
            if not result.success:
                logger.error(f"Failed to fetch content: {result.error_message}")
                return None
                
            documents = await extractor.extract(result)
            
            # Process links
            links = self._process_links(url, result)
            
            # Create PageContent
            return PageContent(
                url=url,
                documents=documents,
                content_type=content_type,
                extraction_date=datetime.now(UTC),
                links=links,
                backlinks=backlinks_map.get(url, []) if backlinks_map else [],
                media_references=getattr(result, 'media', {}),
                metadata=result.metadata or {}
            )
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return None

    def _process_links(self, url: str, result) -> List[str]:
        """Process and filter links from crawl result"""
        links = []
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        base_path = parsed_url.path.split('/')[1] if len(parsed_url.path.split('/')) > 1 else ''

        if hasattr(result, 'links') and 'internal' in result.links:
            for link_obj in result.links['internal']:
                if isinstance(link_obj, dict) and 'href' in link_obj:
                    href = link_obj['href']
                    parsed_href = urlparse(href)
                    
                    # Filter valid internal links
                    if (href.startswith('http') and 
                        parsed_href.netloc == parsed_url.netloc and 
                        parsed_href.path.startswith(f'/{base_path}') and 
                        not parsed_href.fragment):
                        links.append(href)
                    elif (href.startswith(f'/{base_path}') and '#' not in href):
                        links.append(urljoin(base_url, href))

        return links
        