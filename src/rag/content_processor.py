from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import List, Dict, Optional
from crawl4ai import AsyncWebCrawler
from langchain_core.documents import Document
import logging
from enum import Enum
from ..indexer.extraction_strategies import (
    CodeDocumentationExtractor,
    MediaRichExtractor,
    ArticleExtractor,
    AcademicExtractor
)

logger = logging.getLogger(__name__)

class ContentType(Enum):
    ARTICLE = "article"
    DOCUMENTATION = "documentation"
    MEDIA = "media"
    ACADEMIC = "academic"

@dataclass
class ProcessingConfig:
    """Configuration for content processing"""
    word_count_threshold: int = 10
    excluded_tags: List[str] = field(default_factory=lambda: ['nav', 'footer', 'aside'])
    remove_overlay_elements: bool = True
    wait_for_lazy_load: bool = True
    delay_time: float = 2.0
    exclude_external_links: bool = False
    exclude_social_media: bool = True
    keep_data_attributes: bool = False
    process_iframes: bool = True
    min_media_score: int = 5

class ContentProcessor:
    def __init__(self):
        self.crawler = AsyncWebCrawler()
        self.extractors = {
            ContentType.DOCUMENTATION: CodeDocumentationExtractor(),
            ContentType.MEDIA: MediaRichExtractor(),
            ContentType.ARTICLE: ArticleExtractor(),
            ContentType.ACADEMIC: AcademicExtractor()
        }

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

    async def process_url(
        self, 
        url: str, 
        content_type: ContentType
    ) -> List[Document]:
        """Process URL using appropriate extractor and config"""
        try:
            # Get appropriate extractor
            extractor = self.extractors.get(content_type)
            if not extractor:
                logger.warning(f"No specific extractor for {content_type}, using ArticleExtractor")
                extractor = ArticleExtractor()

            # Extract content
            documents = await extractor.extract(url)
            
            # Add processing metadata
            for doc in documents:
                doc.metadata.update({
                    'processor_version': '1.0',
                    'content_type': content_type.value,
                    'extraction_date': datetime.now(UTC).isoformat()
                })

            logger.info(
                f"Processed {url} as {content_type.value}: "
                f"extracted {len(documents)} documents"
            )

            return documents

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            raise

    async def analyze_url(self, url: str) -> ContentType:
        """Analyze URL to determine content type"""
        try:
            # Get initial content
            result = await self.crawler.arun(
                url=url,
                word_count_threshold=10,
                remove_overlay_elements=True
            )

            # Analyze content patterns
            patterns = {
                'code_patterns': len(result.code_blocks),
                'media_count': len(result.media.get('images', [])) + 
                             len(result.media.get('videos', [])) +
                             len(result.media.get('audios', [])),
                'academic_markers': any(
                    marker in result.metadata.get('keywords', []).lower()
                    for marker in ['research', 'paper', 'journal', 'study']
                ),
                'documentation_markers': any(
                    marker in url.lower()
                    for marker in ['docs', 'documentation', 'api', 'reference']
                )
            }

            # Determine content type
            if patterns['documentation_markers'] or patterns['code_patterns'] > 3:
                return ContentType.DOCUMENTATION
            elif patterns['media_count'] > 5:
                return ContentType.MEDIA
            elif patterns['academic_markers']:
                return ContentType.ACADEMIC
            else:
                return ContentType.ARTICLE

        except Exception as e:
            logger.error(f"Error analyzing {url}: {str(e)}")
            return ContentType.ARTICLE  # Default to article

    def _process_metadata(self, metadata: Dict) -> Document:
        """Create structured metadata document"""
        return Document(
            page_content=f"""
            Title: {metadata.get('title', '')}
            Description: {metadata.get('description', '')}
            Keywords: {', '.join(metadata.get('keywords', []))}
            Author: {metadata.get('author', '')}
            Published: {metadata.get('published_date', '')}
            Modified: {metadata.get('modified_date', '')}
            Language: {metadata.get('language', '')}
            """.strip(),
            metadata={
                'type': 'metadata',
                'language': metadata.get('language'),
                'has_schema': bool(metadata.get('schema_org'))
            }
        )

    async def batch_process(
        self, 
        urls: List[str],
        content_type: Optional[ContentType] = None
    ) -> Dict[str, List[Document]]:
        """Process multiple URLs with automatic type detection"""
        results = {}
        
        for url in urls:
            try:
                # Determine content type if not specified
                url_type = content_type or await self.analyze_url(url)
                
                # Process URL
                documents = await self.process_url(url, url_type)
                results[url] = documents
                
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                results[url] = []
                
        return results