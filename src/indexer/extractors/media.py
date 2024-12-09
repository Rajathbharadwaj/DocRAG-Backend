from typing import List
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from crawl4ai import CrawlResult
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from .base import BaseExtractor
import logging

logger = logging.getLogger(__name__)

class MediaRichExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.extraction_strategy = LLMExtractionStrategy(
            provider="openai/gpt-4",
            instruction="""
            Extract media-rich content with emphasis on:
            1. Images with descriptions and context
            2. Videos with transcripts and summaries
            3. Audio content with transcripts
            4. Interactive media elements
            5. Captions and alt text
            6. Format all the content in a way that is easy to read and understand. Add blocks to identify the content. For example, if there is an image, add a block that says "Image: [image description]"
            NOTE: The content you're reading is a html page, so make sure to take out all the unncessary page elements.
            """
        )

    async def extract(self, result: CrawlResult) -> List[Document]:
        documents = []
        try:
            # Use LLM extraction with media focus
            extracted = await self.crawler.arun(
                url=result.url,
                extraction_strategy=self.extraction_strategy,
                chunking_strategy=self.chunking_strategy,
                wait_for="css:img[data-src]",  # Wait for lazy images
                delay_before_return_html=2.0
            )
            
            # Process images
            for image in extracted.media.get("images", []):
                if image.get('score', 0) > 5:  # Only include relevant images
                    doc = Document(
                        page_content=f"""
                        Type: Image
                        Description: {image.get('desc', '')}
                        Alt Text: {image.get('alt', '')}
                        Context: {image.get('context', '')}
                        """,
                        metadata={
                            'type': 'image',
                            'src': image.get('src'),
                            'relevance_score': image.get('score'),
                            'priority': 'high'
                        }
                    )
                    documents.append(doc)
            
            # Process videos
            for video in extracted.media.get("videos", []):
                doc = Document(
                    page_content=f"""
                    Type: Video
                    Title: {video.get('title', '')}
                    Description: {video.get('description', '')}
                    Transcript: {video.get('transcript', '')}
                    """,
                    metadata={
                        'type': 'video',
                        'src': video.get('src'),
                        'duration': video.get('duration'),
                        'priority': 'high'
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error in media extraction: {str(e)}")
            return [] 