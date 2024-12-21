from typing import List
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from crawl4ai import CrawlResult
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from .base import BaseExtractor
import logging
import asyncio
from datetime import datetime

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

    async def extract(self, result: CrawlResult, batch_callback) -> List[Document]:
        try:
            extracted = await self.crawler.arun(
                url=result.url,
                extraction_strategy=self.extraction_strategy,
                chunking_strategy=self.chunking_strategy,
                wait_for="css:img[data-src]",
                delay_before_return_html=2.0
            )
            
            async def process_image_batch(images, batch_idx):
                try:
                    documents = []
                    for img in images:
                        if img.get('score', 0) > 5:
                            doc = Document(
                                page_content=f"""
                                Type: Image
                                Description: {img.get('desc', '')}
                                Alt Text: {img.get('alt', '')}
                                Context: {img.get('context', '')}
                                """.strip(),
                                metadata={
                                    'type': 'image',
                                    'src': img.get('src'),
                                    'relevance_score': img.get('score'),
                                    'priority': 'high',
                                    'batch_index': batch_idx,
                                    'extraction_date': datetime.now().isoformat()
                                }
                            )
                            documents.append(doc)
                    
                    if documents:
                        await batch_callback(documents)
                        print(f"Processed and stored image batch {batch_idx + 1}")
                        
                except Exception as e:
                    logger.error(f"Error processing image batch {batch_idx}: {str(e)}")

            async def process_video_batch(videos, batch_idx):
                try:
                    documents = []
                    for video in videos:
                        doc = Document(
                            page_content=f"""
                            Type: Video
                            Title: {video.get('title', '')}
                            Description: {video.get('description', '')}
                            Transcript: {video.get('transcript', '')}
                            """.strip(),
                            metadata={
                                'type': 'video',
                                'src': video.get('src'),
                                'duration': video.get('duration'),
                                'priority': 'high',
                                'batch_index': batch_idx,
                                'extraction_date': datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
                    
                    if documents:
                        await batch_callback(documents)
                        print(f"Processed and stored video batch {batch_idx + 1}")
                        
                except Exception as e:
                    logger.error(f"Error processing video batch {batch_idx}: {str(e)}")

            # Process images concurrently in batches
            image_tasks = []
            batch_size = 5
            images = extracted.media.get("images", [])
            
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                task = asyncio.create_task(process_image_batch(batch, i // batch_size))
                image_tasks.append(task)
            
            # Process videos concurrently in batches
            video_tasks = []
            videos = extracted.media.get("videos", [])
            
            for i in range(0, len(videos), batch_size):
                batch = videos[i:i + batch_size]
                task = asyncio.create_task(process_video_batch(batch, i // batch_size))
                video_tasks.append(task)
            
            # Wait for all media processing to complete
            if image_tasks or video_tasks:
                await asyncio.gather(*(image_tasks + video_tasks))
                print(f"Completed processing {len(image_tasks)} image batches and {len(video_tasks)} video batches")
            
            return []  # Return empty as documents are processed via callback
            
        except Exception as e:
            logger.error(f"Error in media extraction: {str(e)}")
            return [] 