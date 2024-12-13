from typing import Optional, Dict, List, Set
import logging
from urllib.parse import urlparse
import chromadb
from chromadb.utils import embedding_functions
from crawl4ai import AsyncWebCrawler
from model.types import ContentType
from .content_processor import ContentProcessor
from model.content import PageContent
import asyncio

class WebIndexer:
    def __init__(self, 
                 max_depth: int = 2, 
                 backlink_threshold: float = 0.3,
                 rag_engine = None):
        self.max_depth = max_depth
        self.backlink_threshold = backlink_threshold
        self.rag_engine = rag_engine
        self.visited_urls: Set[str] = set()
        self.url_queue: Set[str] = set()
        self.backlinks_map: Dict[str, Set[str]] = {}
        self.background_task: Optional[asyncio.Task] = None
        self.content_processor = ContentProcessor()
        self.logger = logging.getLogger(__name__)

    async def initialize_crawler(self):
        """Initialize crawler with proper configuration"""
        crawler_config = {
            'verbose': True,
            'headless': True,
            'browser_type': 'chromium',
            'page_timeout': 60000,  # 60 seconds
            
            # Content processing
            'word_count_threshold': 10,
            'remove_overlay_elements': True,
            'exclude_social_media_links': True,
            
            # Anti-detection
            'magic': True,
            'simulate_user': True,
            
            # Dynamic content handling
            'delay_before_return_html': 2.0,
            'wait_for': '.content',  # Wait for main content
            
            # Link handling
            'exclude_external_links': False,
            'exclude_domains': [],
            
            # Clean content
            'excluded_tags': ['nav', 'aside'],
            'keep_data_attributes': False
        }
        
        self.crawler = await AsyncWebCrawler(**crawler_config).__aenter__()
        print("[DEBUG] Crawler initialized with config:", crawler_config)
        return self

    async def process_initial_url(self, 
                                url: str,
                                content_type: Optional[ContentType] = None,
                                max_depth: Optional[int] = None,
                                max_links: Optional[int] = None,
                                backlink_threshold: Optional[float] = None) -> Optional[PageContent]:
        """Process the main URL and start background processing"""
        try:
            if max_depth is not None:
                self.max_depth = max_depth
            if backlink_threshold is not None:
                self.backlink_threshold = backlink_threshold
            self.max_links = max_links
            
            # Crawl and process URL
            async with self.crawler as crawler:
                crawl_result = await crawler.arun(
                    url=url,
                    clean_content=True,
                    bypass_cache=True
                )
            documents = await self.content_processor.process(crawl_result, content_type)
            
            if documents:
                # Limit number of links if max_links is set
                links_to_process = crawl_result.links
                if self.max_links:
                    links_to_process = list(links_to_process)[:self.max_links]
                
                # Store in vector DB
                if self.rag_engine:
                    await self.rag_engine.add_documents(documents)
                
                # Update tracking with limited links
                self.visited_urls.add(url)
                self.url_queue.update(links_to_process)
                
                # Update backlinks for limited set
                for link in links_to_process:
                    if link not in self.backlinks_map:
                        self.backlinks_map[link] = set()
                    self.backlinks_map[link].add(url)
                
                # Start background processing
                self.background_task = asyncio.create_task(
                    self._process_backlinks_async()
                )
                
                return PageContent(
                    url=url,
                    documents=documents,
                    links=links_to_process
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing URL: {str(e)}")
            raise

    async def _process_backlinks_async(self):
        """Process remaining URLs in background using efficient batching"""
        try:
            print(f"[DEBUG] Starting batch background processing")
            
            current_depth = 1
            while current_depth < self.max_depth and self.url_queue:
                print(f"[DEBUG] Processing depth {current_depth}")
                urls_at_depth = self.url_queue.copy()
                self.url_queue.clear()
                
                # Filter URLs based on backlink threshold
                urls_to_process = [
                    url for url in urls_at_depth
                    if url not in self.visited_urls and
                    len(self.backlinks_map.get(url, set())) / len(self.visited_urls) >= self.backlink_threshold
                ]
                
                # Process in larger batches
                batch_size = 10  # Increased from 5
                for i in range(0, len(urls_to_process), batch_size):
                    batch = urls_to_process[i:i + batch_size]
                    print(f"[DEBUG] Processing batch {i//batch_size + 1} of {(len(urls_to_process) + batch_size - 1)//batch_size}")
                    
                    # Crawl all URLs in batch
                    async with self.crawler as crawler:
                        crawl_tasks = [crawler.arun(url, clean_content=True, bypass_cache=True) for url in batch]
                        crawl_results = await asyncio.gather(*crawl_tasks)
                    
                    # Process all results in batch
                    process_tasks = [
                        self.content_processor.process(result)
                        for result in crawl_results
                        if result is not None
                    ]
                    all_documents = await asyncio.gather(*process_tasks)
                    
                    # Batch update RAG engine
                    if self.rag_engine:
                        documents_to_add = [
                            doc for docs in all_documents 
                            for doc in docs 
                            if docs
                        ]
                        if documents_to_add:
                            await self.rag_engine.add_documents(documents_to_add)
                    
                    # Batch update tracking
                    for url, result in zip(batch, crawl_results):
                        if result and any(docs for docs in all_documents):
                            self.visited_urls.add(url)
                            self.url_queue.update(result.links)
                            
                            # Update backlinks
                            for link in result.links:
                                if link not in self.backlinks_map:
                                    self.backlinks_map[link] = set()
                                self.backlinks_map[link].add(url)
                    
                current_depth += 1
                print(f"[DEBUG] Completed depth {current_depth}")
                
        except Exception as e:
            self.logger.error(f"Error in batch background processing: {str(e)}")

    async def _process_single_url(self, url: str):
        """Process a single URL"""
        try:
            crawl_result = await self.crawler.crawl(url)
            documents = await self.content_processor.process(crawl_result)
            
            if documents and self.rag_engine:
                await self.rag_engine.add_documents(documents)
                self.visited_urls.add(url)
                self.url_queue.update(crawl_result.links)
                
                # Update backlinks
                for link in crawl_result.links:
                    if link not in self.backlinks_map:
                        self.backlinks_map[link] = set()
                    self.backlinks_map[link].add(url)
            
        except Exception as e:
            self.logger.error(f"Error processing URL {url}: {str(e)}")

    def get_indexing_status(self) -> Dict:
        """Get current indexing status"""
        return {
            "is_processing": self.background_task is not None and not self.background_task.done(),
            "urls_processed": len(self.visited_urls),
            "urls_queued": len(self.url_queue),
            "current_depth": len(self.visited_urls) > 0,
        }

    async def cleanup(self):
        """Cleanup resources"""
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass