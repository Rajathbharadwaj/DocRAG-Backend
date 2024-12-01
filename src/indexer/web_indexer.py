from typing import Optional, Dict, List, Set
import logging
from urllib.parse import urlparse
import chromadb
from chromadb.utils import embedding_functions
from crawl4ai import AsyncWebCrawler
from .content_processor import ContentProcessor
from models import PageContent
from tqdm import tqdm
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
            'page_timeout': 60000,  # 60 seconds for good measure
            
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
            'exclude_external_links': False,  # We want all links
            'exclude_domains': [],  # No domain exclusions
            
            # Clean content
            'excluded_tags': ['nav', 'aside'],
            'keep_data_attributes': False
        }
        
        self.crawler = await AsyncWebCrawler(**crawler_config).__aenter__()
        print("[DEBUG] Crawler initialized with config:", crawler_config)
        return self

    async def process_initial_url(self, 
                                url: str,
                                max_depth: Optional[int] = None,
                                backlink_threshold: Optional[float] = None,
                                doc_name: Optional[str] = None) -> Optional[PageContent]:
        """Process the main URL immediately and start background processing."""
        print(f"\n[DEBUG] Starting to process URL: {url}")
        
        try:
            # Override instance values if provided
            if max_depth is not None:
                self.max_depth = max_depth
            if backlink_threshold is not None:
                self.backlink_threshold = backlink_threshold
            
            print(f"[DEBUG] Calling content processor for URL: {url}")
            page_content = await self.content_processor.extract_page_content(
                url=url,
                crawler=self.crawler,  # Now using self.crawler
                backlinks_map=self.backlinks_map
            )
            
            if page_content:
                print(f"[DEBUG] Successfully extracted content from {url}")
                # Store in vector DB
                if self.rag_engine:
                    print("[DEBUG] Adding documents to RAG engine")
                    await self.rag_engine.add_documents([page_content.document])
                
                self.visited_urls.add(url)
                self.url_queue.update(page_content.links)
                print(f"[DEBUG] Found {len(page_content.links)} links to process")
                
                # Update backlinks map with the current page's outgoing links
                for link in page_content.links:
                    if link not in self.backlinks_map:
                        self.backlinks_map[link] = set()
                    self.backlinks_map[link].add(url)  # Current page links to 'link'
                
                print(f"[DEBUG] Updated backlinks map: {self.backlinks_map}")
                
                # Start background processing
                print("[DEBUG] Starting background processing")
                self.background_task = asyncio.create_task(
                    self._process_backlinks_async(doc_name or "default")
                )
                
                return page_content
            else:
                print(f"[DEBUG] Failed to extract content from {url}")
                return None
                
        except Exception as e:
            print(f"[DEBUG] Error processing URL: {str(e)}")
            raise

    async def _process_backlinks_async(self, doc_name: str):
        """Process remaining URLs in background."""
        try:
            print(f"[DEBUG] Starting background processing for {doc_name}")
            print(f"[DEBUG] Initial queue size: {len(self.url_queue)}")
            print(f"[DEBUG] URLs in queue: {self.url_queue}")
            
            current_depth = 1
            while current_depth < self.max_depth and self.url_queue:
                print(f"[DEBUG] Processing depth {current_depth}")
                urls_at_depth = self.url_queue.copy()
                self.url_queue.clear()
                
                for url in urls_at_depth:
                    if url in self.visited_urls:
                        print(f"[DEBUG] Skipping already visited URL: {url}")
                        continue
                        
                    # Check backlink threshold
                    backlink_count = len(self.backlinks_map.get(url, set()))
                    print(f"[DEBUG] URL {url} has {backlink_count} backlinks")
                    if backlink_count / len(self.visited_urls) < self.backlink_threshold:
                        print(f"[DEBUG] URL {url} below backlink threshold")
                        continue
                    
                    print(f"[DEBUG] Processing URL: {url}")
                    page_content = await self.content_processor.extract_page_content(
                        url=url,
                        crawler=self.crawler,
                        backlinks_map=self.backlinks_map
                    )
                    
                    if page_content and self.rag_engine:
                        await self.rag_engine.add_documents([page_content.document])
                        self.visited_urls.add(url)
                        self.url_queue.update(page_content.links)
                        print(f"[DEBUG] Added {len(page_content.links)} new URLs to queue")
                
                    # Update backlinks for newly discovered links
                    for link in page_content.links:
                        if link not in self.backlinks_map:
                            self.backlinks_map[link] = set()
                        self.backlinks_map[link].add(url)
                
                    print(f"[DEBUG] Current backlinks map: {self.backlinks_map}")
                
                current_depth += 1
                
            print(f"[DEBUG] Background processing complete. Processed {len(self.visited_urls)} URLs")
                
        except Exception as e:
            print(f"[DEBUG] Error in background processing: {str(e)}")
            self.logger.error(f"Error in background processing: {e}")

    def get_indexing_status(self) -> Dict:
        """Get current indexing status."""
        return {
            "is_processing": self.background_task is not None and not self.background_task.done(),
            "urls_processed": len(self.visited_urls),
            "urls_queued": len(self.url_queue),
            "current_depth": len(self.visited_urls) > 0,  # Simplified depth calculation
        }

    async def cleanup(self):
        """Cleanup resources."""
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass

    def setup_vector_store(self, vector_db_path: str):
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key="your_openai_api_key",
            model_name="text-embedding-3-small"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="web_content",
            embedding_function=self.embedding_function
        )

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def is_valid_url(self, url: str, base_domain: str) -> bool:
        try:
            parsed = urlparse(url)
            base_parsed = urlparse(base_domain)
            return parsed.netloc == base_parsed.netloc and parsed.scheme in ['http', 'https']
        except:
            return False

    def _process_page_content(self, page_content: PageContent):
        # Update backlinks
        for link in page_content.links:
            if link not in self.backlinks_map:
                self.backlinks_map[link] = set()
            self.backlinks_map[link].add(page_content.url)

        # Store in vector database
        self.content_processor.store_in_vector_db(page_content, self.collection)