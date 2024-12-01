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
                 vector_db_path: str = "vector_store",
                 max_depth: int = 2,
                 backlink_threshold: float = 0.3):
        self.setup_logging()
        self.max_depth = max_depth
        self.backlink_threshold = backlink_threshold
        self.visited_urls: Set[str] = set()
        self.url_queue: Set[str] = set()
        self.backlinks_map: Dict[str, List[str]] = {}
        self.content_processor = ContentProcessor()
        self.background_task = None
        
        # Initialize ChromaDB
        self.setup_vector_store(vector_db_path)

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

    async def initialize_crawler(self):
        crawler_config = {
            'verbose': True,
            'headless': True,
            'wait_for_selectors': True,
            'scroll_to_bottom': True,
            'extract_strategy': 'llm',
            'chunk_size': 1000
        }
        
        self.crawler = await AsyncWebCrawler(**crawler_config).__aenter__()
        return self

    async def process_initial_url(self, url: str) -> Optional[PageContent]:
        """Process the main URL immediately and return the content."""
        self.logger.info(f"Processing initial URL: {url}")
        
        page_content = await self.content_processor.extract_page_content(
            url, self.crawler, self.backlinks_map
        )
        
        if page_content:
            self._process_page_content(page_content)
            self.visited_urls.add(url)
            self.url_queue.update(page_content.links)
            
            # Start background processing of backlinks
            self.background_task = asyncio.create_task(
                self._process_backlinks_async()
            )
            
            return page_content
        return None

    async def _process_backlinks_async(self):
        """Process backlinks asynchronously in the background."""
        current_depth = 1  # Start from depth 1 as main URL is already processed

        try:
            while current_depth < self.max_depth and self.url_queue:
                current_urls = self.url_queue.copy()
                self.url_queue.clear()

                for url in tqdm(current_urls, desc=f"Crawling depth {current_depth}"):
                    if url in self.visited_urls:
                        continue

                    page_content = await self.content_processor.extract_page_content(
                        url, self.crawler, self.backlinks_map
                    )
                    if not page_content:
                        continue

                    self._process_page_content(page_content)
                    self.url_queue.update(page_content.links)
                    self.visited_urls.add(url)

                    # Add small delay to prevent overwhelming the server
                    await asyncio.sleep(0.5)

                current_depth += 1
                
        except Exception as e:
            self.logger.error(f"Error in background processing: {str(e)}")

    def get_indexing_status(self) -> Dict:
        """Get the current status of background indexing."""
        return {
            'total_urls_processed': len(self.visited_urls),
            'urls_in_queue': len(self.url_queue),
            'is_complete': self.background_task.done() if self.background_task else True
        }

    async def cleanup(self):
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(self, 'crawler'):
            await self.crawler.__aexit__(None, None, None)

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
                self.backlinks_map[link] = []
            self.backlinks_map[link].append(page_content.url)

        # Store in vector database
        self.content_processor.store_in_vector_db(page_content, self.collection)