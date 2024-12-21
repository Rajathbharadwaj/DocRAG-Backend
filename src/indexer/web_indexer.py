from typing import List, Dict, Set, Optional
import logging
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler
from model.types import ContentType
from .content_processor import ContentProcessor
from model.content import PageContent
import asyncio
from langchain.prompts import PromptTemplate
from rag.vectorstore_engine import VectorStoreEngine
from datetime import datetime
from langchain_openai import ChatOpenAI
import traceback

class WebIndexer:
    def __init__(
        self, 
        doc_name: str, 
        max_links: Optional[int] = 25,
        use_serverless: bool = True,
        region: str = "us-east-1"
    ):
        self.doc_name = doc_name
        self.max_links = max_links
        self.content_type: Optional[ContentType] = None
        
        # Initialize VectorStore with doc_name
        self.vector_store = VectorStoreEngine(
            doc_name=doc_name,
            use_serverless=use_serverless,
            region=region
        )
        
        self.visited_urls: Set[str] = set()
        self.url_queue: Set[str] = set()
        self.background_task: Optional[asyncio.Task] = None
        self.content_processor = ContentProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Summary prompt template
        self.summary_prompt = PromptTemplate(
            template="""Given the following document content, provide a concise and accurate summary in 2-3 sentences
Content:
{content}

Summary:""",
            input_variables=["content"]
        )
        
        # Initialize LLM for summaries
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=200
        )

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

    def _clean_metadata(self, metadata: dict) -> dict:
        """Clean metadata to ensure all values are valid for Pinecone"""
        cleaned = {}
        for key, value in metadata.items():
            if value is not None:
                # Convert empty strings to "unknown"
                if isinstance(value, str) and not value.strip():
                    cleaned[key] = "unknown"
                # Convert empty lists to ["unknown"]
                elif isinstance(value, list) and not value:
                    cleaned[key] = ["unknown"]
                else:
                    cleaned[key] = value
            else:
                # Replace None values with appropriate defaults
                if key in ['author', 'title', 'description']:
                    cleaned[key] = "unknown"
                elif key == 'keywords':
                    cleaned[key] = ["unknown"]
                elif key == 'last_modified':
                    cleaned[key] = datetime.now().isoformat()
                else:
                    cleaned[key] = "not_specified"
        return cleaned

    async def process_initial_url(self, url: str, content_type: ContentType) -> Optional[PageContent]:
        """Process the main URL immediately and queue background processing"""
        try:
            self.initial_url = url
            self.content_type = content_type
            
            # Immediately crawl and process the initial URL
            async with self.crawler as crawler:
                crawl_result = await crawler.arun(
                    url=url,
                    clean_content=True,
                    bypass_cache=True
                )
                
            # Debug: Log the raw crawl result structure
            self.logger.debug(f"Crawl result attributes: {dir(crawl_result)}")
            self.logger.debug(f"Has links attribute: {hasattr(crawl_result, 'links')}")
            if hasattr(crawl_result, 'links'):
                self.logger.debug(f"Links type: {type(crawl_result.links)}")
                self.logger.debug(f"Raw links data: {crawl_result.links}")
                
            # Process and store initial page content using helper
            documents = await self._process_content(crawl_result)
            if documents:
                self.visited_urls.add(url)
                
                # Collect unique links for background processing
                unique_links = set()
                if hasattr(crawl_result, 'links'):
                    if isinstance(crawl_result.links, dict):
                        initial_domain = urlparse(url).netloc
                        all_links = []
                        
                        # Add internal links first
                        if 'internal' in crawl_result.links:
                            internal_links = [
                                link['href'] for link in crawl_result.links['internal']
                                if isinstance(link, dict) and 'href' in link
                                and urlparse(link['href']).netloc == initial_domain
                            ]
                            self.logger.debug(f"Found {len(internal_links)} internal links: {internal_links}")
                            all_links.extend(internal_links)
                        
                        # Then add external links
                        if 'external' in crawl_result.links:
                            external_links = [
                                link['href'] for link in crawl_result.links['external']
                                if isinstance(link, dict) and 'href' in link
                            ]
                            self.logger.debug(f"Found {len(external_links)} external links: {external_links}")
                            all_links.extend(external_links)
                        
                        # Take first max_links unique links
                        unique_links = set(all_links[:self.max_links])
                        self.logger.debug(f"Final unique links to process: {unique_links}")
                
                self.logger.info(f"Initial page processed. Found {len(unique_links)} unique links")
                
                # Start background processing of unique links
                if unique_links:
                    self.url_queue = unique_links
                    self.logger.info(f"Starting background processing for {len(unique_links)} links: {unique_links}")
                    # Create and start the background task
                    self.background_task = asyncio.create_task(self._process_background_links())
                    # Add task done callback to handle completion
                    self.background_task.add_done_callback(self._on_background_task_complete)
                else:
                    self.logger.warning("No links found for background processing")
                
                return PageContent(
                    url=url,
                    content_type=content_type,
                    documents=documents,
                    links=unique_links
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing initial URL {url}: {str(e)}")
            raise

    def _on_background_task_complete(self, task):
        """Callback for when background task completes"""
        try:
            # Check if task raised any exceptions
            task.result()
            self.logger.info("Background processing completed successfully")
        except Exception as e:
            self.logger.error(f"Background task failed: {str(e)}\nTraceback:\n{traceback.format_exc()}")

    async def _process_background_links(self):
        """Process queued links in background"""
        try:
            self.logger.info("Starting background link processing")
            batch_size = 5
            
            while self.url_queue:
                batch_urls = set(list(self.url_queue)[:batch_size])
                self.url_queue -= batch_urls
                
                self.logger.info(f"Processing batch of {len(batch_urls)} URLs. {len(self.url_queue)} remaining")
                
                async with self.crawler as crawler:
                    for url in batch_urls:
                        if url not in self.visited_urls:
                            try:
                                self.logger.debug(f"Crawling URL: {url}")
                                crawl_result = await crawler.arun(
                                    url=url,
                                    clean_content=True,
                                    bypass_cache=True
                                )
                                
                                # Process using the helper method
                                self.logger.debug(f"Processing content for URL: {url}")
                                documents = await self._process_content(crawl_result)
                                
                                # Mark URL as visited regardless of document processing
                                self.visited_urls.add(url)
                                self.logger.debug(f"Processed URL {url}: Got {len(documents) if documents else 0} documents")
                                
                            except Exception as e:
                                self.logger.error(
                                    f"Error processing URL {url}: {str(e)}\n"
                                    f"Content type: {self.content_type}\n"
                                    f"Traceback:\n{traceback.format_exc()}"
                                )
                                continue
                
                # Small delay between batches
                await asyncio.sleep(1)
            
            self.logger.info("Background processing complete")
                
        except Exception as e:
            self.logger.error(f"Error in background processing: {str(e)}")
            raise

    async def get_indexing_status(self):
        """Get the current status of the indexing process"""
        try:
            # Check if initial URL has been processed
            initial_processed = len(self.visited_urls) > 0
            
            # Get background task status
            background_active = (
                self.background_task and 
                not self.background_task.done()
            )
            
            # Check if background task failed
            if self.background_task and self.background_task.done():
                try:
                    self.background_task.result()
                except Exception as e:
                    self.logger.error(f"Background task failed: {str(e)}")
                    return {
                        "status": "error",
                        "error": str(e),
                        "initial_url_processed": initial_processed,
                        "urls_processed": len(self.visited_urls),
                        "urls_queued": len(self.url_queue),
                        "is_complete": True,
                        "background_task_active": False
                    }

            self.logger.info(
                f"Status check - Initial processed: {initial_processed}, "
                f"Total processed: {len(self.visited_urls)}, "
                f"Queued: {len(self.url_queue)}"
            )
            
            return {
                "status": "processing" if background_active else "complete",
                "initial_url_processed": initial_processed,
                "urls_processed": len(self.visited_urls),
                "urls_queued": len(self.url_queue),
                "is_complete": not background_active,
                "background_task_active": background_active,
                "visited_urls": list(self.visited_urls),
                "queued_urls": list(self.url_queue)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting status: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass

    async def _generate_summary(self, content: str) -> str:
        """Generate a summary of the content"""
        try:
            response = await self.llm.ainvoke(
                self.summary_prompt.format(content=content)
            )
            # Extract the content from AIMessage
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            self.logger.warning(f"Error generating summary: {str(e)}")
            return ""

    async def _process_content(self, crawl_result):
        """Process content with appropriate extractor"""
        try:
            # List to collect all documents
            all_documents = []
            
            async def batch_callback(documents):
                """Callback to process document batches"""
                # Add metadata to documents before storing
                for doc in documents:
                    metadata = {
                        'url': crawl_result.url,
                        'content_type': self.content_type.value,
                        'extraction_date': datetime.now().isoformat(),
                        'title': crawl_result.metadata.get('title', ''),
                        'description': crawl_result.metadata.get('description', ''),
                        'keywords': crawl_result.metadata.get('keywords', []),
                        'author': crawl_result.metadata.get('author', ''),
                        'last_modified': crawl_result.metadata.get('last_modified', ''),
                        'source': crawl_result.url
                    }
                    # Clean metadata before updating
                    doc.metadata.update(self._clean_metadata(metadata))
                
                # Store in vector store
                await self.vector_store.add_documents(documents)
                self.logger.info(f"Added batch of {len(documents)} documents to vector store")
                
                # Collect documents
                all_documents.extend(documents)

            # Process content with callback
            await self.content_processor.process(
                crawl_result, 
                self.content_type,
                batch_callback
            )
            
            self.logger.debug(f"Total documents collected: {len(all_documents)}")
            return all_documents

        except Exception as e:
            self.logger.error(f"Error in _process_content: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            return []
