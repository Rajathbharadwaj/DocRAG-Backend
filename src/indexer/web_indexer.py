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

class WebIndexer:
    def __init__(
        self, 
        doc_name: str, 
        max_depth: int = 2, 
        max_links: Optional[int] = None,
        backlink_threshold: float = 0.3,
        use_serverless: bool = True,
        region: str = "us-east-1"
    ):
        self.doc_name = doc_name
        self.max_depth = max_depth
        self.max_links = max_links
        self.backlink_threshold = backlink_threshold
        self.content_type: Optional[ContentType] = None
        
        # Initialize VectorStore with doc_name
        self.vector_store = VectorStoreEngine(
            doc_name=doc_name,
            use_serverless=use_serverless,
            region=region
        )
        
        self.visited_urls: Set[str] = set()
        self.url_queue: Set[str] = set()
        self.backlinks_map: Dict[str, Set[str]] = {}
        self.background_task: Optional[asyncio.Task] = None
        self.content_processor = ContentProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Summary prompt template
        self.summary_prompt = PromptTemplate(
            template="""Given the following document content, provide a concise summary in 2-3 sentences
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

    async def process_initial_url(self, 
                                url: str,
                                content_type: ContentType,
                                max_depth: Optional[int] = None,
                                backlink_threshold: Optional[float] = None) -> Optional[PageContent]:
        """Process the main URL and start background processing"""
        try:
            if max_depth is not None:
                self.max_depth = max_depth
            if backlink_threshold is not None:
                self.backlink_threshold = backlink_threshold
            
            # Crawl and process URL
            async with self.crawler as crawler:
                crawl_result = await crawler.arun(
                    url=url,
                    clean_content=True,
                    bypass_cache=True
                )
            self.content_type = content_type
            documents = await self.content_processor.process(crawl_result, self.content_type)
            
            if documents:
                # Process links from crawl_result
                processed_links = set()
                total_links = 0
                
                # For logging clarity
                max_links_display = f"{self.max_links}" if self.max_links is not None else "unlimited"
                
                if hasattr(crawl_result, 'links'):
                    if isinstance(crawl_result.links, dict):
                        # Collect all links first
                        all_links = []
                        if 'internal' in crawl_result.links:
                            all_links.extend([
                                link['href'] for link in crawl_result.links['internal']
                                if isinstance(link, dict) and 'href' in link
                            ])
                        if 'external' in crawl_result.links:
                            all_links.extend([
                                link['href'] for link in crawl_result.links['external']
                                if isinstance(link, dict) and 'href' in link
                            ])
                        
                        # Count total before limiting
                        total_links = len(all_links)
                        
                        # Apply limit to combined set
                        if self.max_links:
                            all_links = all_links[:self.max_links]
                        
                        processed_links.update(all_links)
                        
                    elif isinstance(crawl_result.links, (list, set)):
                        total_links = len(crawl_result.links)
                        links_list = list(crawl_result.links)
                        if self.max_links:
                            links_list = links_list[:self.max_links]
                        processed_links.update(links_list)
                
                self.logger.info(f"Found {total_links} total links")
                self.logger.info(f"Processing {len(processed_links)} links (limit: {max_links_display})")
                
                # Update tracking
                self.visited_urls.add(url)
                self.url_queue = processed_links
                self.logger.info(f"Added {len(self.url_queue)} links to queue (limit: {max_links_display})")
                
                # Generate summary for the entire content
                full_content = "\n\n".join(doc.page_content for doc in documents)
                summary = await self._generate_summary(full_content)
                
                # Add enhanced metadata to documents
                for doc in documents:
                    # Clean metadata values - replace None with empty strings
                    metadata = {
                        'url': url,
                        'content_type': content_type.value,
                        'extraction_date': datetime.now().isoformat(),
                        'document_summary': summary or "",
                        'title': crawl_result.metadata.get('title', ''),
                        'description': crawl_result.metadata.get('description', ''),
                        'keywords': crawl_result.metadata.get('keywords', []),
                        'author': crawl_result.metadata.get('author', ''),
                        'last_modified': crawl_result.metadata.get('last_modified', ''),
                        'source': url
                    }
                    
                    # Clean any remaining None values
                    cleaned_metadata = {
                        k: (v if v is not None else "") 
                        for k, v in metadata.items()
                    }
                    
                    doc.metadata.update(cleaned_metadata)
                
                # Use the vector store
                if self.vector_store:
                    await self.vector_store.add_documents(documents)
                
                # Start background processing
                self.background_task = asyncio.create_task(self._process_backlinks_async())
                
                return PageContent(
                    url=url,
                    content_type=content_type,
                    documents=documents,
                    links=processed_links,
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing URL {url}: {str(e)}")
            raise

    async def _process_backlinks_async(self):
        """Process backlinks in background"""
        try:
            self.logger.info("Starting background backlinks processing")
            current_depth = 1
            batch_size = 5
            
            while current_depth <= self.max_depth:
                self.logger.info(f"Processing depth {current_depth}")
                
                urls_to_process = list(self.url_queue)
                self.url_queue.clear()
                
                if not urls_to_process:
                    self.logger.info("No more URLs to process")
                    break
                
                # Calculate backlink threshold
                total_pages = len(self.visited_urls)
                if total_pages < 3:  # If we have very few pages, use simple threshold
                    min_backlinks = 1
                    self.logger.info(
                        f"Not enough pages ({total_pages}) for meaningful threshold, "
                        f"requiring minimum 1 backlink"
                    )
                else:
                    # Use float comparison instead of rounding
                    required_backlinks = total_pages * self.backlink_threshold
                    self.logger.info(
                        f"Requiring {required_backlinks:.1f} backlinks "
                        f"(total pages: {total_pages}, threshold: {self.backlink_threshold})"
                    )
                
                # Filter URLs based on backlink threshold
                filtered_urls = []
                for url in urls_to_process:
                    backlink_count = len(self.backlinks_map.get(url, set()))
                    if total_pages < 3:
                        # Simple check for minimal pages
                        if backlink_count >= min_backlinks:
                            filtered_urls.append(url)
                    else:
                        # Float comparison for meaningful thresholds
                        if backlink_count >= required_backlinks:
                            filtered_urls.append(url)
                            self.logger.debug(
                                f"Accepting {url} - sufficient backlinks "
                                f"({backlink_count} >= {required_backlinks:.1f})"
                            )
                        else:
                            self.logger.debug(
                                f"Skipping {url} - insufficient backlinks "
                                f"({backlink_count} < {required_backlinks:.1f})"
                            )
                
                # Apply max_links limit to filtered URLs
                if self.max_links:
                    original_count = len(filtered_urls)
                    filtered_urls = filtered_urls[:self.max_links]
                    self.logger.info(
                        f"Applied max_links limit: {len(filtered_urls)} URLs "
                        f"(reduced from {original_count})"
                    )
                
                self.logger.info(
                    f"Processing {len(filtered_urls)} URLs after filtering "
                    f"(from initial {len(urls_to_process)})"
                )
                
                # Process in batches
                for i in range(0, len(filtered_urls), batch_size):
                    batch = filtered_urls[i:i + batch_size]
                    self.logger.info(f"Processing batch of {len(batch)} URLs")
                    
                    # Process URLs in parallel
                    crawl_results = []
                    process_tasks = []
                    
                    # Use a single crawler context for the batch
                    async with self.crawler as crawler:
                        for url in batch:
                            if url not in self.visited_urls:
                                try:
                                    crawl_result = await crawler.arun(
                                        url=url,
                                        clean_content=True,
                                        bypass_cache=True
                                    )
                                    
                                    # Update backlinks for any new links found
                                    if hasattr(crawl_result, 'links'):
                                        new_links = []
                                        if isinstance(crawl_result.links, dict):
                                            for link_type in ['internal', 'external']:
                                                if link_type in crawl_result.links:
                                                    new_links.extend([
                                                        link['href'] for link in crawl_result.links[link_type]
                                                        if isinstance(link, dict) and 'href' in link
                                                    ])
                                        else:
                                            new_links = list(crawl_result.links)
                                        
                                        # Update backlinks map
                                        for link in new_links:
                                            if link not in self.backlinks_map:
                                                self.backlinks_map[link] = set()
                                            self.backlinks_map[link].add(url)
                                        
                                    crawl_results.append((url, crawl_result))
                                except Exception as e:
                                    self.logger.error(f"Error crawling {url}: {str(e)}")
                                    continue
                    
                    # Process documents and update vector store
                    if crawl_results:
                        all_documents = []
                        for url, result in crawl_results:
                            if result:
                                docs = await self.content_processor.process(result, self.content_type)
                                if docs:
                                    for doc in docs:
                                        metadata = {
                                            'url': url,
                                            'content_type': self.content_type.value,
                                            'extraction_date': datetime.now().isoformat(),
                                            'backlinks': len(self.backlinks_map.get(url, set())),
                                            'title': result.metadata.get('title', ''),
                                            'description': result.metadata.get('description', ''),
                                            'keywords': result.metadata.get('keywords', []),
                                            'author': result.metadata.get('author', ''),
                                            'last_modified': result.metadata.get('last_modified', ''),
                                            'source': url
                                        }
                                        doc.metadata.update(metadata)
                                    all_documents.extend(docs)
                                    self.visited_urls.add(url)
                        
                        if self.vector_store and all_documents:
                            await self.vector_store.add_documents(all_documents)
                    
                    current_depth += 1
                    self.logger.info(
                        f"Completed depth {current_depth-1}, "
                        f"processed {len(self.visited_urls)} URLs total"
                    )
                
        except Exception as e:
            self.logger.error(f"Error in background processing: {str(e)}")
            raise

    async def _process_single_url(self, url: str):
        """Process a single URL"""
        try:
            crawl_result = await self.crawler.crawl(url)
            documents = await self.content_processor.process(crawl_result)
            
            if documents and self.vector_store:
                await self.vector_store.add_documents(documents)
                self.visited_urls.add(url)
                self.url_queue.update(crawl_result.links)
                
                # Update backlinks
                for link in crawl_result.links:
                    if link not in self.backlinks_map:
                        self.backlinks_map[link] = set()
                    self.backlinks_map[link].add(url)
            
        except Exception as e:
            self.logger.error(f"Error processing URL {url}: {str(e)}")

    async def get_indexing_status(self):
        """Get the current status of the indexing process"""
        try:
            # Get background task status
            is_processing = (
                (self.background_task and not self.background_task.done()) or 
                len(self.visited_urls) == 0  # Only check if we've started processing
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
                        "urls_processed": len(self.visited_urls),
                        "urls_queued": len(self.url_queue),
                        "is_complete": True,
                        "background_task_active": False
                    }

            self.logger.info(f"Status check - Processed: {len(self.visited_urls)}, Queued: {len(self.url_queue)}")
            
            return {
                "status": "processing" if is_processing else "complete",
                "urls_processed": len(self.visited_urls),
                "urls_queued": len(self.url_queue),
                "is_complete": not is_processing,
                "background_task_active": bool(self.background_task and not self.background_task.done()),
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
