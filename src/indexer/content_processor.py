import tiktoken
from typing import List, Optional, Dict
import json
from models.page_content import PageContent
from langchain_core.documents import Document
from urllib.parse import urljoin, urlparse
from crawl4ai.chunking_strategy import FixedLengthWordChunking

class ContentProcessor:
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk = encoding.decode(tokens[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks

    async def extract_page_content(self, url: str, crawler, backlinks_map: Dict) -> Optional[PageContent]:
        try:
            print(f"[DEBUG] Starting content extraction for {url}")
            
            # Get base URL and path from input URL
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            base_path = parsed_url.path.split('/')[1] if len(parsed_url.path.split('/')) > 1 else ''
            print(f"[DEBUG] Base URL: {base_url}, Base path: {base_path}")
            
            # Create chunking strategy
            chunking_strategy = FixedLengthWordChunking(chunk_size=1000)
            
            result = await crawler.arun(
                url=url,
                extract_images=True,
                extract_links=True,
                clean_content=True,
                js_code="window.scrollTo(0, document.body.scrollHeight);",  # Scroll for lazy content
                chunking_strategy=chunking_strategy
            )
            print(f"[DEBUG] Crawler result: {result is not None}")
            print(f"[DEBUG] Result attributes: {dir(result)}")  # Debug available attributes
            
            if not result or not result.markdown:
                print(f"[DEBUG] No content returned from crawler for {url}")
                return None
            # Debug the raw links from crawler
            print(f"[DEBUG] Raw links from crawler: {result.links}")
            
            # Extract valid URLs from the nested structure
            links = []
            if hasattr(result, 'links'):
                # Handle internal links
                if 'internal' in result.links:
                    for link_obj in result.links['internal']:
                        if isinstance(link_obj, dict) and 'href' in link_obj:
                            href = link_obj['href']
                            parsed_href = urlparse(href)
                            
                            # Only follow links that:
                            # 1. Are from the same domain
                            # 2. Start with the same base path (e.g., '/docs/')
                            # 3. Don't contain fragments (#)
                            if (href.startswith('http') and parsed_href.netloc == parsed_url.netloc and 
                                parsed_href.path.startswith(f'/{base_path}') and 
                                not parsed_href.fragment):
                                links.append(href)
                            elif (href.startswith(f'/{base_path}') and not '#' in href):
                                links.append(urljoin(base_url, href))
                
                # Handle external links
                # if 'external' in result.links:
                #     for link_obj in result.links['external']:
                #         if isinstance(link_obj, dict) and 'href' in link_obj:
                #             href = link_obj['href']
                #             if href.startswith('http'):
                #                 links.append(href)
            
            print(f"[DEBUG] Extracted {len(links)} filtered URLs: {links[:5]}")
            
            document = Document(
                page_content=result.markdown,
                metadata={
                    'url': url,
                    'title': getattr(result, 'title', 'No title'),
                    'word_count': len(result.markdown.split()),
                }
            )
            print(f"[DEBUG] Document created with {len(document.page_content)} chars")
            
            # Create PageContent object
            page_content = PageContent(
                url=url,
                content=result.markdown,
                links=links,
                backlinks=backlinks_map.get(url, []),
                media_references=getattr(result, 'media', {}),
                metadata=document.metadata,
                document=document
            )
            print(f"[DEBUG] PageContent object created successfully")
            
            return page_content
            
        except Exception as e:
            print(f"[DEBUG] Error in content extraction: {str(e)}")
            print(f"[DEBUG] Exception type: {type(e)}")
            return None

    async def store_in_vector_db(self, page_content: PageContent, rag_engine):
        """Store content using RAG engine instead of directly using collection"""
        return await rag_engine.add_documents([page_content.document])
        