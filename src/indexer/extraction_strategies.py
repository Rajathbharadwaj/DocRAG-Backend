from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from langchain_core.documents import Document
from crawl4ai import AsyncWebCrawler, CrawlResult
import logging
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class ExtractedContent:
    content: str
    metadata: Dict
    confidence: float  # How confident we are in the extraction quality

class BaseExtractor(ABC):
    def __init__(self):
        self.crawler = AsyncWebCrawler()
    
    @abstractmethod
    async def extract(self, url: str) -> List[Document]:
        pass
    
    async def get_base_content(self, url: str) -> Dict:
        """Get cleaned content using Crawl4AI"""
        result = await self.crawler.arun(
            url=url,
            word_count_threshold=10,
            remove_overlay_elements=True
        )
        return result

class CodeDocumentationExtractor(BaseExtractor):
    """Specialized for API docs, technical documentation"""
    
    async def extract(self, url: str) -> List[Document]:
        documents = []
        
        # Use Crawl4AI with code-specific settings
        result = await self.crawler.arun(
            url=url,
            excluded_tags=['nav', 'footer', 'aside'],
            keep_data_attributes=True,  # Important for code blocks
            process_iframes=True  # Some docs use iframes for examples
        )
        
        # Process code blocks with context
        for code_block in result.code_blocks:
            documents.append(Document(
                page_content=f"""
                Language: {code_block['language']}
                Context: {code_block['context']}
                Code:
                ```{code_block['language']}
                {code_block['content']}
                ```
                """,
                metadata={
                    'source': url,
                    'type': 'code_block',
                    'language': code_block['language'],
                    'line_count': len(code_block['content'].splitlines())
                }
            ))
        
        # Extract API documentation
        api_content = result.fit_markdown  # Gets main content
        if api_content:
            documents.append(Document(
                page_content=api_content,
                metadata={
                    'source': url,
                    'type': 'api_documentation',
                    'metadata': result.metadata
                }
            ))
        
        return documents

class MediaRichExtractor(BaseExtractor):
    """Specialized for content with images, videos, etc."""
    
    async def extract(self, url: str) -> List[Document]:
        documents = []
        
        # Configure crawler for media content
        result = await self.crawler.arun(
            url=url,
            wait_for="css:img[data-src]",  # Wait for lazy images
            delay_before_return_html=2.0
        )
        
        # Process images
        for image in result.media["images"]:
            if image['score'] > 5:  # Only include relevant images
                documents.append(Document(
                    page_content=f"""
                    Type: Image
                    Description: {image['desc']}
                    Alt Text: {image['alt']}
                    Context: {image['context']}
                    """,
                    metadata={
                        'source': url,
                        'type': 'image',
                        'src': image['src'],
                        'relevance_score': image['score']
                    }
                ))
        
        # Process videos
        for video in result.media["videos"]:
            documents.append(Document(
                page_content=f"""
                Type: Video
                Title: {video.get('title', '')}
                Description: {video.get('description', '')}
                Duration: {video.get('duration', '')}
                Transcript: {video.get('transcript', '')}
                """,
                metadata={
                    'source': url,
                    'type': 'video',
                    'src': video['src'],
                    'thumbnail': video.get('poster')
                }
            ))
        
        # Process audio
        for audio in result.media["audios"]:
            documents.append(Document(
                page_content=f"""
                Type: Audio
                Title: {audio.get('title', '')}
                Description: {audio.get('description', '')}
                Duration: {audio.get('duration', '')}
                Transcript: {audio.get('transcript', '')}
                """,
                metadata={
                    'source': url,
                    'type': 'audio',
                    'src': audio['src']
                }
            ))
        
        return documents

class ArticleExtractor(BaseExtractor):
    """Specialized for blog posts, articles"""
    
    async def extract(self, url: str) -> List[Document]:
        documents = []
        
        # Use fit_markdown for article content
        result = await self.crawler.arun(
            url=url,
            word_count_threshold=20,
            excluded_tags=['nav', 'footer', 'aside', 'form'],
            exclude_social_media_links=True
        )
        
        # Get main content using fit_markdown
        main_content = result.fit_markdown
        
        if main_content:
            documents.append(Document(
                page_content=main_content,
                metadata={
                    'source': url,
                    'type': 'article',
                    'title': result.metadata['title'],
                    'author': result.metadata['author'],
                    'published_date': result.metadata['published_date'],
                    'modified_date': result.metadata['modified_date']
                }
            ))
        
        # Process content links for references
        for link in result.links["internal"]:
            if link['type'] == 'content':
                documents.append(Document(
                    page_content=f"Reference: {link['text']}\nContext: {link['context']}",
                    metadata={
                        'source': url,
                        'type': 'reference',
                        'href': link['href']
                    }
                ))
        
        return documents

class AcademicExtractor(BaseExtractor):
    """Specialized for research papers, academic content"""
    
    async def extract(self, url: str) -> List[Document]:
        documents = []
        
        result = await self.crawler.arun(
            url=url,
            word_count_threshold=30,  # Academic content tends to be denser
            process_iframes=True,  # Some papers have embedded content
            exclude_social_media_links=True,
            keep_data_attributes=True
        )
        
        # Extract metadata
        metadata = {
            'source': url,
            'type': 'academic',
            'title': result.metadata['title'],
            'authors': result.metadata['author'],
            'published_date': result.metadata['published_date'],
            'citations': []
        }
        
        # Process citations
        for link in result.links["external"]:
            if link['type'] == 'citation':
                metadata['citations'].append({
                    'text': link['text'],
                    'href': link['href']
                })
        
        # Get main content
        content = result.fit_markdown
        
        if content:
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
            
        # Extract figures and tables
        for figure in result.media["images"]:
            if figure['score'] > 7:  # Higher threshold for academic figures
                documents.append(Document(
                    page_content=f"""
                    Type: Figure
                    Caption: {figure.get('desc', '')}
                    Context: {figure.get('context', '')}
                    """,
                    metadata={
                        'source': url,
                        'type': 'figure',
                        'src': figure['src']
                    }
                ))
        
        return documents 

class CodeExtractor(BaseExtractor):
    """Specialized for extracting code blocks from documentation"""
    
    def __init__(self):
        super().__init__()
        self.code_selectors = [
            'pre', 'code',
            '.highlight',           # Common in Sphinx
            '.sourceCode',          # Common in Pandoc
            '.code-block',          # Common in various platforms
            '[class*="language-"]', # Prism.js style
            '[class*="hljs"]'      # Highlight.js style
        ]
        
    async def extract(self, crawl_result: CrawlResult) -> List[Document]:
        if not crawl_result.success:
            print(f"Error: {crawl_result.error_message}")
            return []
            
        documents = []
        html_content = crawl_result.cleaned_html or crawl_result.html
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find code blocks using multiple selectors
        code_blocks = set()  # Use set to avoid duplicates
        for selector in self.code_selectors:
            blocks = soup.select(selector)
            code_blocks.update(blocks)
        
        for block in code_blocks:
            code_content = self._extract_code_content(block)
            if not code_content:
                continue
                
            language = self._detect_language(block)
            context = self._get_context(block)
            
            # Create document with enhanced metadata
            doc = Document(
                page_content=code_content,
                metadata=self._create_metadata(
                    crawl_result=crawl_result,
                    block=block,
                    language=language,
                    context=context
                )
            )
            documents.append(doc)
            
        return documents

    def _extract_code_content(self, element) -> str:
        """Extract and clean code content"""
        # Handle nested code blocks
        if element.name == 'pre' and element.find('code'):
            element = element.find('code')
            
        content = element.get_text().strip()
        
        # Remove common documentation artifacts
        content = content.replace('Copy code', '')
        content = content.replace('```', '')
        
        return content

    def _detect_language(self, element) -> Optional[str]:
        """Detect programming language from element classes"""
        if 'class' not in element.attrs:
            return None
            
        classes = element['class']
        
        # Common class patterns for language identification
        patterns = [
            r'language-(\w+)',  # Standard pattern
            r'hljs-(\w+)',      # Highlight.js
            r'(\w+)$'           # Fallback for simple classes
        ]
        
        for pattern in patterns:
            for cls in classes:
                match = re.search(pattern, cls)
                if match:
                    return match.group(1)
        
        return None

    def _get_context(self, element) -> str:
        """Extract relevant context around the code block"""
        context = []
        
        # Look for previous header
        prev_header = element.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if prev_header:
            context.append(prev_header.get_text().strip())
            
        # Look for previous paragraph
        prev_para = element.find_previous('p')
        if prev_para:
            context.append(prev_para.get_text().strip())
            
        return ' | '.join(context)
    
    def _find_section(self, element) -> str:
        """Find the section this code block belongs to"""
        headers = []
        current = element
        
        while current:
            if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                headers.append(current.get_text().strip())
            current = current.parent
            
        return ' > '.join(reversed(headers)) 