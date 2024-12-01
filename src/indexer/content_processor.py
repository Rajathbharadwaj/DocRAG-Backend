import tiktoken
from typing import List, Optional
import json
from .models import PageContent
from langchain_core.documents import Document

class ContentProcessor:
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk = encoding.decode(tokens[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks

    async def extract_page_content(self, url: str, crawler, backlinks_map) -> Optional[PageContent]:
        try:
            result = await crawler.arun(
                url=url,
                extract_images=True,
                extract_links=True,
                clean_content=True
            )

            if not result:
                return None

            # Create LangChain Document
            document = Document(
                page_content=result.markdown,
                metadata={
                    'url': url,
                    'title': result.title,
                    'backlinks': backlinks_map.get(url, []),
                    'media_references': result.media
                }
            )

            return PageContent(
                url=url,
                content=result.markdown,
                links=result.links,
                backlinks=backlinks_map.get(url, []),
                media_references=result.media,
                metadata={
                    'title': result.title,
                    'word_count': len(result.markdown.split()),
                    'link_count': len(result.links)
                },
                document=document  # Add document to PageContent
            )

        except Exception as e:
            return None

    async def store_in_vector_db(self, page_content: PageContent, rag_engine):
        """Store content using RAG engine instead of directly using collection"""
        return await rag_engine.add_documents([page_content.document])
        