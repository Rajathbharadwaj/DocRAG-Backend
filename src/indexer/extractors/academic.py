from typing import List
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from crawl4ai import CrawlResult
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from .base import BaseExtractor
import logging

logger = logging.getLogger(__name__)

class AcademicExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.extraction_strategy = LLMExtractionStrategy(
            provider="openai/gpt-4",
            instruction="""
            Extract academic content with emphasis on:
            1. Abstract and key findings
            2. Methodology and research design
            3. Results and conclusions
            4. Citations and references
            5. Tables and figures with captions
            6. Format all the content in a way that is easy to read and understand. Add blocks to identify the content. For example, if there is an image, add a block that says "Image: [image description]"
            NOTE: The content you're reading is a html page, so make sure to take out all the unncessary page elements.
            """
        )

    async def extract(self, result: CrawlResult) -> List[Document]:
        documents = []
        try:
            # Use LLM extraction with academic focus
            extracted = await self.crawler.arun(
                url=result.url,
                extraction_strategy=self.extraction_strategy,
                chunking_strategy=self.chunking_strategy,
                word_count_threshold=30  # Academic content tends to be denser
            )
            
            # Process main sections
            sections = ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion']
            for section in sections:
                if section_content := extracted.get(section):
                    doc = Document(
                        page_content=section_content,
                        metadata={
                            'type': 'academic_section',
                            'section': section,
                            'priority': 'high'
                        }
                    )
                    documents.append(doc)
            
            # Process citations and references
            for citation in extracted.get('citations', []):
                doc = Document(
                    page_content=citation.get('text', ''),
                    metadata={
                        'type': 'citation',
                        'reference': citation.get('reference'),
                        'doi': citation.get('doi'),
                        'priority': 'medium'
                    }
                )
                documents.append(doc)
            
            # Process figures and tables
            for figure in extracted.get('figures', []):
                doc = Document(
                    page_content=f"""
                    Type: Figure
                    Caption: {figure.get('caption', '')}
                    Description: {figure.get('description', '')}
                    """,
                    metadata={
                        'type': 'figure',
                        'figure_type': figure.get('type'),
                        'priority': 'high'
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error in academic extraction: {str(e)}")
            return [] 