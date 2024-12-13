from typing import List, Dict
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from crawl4ai import CrawlResult
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import logging
import re

# Import models using absolute imports
from model.content import CodeBlock, APIEndpoint
from model.types import ContentType

# These relative imports are fine since they're in the same package
from .base import BaseExtractor
from .code import CodeDocumentationExtractor

from .academic import AcademicExtractor
from .media import MediaRichExtractor

__all__ = [
    'CodeDocumentationExtractor',
    'AcademicExtractor',
    'MediaRichExtractor'
]
