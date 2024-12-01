from dataclasses import dataclass
from typing import List, Dict, Optional
from langchain_core.documents import Document

@dataclass
class PageContent:
    url: str
    content: str
    links: List[str]
    backlinks: List[str]
    media_references: Dict
    metadata: Dict
    document: Document 