from dataclasses import dataclass
from typing import List, Dict

@dataclass
class PageContent:
    url: str
    content: str
    links: List[str]
    backlinks: List[str]
    media_references: Dict
    metadata: Dict 