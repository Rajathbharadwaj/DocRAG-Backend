from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set
from datetime import datetime
from langchain_core.documents import Document
from .types import ContentType

class CodeBlock(BaseModel):
    content: str = Field(..., description="The code snippet")
    language: str = Field(..., description="Programming language of the code")
    context: str = Field(..., description="Surrounding context or explanation")
    section: str = Field(..., description="Section where this code appears")

class APIEndpoint(BaseModel):
    method: str = Field(..., description="HTTP method (GET, POST, etc.)")
    endpoint: str = Field(..., description="API endpoint path")
    description: str = Field(..., description="Description of what the endpoint does")
    parameters: Optional[Dict] = Field(default=None)
    response: Optional[Dict] = Field(default=None)

class PageContent(BaseModel):
    """Represents processed page content"""
    url: str
    content_type: ContentType
    documents: List[Document]
    links: Set[str] = Field(default_factory=set)
    backlinks: Set[str] = Field(default_factory=set)
    media_references: Dict = Field(default_factory=dict)
    metadata: Dict = Field(default_factory=dict)
    extraction_date: datetime = Field(default_factory=lambda: datetime.now())

class ArticleSection(BaseModel):
    """Represents a section within an article"""
    title: str = Field(..., description="Section title or heading")
    content: str = Field(..., description="Section content")
    heading_level: int = Field(..., description="Heading level (h1=1, h2=2, etc.)")
    position: int = Field(..., description="Position in the article")

class ArticleMetadata(BaseModel):
    """Metadata specific to articles"""
    author: Optional[str] = Field(default=None, description="Article author")
    publish_date: Optional[datetime] = Field(default=None, description="Publication date")
    modified_date: Optional[datetime] = Field(default=None, description="Last modified date")
    category: Optional[str] = Field(default=None, description="Article category")
    tags: List[str] = Field(default_factory=list, description="Article tags")
    reading_time: Optional[int] = Field(default=None, description="Estimated reading time in minutes")
    summary: Optional[str] = Field(default=None, description="Article summary or excerpt")

class ArticleContent(PageContent):
    """Specialized content model for articles"""
    content_type: ContentType = Field(default=ContentType.ARTICLE)
    metadata: ArticleMetadata
    sections: List[ArticleSection] = Field(default_factory=list)
    main_image: Optional[Dict] = Field(default=None, description="Featured image details")
    social_shares: Optional[Dict] = Field(default=None, description="Social media share counts")
    comments_count: Optional[int] = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True  # For datetime fields 