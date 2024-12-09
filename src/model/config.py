from pydantic import BaseModel, Field
from typing import List

class ProcessingConfig(BaseModel):
    """Configuration for content processing"""
    word_count_threshold: int = Field(default=10)
    excluded_tags: List[str] = Field(default_factory=lambda: ['nav', 'footer', 'aside'])
    remove_overlay_elements: bool = Field(default=True)
    wait_for_lazy_load: bool = Field(default=True)
    delay_time: float = Field(default=2.0)
    exclude_external_links: bool = Field(default=False)
    exclude_social_media: bool = Field(default=True)
    keep_data_attributes: bool = Field(default=False)
    process_iframes: bool = Field(default=True)
    min_media_score: int = Field(default=5) 