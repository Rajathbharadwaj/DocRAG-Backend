"""State management for the retrieval agent."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, TypedDict, Literal
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class Router(TypedDict):
    """Router type and logic."""
    type: str
    logic: str

class InputState(TypedDict):
    """Input state for the retrieval agent."""
    messages: list[BaseMessage]

@dataclass
class AgentState:
    """State for the retrieval agent."""
    messages: list[BaseMessage]
    router: Optional[Router] = None
    steps: Optional[list[str]] = None
    documents: Optional[list[Document]] = None
    query: Optional[str] = None 