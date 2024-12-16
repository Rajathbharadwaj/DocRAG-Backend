"""Define the configurable parameters for the research agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

MODEL_NAME_TO_RESPONSE_MODEL = {
    "anthropic_claude_3_5_sonnet": "anthropic/claude-3-5-sonnet-20241022",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}

@dataclass(kw_only=True)
class AgentConfiguration:
    """Configuration for the research agent.
    
    This class defines parameters for query generation, document retrieval,
    and response generation in the research process.
    """
    
    # Model Configuration
    query_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "llm"}},
    ] = field(
        default="gpt-4o-mini",
        metadata={
            "description": "Model to use for query generation and analysis"
        },
    )
    
    response_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "llm"}},
    ] = field(
        default="gpt-4o-mini",
        metadata={
            "description": "Model to use for final response generation"
        },
    )

    # Retrieval Configuration
    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Embedding model for document retrieval"
        },
    )

    retriever_provider: Annotated[
        Literal["pinecone"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="pinecone",
        metadata={"description": "Vector store provider for retrieval"},
    )

    # Search Parameters
    top_k: int = field(
        default=4,
        metadata={
            "description": "Number of documents to retrieve per query"
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"k": 4},
        metadata={
            "description": "Additional search parameters for retrieval"
        },
    )

    index_name: str = field(
        default="rag-index",
        metadata={
            "description": "Name of the Pinecone index to use for retrieval"
        },
    )
    

    # Prompts
    generate_queries_system_prompt: str = field(
        default="""Generate 3-5 diverse search queries to help answer the given question.
Focus on different aspects of the question to get comprehensive results.
Make queries specific and targeted.""",
        metadata={
            "description": "System prompt for query generation"
        },
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], 
        config: Optional[RunnableConfig] = None
    ) -> T:
        """Create configuration from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        
        # Update model names if needed
        if "model_name" in configurable:
            configurable["response_model"] = MODEL_NAME_TO_RESPONSE_MODEL.get(
                configurable["model_name"], 
                configurable["model_name"]
            )
        
        # Get valid fields
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=AgentConfiguration)