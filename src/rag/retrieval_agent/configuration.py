"""Define the configurable parameters for the agent."""

from __future__ import annotations
from dataclasses import dataclass, field
from rag.retrieval_agent import prompts
from rag.configuration import AgentConfiguration as BaseConfig

@dataclass(kw_only=True)
class AgentConfiguration(BaseConfig):
    """Configuration for the retrieval agent."""
    
    # Model Configuration
    query_model: str = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "Model for query processing and refinement"
        }
    )
    
    response_model: str = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "Model for response generation"
        }
    )

    # Prompt Configuration
    router_system_prompt: str = field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        metadata={
            "description": "System prompt for query routing"
        }
    )

    more_info_system_prompt: str = field(
        default=prompts.MORE_INFO_SYSTEM_PROMPT,
        metadata={
            "description": "System prompt for requesting clarification"
        }
    )


    research_plan_system_prompt: str = field(
        default=prompts.RESEARCH_PLAN_SYSTEM_PROMPT,
        metadata={
            "description": "System prompt for research planning"
        }
    )

    generate_queries_system_prompt: str = field(
        default=prompts.GENERATE_QUERIES_SYSTEM_PROMPT,
        metadata={
            "description": "System prompt for query generation"
        }
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={
            "description": "System prompt for response generation"
        }
    )