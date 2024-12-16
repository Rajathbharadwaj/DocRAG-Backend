"""Researcher graph using Pinecone retriever for document retrieval."""

from typing import cast, List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from rag.retriever import retrieve_documents as pinecone_retrieve
from rag.research_agent.state import QueryState, ResearcherState
from rag.configuration import AgentConfiguration
from rag.utils import load_chat_model

import logging
logger = logging.getLogger(__name__)

async def generate_queries(
    state: ResearcherState, 
    *, 
    config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate search queries based on the research question."""
    
    class Response(TypedDict):
        queries: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Response)
    
    messages = [
        {
            "role": "system", 
            "content": configuration.generate_queries_system_prompt
        },
        {
            "role": "human", 
            "content": state.question
        },
    ]
    
    response = cast(Response, await model.ainvoke(messages))
    return {"queries": response["queries"]}

async def retrieve_documents(
    state: QueryState, 
    *, 
    config: RunnableConfig
) -> dict[str, List[Document]]:
    """Retrieve documents using Pinecone."""
    try:
        # Get configuration
        configuration = AgentConfiguration.from_runnable_config(config)
        
        # Use our Pinecone retriever
        docs = await pinecone_retrieve(
            query=state.query,
            index_name=configuration.index_name,  # Add this to AgentConfiguration
            top_k=configuration.top_k  # Add this to AgentConfiguration
        )
        
        # Add query metadata
        for doc in docs:
            doc.metadata["query"] = state.query
            
        return {"documents": docs}
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return {"documents": []}

def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Create parallel retrieval tasks for each query."""
    return [
        Send("retrieve_documents", QueryState(query=query)) 
        for query in state.queries
    ]

# Define the graph
builder = StateGraph(ResearcherState)

# Add nodes
builder.add_node("generate_queries", generate_queries)
builder.add_node("retrieve_documents", retrieve_documents)

# Add edges
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,
    path_map=["retrieve_documents"],
)
builder.add_edge("retrieve_documents", END)

# Compile graph
graph = builder.compile()
graph.name = "ResearcherGraph"