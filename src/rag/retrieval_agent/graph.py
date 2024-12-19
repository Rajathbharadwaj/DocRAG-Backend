"""Main entrypoint for the conversational retrieval graph."""

from typing import Any, Literal, TypedDict, cast
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from rag.retrieval_agent.configuration import AgentConfiguration
from rag.research_agent.graph import graph as researcher_graph
from rag.retrieval_agent.state import AgentState, InputState, Router
from rag.utils import format_docs, load_chat_model

import logging
logger = logging.getLogger(__name__)

async def analyze_and_route_query(
    state: AgentState, 
    *, 
    config: RunnableConfig
) -> dict[str, Router]:
    """Analyze and route the user's query."""
    # Skip router for testing
    if state.router and state.router["logic"]:
        return {"router": state.router}

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    messages = [
        {"role": "system", "content": configuration.router_system_prompt}
    ] + state.messages
    
    response = cast(Router, await model.with_structured_output(Router).ainvoke(messages))
    return {"router": response}

def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info"]:
    """Route query based on classification."""
    _type = state.router["type"]
    if _type == "documents":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    else:
        raise ValueError(f"Unknown router type {_type}")

async def ask_for_more_info(
    state: AgentState, 
    *, 
    config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Ask user for more information."""
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    
    system_prompt = configuration.more_info_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def respond_to_general_query(
    state: AgentState, 
    *, 
    config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Respond to general queries."""
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    
    system_prompt = configuration.general_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def create_research_plan(
    state: AgentState, 
    *, 
    config: RunnableConfig
) -> dict[str, list[str]]:
    """Create research plan."""
    class Plan(TypedDict):
        steps: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Plan)
    
    messages = [
        {"role": "system", "content": configuration.research_plan_system_prompt}
    ] + state.messages
    
    response = cast(Plan, await model.ainvoke(messages))
    return {
        "steps": response["steps"],
        "documents": "delete",
        "query": state.query,
    }

async def conduct_research(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute research plan steps."""
    try:
        result = await researcher_graph.ainvoke({"question": state.steps[0]}, config=config)
        return {
            "documents": result["documents"], 
            "steps": state.steps[1:]
        }
    except Exception as e:
        logger.error(f"Error conducting research: {str(e)}")
        return {
            "documents": [],
            "steps": state.steps[1:]
        }

def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Check if research is complete."""
    if len(state.steps or []) > 0:
        return "conduct_research"
    return "respond"

async def respond(
    state: AgentState, 
    *, 
    config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate final response."""
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    
    # TODO: add re-ranker
    top_k = 20
    context = format_docs(state.documents[:top_k])
    prompt = configuration.response_system_prompt.format(context=context)
    
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages)
    
    return {
        "messages": [response],
        "answer": response.content
    }

# Define graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)

# Add nodes
# builder.add_node("analyze_and_route_query", analyze_and_route_query)
builder.add_node("create_research_plan", create_research_plan)
builder.add_node("ask_for_more_info", ask_for_more_info)
builder.add_node("conduct_research", conduct_research)
builder.add_node("respond", respond)

# Add edges
builder.add_edge(START, "create_research_plan")
# builder.add_conditional_edges(
#     "create_research_plan",
#     route_query,
#     {
#         "create_research_plan": "create_research_plan",
#         "ask_for_more_info": "ask_for_more_info",
#     }
# )
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("respond", END)
builder.add_edge("ask_for_more_info", END)

# Compile graph
graph = builder.compile()
graph.name = "RetrievalGraph" 

