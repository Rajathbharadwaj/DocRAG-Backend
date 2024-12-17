"""System prompts for the retrieval agent."""

ROUTER_SYSTEM_PROMPT = """Analyze the user's question and classify it into one of these categories:
- documents: Questions about documents that need research
- more-info: Questions that need clarification

Output Format:
{
    "type": "category",
    "logic": "Explanation of why this category was chosen"
}"""

MORE_INFO_SYSTEM_PROMPT = """The user's question needs clarification. {logic}
Ask for specific details that would help provide a better answer."""

GENERAL_SYSTEM_PROMPT = """This is a general question. {logic}
Provide a helpful response based on general knowledge."""

RESEARCH_PLAN_SYSTEM_PROMPT = """Create a step-by-step research plan to answer the user's question.
Each step should be a specific query or research task.
Focus on breaking down complex questions into smaller, searchable components."""

GENERATE_QUERIES_SYSTEM_PROMPT = """Generate 3-5 diverse search queries to help answer the given question.
Focus on different aspects of the question to get comprehensive results.
Make queries specific and targeted."""

RESPONSE_SYSTEM_PROMPT = """Use the following context to answer the user's question:
{context}

Provide a comprehensive answer that directly addresses the question. If code is required, provide it in a code block."""