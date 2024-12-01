import asyncio
from indexer.web_indexer import WebIndexer
from rag.query_engine import RAGQueryEngine
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

async def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize LLM and embeddings
    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Initialize RAG engine
    rag_engine = RAGQueryEngine(
        llm=llm,
        embeddings=embeddings,
        vector_db_path="vector_store"
    )

    # Initialize and run indexer
    indexer = await WebIndexer(
        max_depth=2,
        backlink_threshold=0.3,
        rag_engine=rag_engine
    ).initialize_crawler()

    try:
        # Process initial URL
        url = "https://example.com"  # Replace with your URL
        page_content = await indexer.process_initial_url(url)
        
        if page_content:
            print(f"Initial page processed: {page_content.metadata['title']}")
            
            # Example conversation flow
            questions = [
                "What is the main topic of this website?",
                "Can you elaborate on that?",
                "What are the key points discussed?"
            ]
            
            thread_id = "demo_conversation"
            
            for question in questions:
                print(f"\nQuestion: {question}")
                
                response = await rag_engine.query(
                    question=question,
                    thread_id=thread_id,
                    return_sources=True
                )
                
                print(f"Answer: {response['answer']}")
                print("\nSources:")
                for source in response['sources']:
                    print(f"- {source['metadata']['url']}: {source['content'][:200]}...")

    finally:
        await indexer.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 