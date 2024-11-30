import asyncio
from indexer.web_indexer import WebIndexer
from rag.query_engine import RAGQueryEngine

async def main():
    # Initialize and run indexer
    indexer = await WebIndexer(
        max_depth=2,
        backlink_threshold=0.3
    ).initialize_crawler()

    try:
        # Initialize RAG query engine
        rag_engine = RAGQueryEngine()

        # Example usage
        page_content = await indexer.process_initial_url("https://example.com")
        if page_content:
            print(f"Initial page processed: {page_content.metadata['title']}")
            
            # Example query
            results = await rag_engine.query(
                "What is the main topic of this website?"
            )
            
            print("\nQuery Results:")
            for result in results:
                print(f"\nContent: {result['content'][:200]}...")
                print(f"Source URL: {result['metadata']['url']}")

    finally:
        await indexer.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 