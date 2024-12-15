import os
import logging
import sys

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import StorageContext, load_index_from_storage
from deepeval.integrations.llama_index import (
    DeepEvalAnswerRelevancyEvaluator,
    DeepEvalFaithfulnessEvaluator,
    DeepEvalContextualRelevancyEvaluator
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Configuration for LLM and Embeddings ---
llm = Ollama(model="llama3.3:latest", request_timeout=120.0)
embed_model = HuggingFaceEmbedding(model_name="nvidia/NV-Embed-v2", trust_remote_code=True)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 128

# --- Load Documents ---
print("\nLoading documents...")
documents = SimpleDirectoryReader("./llamamodel/").load_data()
print(f"Loaded {len(documents)} documents")

# --- Function to Build or Load Automerging Index ---
def build_automerging_index(
    documents,
    llm,
    embed_model,
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = Settings
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        print("Creating a new index...")
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        print("Loading existing index from storage...")
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index

# Build or Load Automerging Index
automerge_index = build_automerging_index(
    documents=documents,
    llm=llm,
    embed_model=embed_model,
    save_dir="merging_index",
)

# --- Function to Create Automerging Query Engine ---
def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine

# Create Automerging Query Engine
automerge_query_engine = get_automerging_query_engine(automerge_index)

# --- Query and Evaluate ---
test_prompt = "How is the raw data processed so it's ready to pretrain the model for CLM?"

print("\nQuerying automerging engine...")
auto_merging_response = automerge_query_engine.query(test_prompt)

# Process Response and Context
actual_output = str(auto_merging_response) if auto_merging_response else ""
retrieval_context = [
    node.get_content() for node in auto_merging_response.source_nodes
] if auto_merging_response and auto_merging_response.source_nodes else []

print(f"\nAuto-Merging Response: {actual_output}")
print("=="*20)
print("Retrieval Context:", retrieval_context)

# Initialize Evaluators
evaluators = [
    DeepEvalAnswerRelevancyEvaluator(threshold=0.5, include_reason=True, model="ollama"),
    DeepEvalFaithfulnessEvaluator(threshold=0.5, include_reason=True, model="ollama"),
    DeepEvalContextualRelevancyEvaluator(threshold=0.5, include_reason=True, model="ollama"),
]

print("\nEvaluating automerging response...")
evaluation_results = []
for evaluator in evaluators:
    try:
        result = evaluator.evaluate_response(
            query=test_prompt,
            response=auto_merging_response
        )
        evaluation_results.append({
            "evaluator": evaluator.__class__.__name__,
            "query": test_prompt,
            "score": result.score,
            "passing": result.passing,
            "feedback": result.feedback
        })
    except KeyError as e:
        logger.error(f"KeyError during evaluation with {evaluator.__class__.__name__}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during evaluation with {evaluator.__class__.__name__}: {e}")

print("\nFinal Evaluation Results:")
for result in evaluation_results:
    print(f"Evaluator: {result['evaluator']}")
    print(f"Query: {result['query']}")
    print(f"Score: {result['score']}")
    print(f"Passing: {result['passing']}")
    print(f"Feedback: {result['feedback']}\n")

