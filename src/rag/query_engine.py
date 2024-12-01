from typing import List, Dict
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class RAGQueryEngine:
    def __init__(self, 
                 llm,
                 embeddings,
                 vector_db_path: str = "vector_store"):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = None
        self.retriever = None
        self.chat_history = {}  # Store chat history by thread_id
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        self.setup_vectorstore(vector_db_path)
        self.setup_rag_chain()

    def setup_vectorstore(self, vector_db_path: str):
        """Initialize or load the vector store"""
        self.vector_store = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

    def setup_rag_chain(self):
        """Setup the RAG prompt and chain"""
        template = """Answer the question based only on the following context and chat history.
        If you cannot answer based on the context, say "I cannot answer this based on the provided information."

        Context: {context}
        
        Chat History: {chat_history}
        
        Question: {question}
        
        Answer:"""
        
        self.prompt = PromptTemplate.from_template(template)
        
        self.chain = (
            {
                "context": self.retriever, 
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.get_chat_history(x.get("thread_id", "default"))
            }
            | self.prompt
            | self.llm
        )

    def get_chat_history(self, thread_id: str) -> str:
        """Get formatted chat history for a thread"""
        history = self.chat_history.get(thread_id, [])
        if not history:
            return "No previous conversation."
        
        return "\n".join([
            f"Human: {h['question']}\nAssistant: {h['answer']}"
            for h in history
        ])

    def update_chat_history(self, thread_id: str, question: str, answer: str):
        """Update chat history for a thread"""
        if thread_id not in self.chat_history:
            self.chat_history[thread_id] = []
        
        self.chat_history[thread_id].append({
            "question": question,
            "answer": answer
        })

    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        # Split documents into chunks before adding to vector store
        splits = self.text_splitter.split_documents(documents)
        await self.vector_store.aadd_documents(splits)

    async def query(self, 
                   question: str,
                   thread_id: str = "default",
                   return_sources: bool = False) -> Dict:
        """Query the RAG system with chat history"""
        # Always retrieve relevant documents first
        context_docs = await self.retriever.aget_relevant_documents(question)
        
        # Get answer using the RAG chain
        answer = await self.chain.ainvoke({
            "question": question,
            "thread_id": thread_id
        })
        
        # Update chat history
        self.update_chat_history(thread_id, question, answer)
        
        if return_sources:
            return {
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in context_docs
                ]
            }
        else:
            # Just get the answer
            return {
                "answer": answer
            } 