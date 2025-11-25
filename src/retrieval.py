import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from dotenv import load_dotenv
from src.ingestion import load_pdfs, split_docs

load_dotenv()

CHROMA_DIR = "./chroma_db"

def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="qwen/qwen3-embedding-8b",
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore

def get_retriever(k: int = 5, filter: dict = None):
    vectorstore = get_vectorstore()
    search_kwargs = {"k": k}
    if filter:
        search_kwargs["filter"] = filter
    return vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

def get_bm25_retriever(docs, k: int = 5):
    return BM25Retriever.from_documents(docs, k=k)

def get_ensemble_retriever(k: int = 5, filter: dict = None):
    # Load and split docs for BM25
    # Note: In a production app, we might want to cache this or use a persistent store for BM25 too if possible,
    # but BM25Retriever is typically in-memory.
    raw_docs = load_pdfs("./data")
    # Filter empty pages as in ingestion
    raw_docs = [d for d in raw_docs if d.page_content and len(d.page_content.strip()) > 10]

    if not raw_docs:
        print("Warning: No documents found for BM25. Returning vector retriever only.")
        return get_retriever(k=k, filter=filter)

    splits = split_docs(raw_docs)

    if not splits:
        print("Warning: No splits created for BM25. Returning vector retriever only.")
        return get_retriever(k=k, filter=filter)

    try:
        bm25_retriever = get_bm25_retriever(splits, k=k)
    except Exception as e:
        print(f"Error initializing BM25Retriever: {e}. Returning vector retriever only.")
        return get_retriever(k=k, filter=filter)

    vector_retriever = get_retriever(k=k, filter=filter)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

def get_advanced_retriever(k: int = 5, filter: dict = None):
    """
    Advanced retriever using hybrid search (BM25 + Vector) with multi-query expansion.

    Note: Previously included reranking, but removed due to local model constraints.
    Future improvements: Add RRF fusion, contextual retrieval, self-RAG filtering.
    """
    # 1. Base Retriever (Ensemble)
    base_retriever = get_ensemble_retriever(k=k, filter=filter)

    # 2. Multi-Query Expansion
    # Use the LLM to generate variations of the query
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # MultiQueryRetriever generates variants, retrieves for each, and takes the union
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm
    )

    return mq_retriever
