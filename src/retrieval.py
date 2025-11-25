import os
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from dotenv import load_dotenv

load_dotenv()

MILVUS_URI = "./milvus_vectorstore.db"

def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="qwen/qwen3-embedding-8b",
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": MILVUS_URI},
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
    # Fetch all documents from Milvus for BM25 (avoids re-processing PDFs)
    vectorstore = get_vectorstore()

    try:
        # Retrieve all documents from Milvus by querying with a dummy query and large k
        # Note: This is a workaround since Milvus doesn't have a native "get all docs" method
        # For large collections, consider implementing pagination or a document cache
        # Future: Upgrade to Milvus Standalone (Docker) for native sparse vector (BM25) support.
        # Milvus Lite does NOT support native BM25 yet.
        docs = vectorstore.similarity_search("", k=10000)  # Fetch up to 10k chunks

        # Filter empty chunks
        docs = [d for d in docs if d.page_content and len(d.page_content.strip()) > 10]

        if not docs:
            print("Warning: No documents found in Milvus for BM25. Returning vector retriever only.")
            return get_retriever(k=k, filter=filter)

        print(f"Loaded {len(docs)} chunks from Milvus for BM25")

    except Exception as e:
        print(f"Error fetching documents from Milvus for BM25: {e}. Returning vector retriever only.")
        return get_retriever(k=k, filter=filter)

    try:
        bm25_retriever = get_bm25_retriever(docs, k=k)
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
        temperature=0.5,  # Increased to 0.5 to encourage diverse query variations
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # MultiQueryRetriever generates variants, retrieves for each, and takes the union
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm
    )

    return mq_retriever
