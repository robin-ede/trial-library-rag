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
        model="openai/text-embedding-3-small",
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore

def get_retriever(k: int = 5):
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_type="similarity", k=k)

def get_bm25_retriever(docs, k: int = 5):
    return BM25Retriever.from_documents(docs, k=k)

def get_ensemble_retriever(k: int = 5):
    # Load and split docs for BM25
    # Note: In a production app, we might want to cache this or use a persistent store for BM25 too if possible,
    # but BM25Retriever is typically in-memory.
    raw_docs = load_pdfs("./data")
    # Filter empty pages as in ingestion
    raw_docs = [d for d in raw_docs if d.page_content and len(d.page_content.strip()) > 10]
    splits = split_docs(raw_docs)
    
    bm25_retriever = get_bm25_retriever(splits, k=k)
    vector_retriever = get_retriever(k=k)
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever
