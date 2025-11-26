import os
import time  # TEMPORARY DEBUG: for timing instrumentation
import logging  # TEMPORARY DEBUG: for timing instrumentation
from typing import List
from langchain_milvus import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from dotenv import load_dotenv
from src.tracked_embeddings import TrackedOpenAIEmbeddings

load_dotenv()

# TEMPORARY DEBUG: Configure logging for timing instrumentation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

MILVUS_URI = "./milvus_vectorstore.db"


def get_vectorstore():
    embeddings = TrackedOpenAIEmbeddings(
        model="qwen/qwen3-embedding-8b",
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": MILVUS_URI},
    )

    # TEMPORARY DEBUG: Monkey-patch similarity_search to add timing
    original_search = vectorstore.similarity_search

    def timed_search(*args, **kwargs):
        start = time.perf_counter()
        result = original_search(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"💾 Milvus query: {elapsed:.3f}s ({len(result)} docs)")
        return result

    vectorstore.similarity_search = timed_search

    return vectorstore

def get_retriever(k: int = 3, filter: dict = None):
    vectorstore = get_vectorstore()
    search_kwargs = {"k": k}
    if filter:
        search_kwargs["filter"] = filter
    return vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

def get_bm25_retriever(docs, k: int = 3):
    return BM25Retriever.from_documents(docs, k=k)


# TEMPORARY DEBUG: Timed wrapper for EnsembleRetriever
class TimedEnsembleRetriever(BaseRetriever):
    """Wrapper around EnsembleRetriever that times BM25 vs Vector retrieval separately."""

    bm25_retriever: BaseRetriever
    vector_retriever: BaseRetriever
    weights: List[float]

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Time each retriever separately and merge results."""
        logging.info("🔄 Ensemble: Starting retrieval...")

        # Time BM25 retrieval
        logging.info("🔄 Ensemble: BM25 retrieval...")
        bm25_start = time.perf_counter()
        bm25_docs = self.bm25_retriever.invoke(query)
        bm25_elapsed = time.perf_counter() - bm25_start
        logging.info(f"🔄 Ensemble: BM25 completed in {bm25_elapsed:.3f}s ({len(bm25_docs)} docs)")

        # Time vector retrieval (includes embedding + Milvus)
        logging.info("🔄 Ensemble: Vector retrieval...")
        vector_start = time.perf_counter()
        vector_docs = self.vector_retriever.invoke(query)
        vector_elapsed = time.perf_counter() - vector_start
        logging.info(f"🔄 Ensemble: Vector completed in {vector_elapsed:.3f}s ({len(vector_docs)} docs)")

        # Merge results (simplified - just combine and deduplicate)
        logging.info("🔄 Ensemble: Merging results...")
        merge_start = time.perf_counter()

        # Weight and merge documents
        doc_dict = {}

        # Add BM25 docs with weight
        for i, doc in enumerate(bm25_docs):
            doc_id = id(doc)
            doc_dict[doc_id] = (doc, self.weights[0] * (1 / (i + 1)))

        # Add vector docs with weight
        for i, doc in enumerate(vector_docs):
            doc_id = id(doc)
            if doc_id in doc_dict:
                # Already exists, add weight
                existing_doc, existing_score = doc_dict[doc_id]
                doc_dict[doc_id] = (existing_doc, existing_score + self.weights[1] * (1 / (i + 1)))
            else:
                doc_dict[doc_id] = (doc, self.weights[1] * (1 / (i + 1)))

        # Sort by score and return
        sorted_docs = sorted(doc_dict.values(), key=lambda x: x[1], reverse=True)
        result = [doc for doc, score in sorted_docs]

        merge_elapsed = time.perf_counter() - merge_start
        logging.info(f"🔄 Ensemble: Merging completed in {merge_elapsed:.3f}s")

        total_elapsed = bm25_elapsed + vector_elapsed + merge_elapsed
        logging.info(f"🔄 Ensemble: Total ensemble time: {total_elapsed:.3f}s")

        return result


def get_ensemble_retriever(k: int = 3, filter: dict = None):
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

    # TEMPORARY DEBUG: Use timed wrapper instead of regular EnsembleRetriever
    # Note: Using construct() to bypass Pydantic validation for custom retriever types
    ensemble_retriever = TimedEnsembleRetriever.construct(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI


# TEMPORARY DEBUG: Timed wrapper for MultiQueryRetriever
class TimedMultiQueryRetriever(BaseRetriever):
    """Wrapper around MultiQueryRetriever that times query generation and per-variation retrieval."""

    base_retriever: BaseRetriever
    llm: object  # ChatOpenAI instance

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Time query generation and each variation's retrieval."""
        logging.info("🔍 Multi-Query: Generating variations...")

        # We need to manually generate queries to time them separately
        # The MultiQueryRetriever uses a prompt to generate variations
        from langchain_core.prompts import PromptTemplate

        # This is the default prompt used by MultiQueryRetriever (generates 3 variations)
        prompt_str = """You are an AI language model assistant. Your task is
    to generate 3 different versions of the given user
    question to retrieve relevant documents from a vector  database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. Provide these alternative
    questions separated by newlines. Original question: {question}"""

        prompt = PromptTemplate.from_template(prompt_str)

        # Time query generation
        gen_start = time.perf_counter()
        prompt_value = prompt.format(question=query)
        response = self.llm.invoke(prompt_value)
        gen_elapsed = time.perf_counter() - gen_start

        # Parse query variations from response (default: don't include original)
        queries = [q.strip() for q in response.content.split('\n') if q.strip()]

        logging.info(f"🔍 Multi-Query: Generated {len(queries)} variations in {gen_elapsed:.3f}s")

        # Time each variation's retrieval
        all_docs = []
        total_retrieval_time = 0

        for i, var_query in enumerate(queries):
            logging.info(f"🔍 Multi-Query: Processing variation {i+1}/{len(queries)}: \"{var_query[:50]}...\"")

            var_start = time.perf_counter()
            docs = self.base_retriever.invoke(var_query)
            var_elapsed = time.perf_counter() - var_start
            total_retrieval_time += var_elapsed

            logging.info(f"🔍 Multi-Query: Variation {i+1}/{len(queries)} completed in {var_elapsed:.3f}s ({len(docs)} docs)")

            all_docs.extend(docs)

        # Deduplicate documents
        unique_docs = []
        seen_ids = set()
        for doc in all_docs:
            doc_id = id(doc)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

        total_time = gen_elapsed + total_retrieval_time
        logging.info(f"🔍 Multi-Query: All {len(queries)} variations completed in {total_time:.3f}s (gen: {gen_elapsed:.3f}s, retrieval: {total_retrieval_time:.3f}s)")
        logging.info(f"🔍 Multi-Query: Returning {len(unique_docs)} unique documents")

        return unique_docs


def get_advanced_retriever(k: int = 3, filter: dict = None):
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

    # TEMPORARY DEBUG: Use timed wrapper instead of regular MultiQueryRetriever
    # Note: Using construct() to bypass Pydantic validation for custom retriever types
    mq_retriever = TimedMultiQueryRetriever.construct(
        base_retriever=base_retriever, llm=llm
    )

    return mq_retriever
