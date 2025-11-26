import os
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from langchain_milvus import Milvus
from dotenv import load_dotenv
from src.tracked_embeddings import TrackedOpenAIEmbeddings
from src.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)

# Configuration
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_TOKENS = 400  # Increased from 200 to reduce boundary-splitting issues
MILVUS_URI = "./milvus_vectorstore.db"


def load_pdfs(data_dir: str = "./data"):
    """
    Load and chunk PDFs using DoclingLoader with HybridChunker.

    Uses tokenization-aware chunking that respects document structure,
    token limits, and semantic boundaries for optimal retrieval performance.

    Args:
        data_dir: Directory containing PDF files

    Returns:
        List of pre-chunked Document objects with dl_meta
    """
    # Find all PDFs in the data directory
    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        return []

    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                 if f.lower().endswith(".pdf")]

    if not pdf_files:
        logger.warning("No PDF files found in data directory")
        return []

    logger.info(f"Processing {len(pdf_files)} PDF files with Docling")

    # Configure HybridChunker with tokenizer
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name=EMBED_MODEL_ID,
        max_tokens=DEFAULT_MAX_TOKENS
    )

    # Load with DoclingLoader - handles both parsing AND chunking
    loader = DoclingLoader(
        file_path=pdf_files,
        export_type=ExportType.DOC_CHUNKS,  # Returns pre-chunked documents
        chunker=HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True  # Merge undersized chunks with same metadata
        )
    )

    docs = loader.load()
    logger.info(f"Successfully loaded {len(docs)} chunks from {len(pdf_files)} PDF files")
    return docs




def build_vectorstore(splits):
    """Build Milvus vectorstore from document splits."""
    embeddings = TrackedOpenAIEmbeddings(
        model="qwen/qwen3-embedding-8b",
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = Milvus.from_documents(
        documents=splits,
        embedding=embeddings,
        connection_args={"uri": MILVUS_URI},
        drop_old=True,  # Drop old collection if exists
        auto_id=True
    )
    return vectorstore


def ingest_docs():
    """
    Main ingestion pipeline using DoclingLoader + HybridChunker.

    Flow: PDFs → DoclingLoader (parse + chunk) → Milvus

    Returns:
        Milvus vectorstore
    """
    # Load and chunk PDFs with DoclingLoader (already includes rich metadata in dl_meta)
    docs = load_pdfs("./data")

    if not docs:
        logger.warning("No valid documents found for ingestion")
        return None

    return build_vectorstore(docs)


if __name__ == "__main__":
    from src.logging_config import setup_logging
    setup_logging()
    logger.info("Starting document ingestion pipeline with Docling")
    ingest_docs()
