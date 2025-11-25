"""
Standard Docling ingestion using LangChain integration.
More stable than MLX version, works on any platform.
"""

import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document

# Configuration
CACHE_DIR_NAME = "parsed_cache_docling"


def load_pdfs_docling(data_dir: str = "./data", verbose: bool = True) -> List[Document]:
    """
    Load PDFs using standard Docling with LangChain integration.

    Args:
        data_dir: Directory containing PDF files
        verbose: Print progress information

    Returns:
        List of LangChain Document objects
    """
    try:
        from langchain_docling import DoclingLoader
        from docling.chunking import HybridChunker
    except ImportError:
        print("Error: langchain-docling not installed.")
        print("Install with: pip install langchain-docling")
        return []

    docs = []

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return []

    # Create cache directory
    cache_dir = os.path.join(data_dir, CACHE_DIR_NAME)
    os.makedirs(cache_dir, exist_ok=True)

    # Find all PDFs
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found.")
        return []

    # Separate cached vs new files
    files_to_parse = []
    files_to_load_from_cache = []

    for f in pdf_files:
        pdf_path = os.path.join(data_dir, f)
        cache_path = os.path.join(cache_dir, f"{f}.md")

        if os.path.exists(cache_path):
            files_to_load_from_cache.append((pdf_path, cache_path))
        else:
            files_to_parse.append(pdf_path)

    # Load from cache
    for pdf_path, cache_path in files_to_load_from_cache:
        if verbose:
            print(f"Loading cached: {os.path.basename(pdf_path)}")
        with open(cache_path, "r", encoding="utf-8") as f:
            text = f.read()
            metadata = {"source": pdf_path}
            docs.append(Document(page_content=text, metadata=metadata))

    # Parse new files with Docling
    if files_to_parse:
        print(f"\nParsing {len(files_to_parse)} new files with Docling...")

        # Note: We're using MARKDOWN export instead of DOC_CHUNKS
        # to maintain compatibility with existing chunking strategy
        from docling_core.transforms.chunker import DocMeta
        from langchain_docling.loader import ExportType

        for pdf_path in files_to_parse:
            try:
                if verbose:
                    print(f"Processing: {os.path.basename(pdf_path)}")

                # Load with DoclingLoader
                loader = DoclingLoader(
                    file_path=pdf_path,
                    export_type=ExportType.MARKDOWN,  # Use MARKDOWN for compatibility
                )

                # Load documents
                new_docs = loader.load()

                if not new_docs:
                    print(f"Warning: No content extracted from {pdf_path}")
                    continue

                # Combine all pages into single document (like LlamaParse)
                combined_text = "\n\n".join([doc.page_content for doc in new_docs])

                # Cache the result
                cache_path = os.path.join(cache_dir, f"{os.path.basename(pdf_path)}.md")
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(combined_text)

                if verbose:
                    print(f"✓ Cached to: {cache_path}")

                # Add to documents
                metadata = {"source": pdf_path}
                docs.append(Document(page_content=combined_text, metadata=metadata))

            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return docs


if __name__ == "__main__":
    # Test on ./data directory
    docs = load_pdfs_docling("./data", verbose=True)
    print(f"\nTotal documents loaded: {len(docs)}")

    if docs:
        print(f"\nFirst document preview ({len(docs[0].page_content)} chars):")
        print(docs[0].page_content[:500])
