import os
import json
from enum import Enum
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "./chroma_db"


class ParserType(Enum):
    """Available PDF parsing backends."""
    LLAMA_PARSE = "llamaparse"
    DOCLING = "docling"  # Docling with MPS acceleration (recommended)


# Configure which parser to use (set via environment variable or default)
DEFAULT_PARSER = os.getenv("PDF_PARSER", ParserType.DOCLING.value)


def load_pdfs(data_dir: str = "./data", parser: str = DEFAULT_PARSER):
    """
    Load PDFs using the specified parser.

    Args:
        data_dir: Directory containing PDF files
        parser: Parser type ("docling" or "llamaparse")

    Returns:
        List of Document objects
    """
    parser_type = ParserType(parser)

    if parser_type == ParserType.DOCLING:
        print("Using Docling parser (MPS-accelerated, recommended)")
        from src.ingestion_docling import load_pdfs_docling
        return load_pdfs_docling(data_dir)
    elif parser_type == ParserType.LLAMA_PARSE:
        print("Using LlamaParse parser")
        return _load_pdfs_llamaparse(data_dir)
    else:
        raise ValueError(f"Unknown parser type: {parser}")


def _load_pdfs_llamaparse(data_dir: str = "./data"):
    """
    Load PDFs using LlamaParse (original implementation).
    Kept as fallback option.
    """
    import nest_asyncio
    from llama_cloud_services import LlamaParse

    # Apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()

    docs = []
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return []

    # Create a cache directory for parsed markdown
    cache_dir = os.path.join(data_dir, "parsed_cache")
    os.makedirs(cache_dir, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        return []

    files_to_parse = []
    files_to_load_from_cache = []

    for f in pdf_files:
        pdf_path = os.path.join(data_dir, f)
        cache_path = os.path.join(cache_dir, f"{f}.md")

        if os.path.exists(cache_path):
            files_to_load_from_cache.append((pdf_path, cache_path))
        else:
            files_to_parse.append(pdf_path)

    # Load cached files
    for pdf_path, cache_path in files_to_load_from_cache:
        with open(cache_path, "r", encoding="utf-8") as f:
            text = f.read()
            metadata = {"source": pdf_path}
            docs.append(Document(page_content=text, metadata=metadata))

    # Parse new files
    if files_to_parse:
        print(f"Parsing {len(files_to_parse)} new files with LlamaParse...")

        # Initialize LlamaParse
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            parse_mode="parse_page_with_agent",
            model="gpt-4o-mini",
            high_res_ocr=True,
            adaptive_long_table=True,
            outlined_table_extraction=True,
            output_tables_as_HTML=True,
            verbose=True,
            language="en",
            num_workers=4,
        )

        try:
            results = parser.parse(files_to_parse)
        except Exception as e:
            print(f"Error during parsing: {e}")
            return docs

        for i, result in enumerate(results):
            pdf_path = files_to_parse[i]
            cache_path = os.path.join(cache_dir, f"{os.path.basename(pdf_path)}.md")

            text_content = ""

            # Extract text from result
            if hasattr(result, "pages"):
                for page in result.pages:
                    page_text = getattr(page, "md", "")
                    if not page_text:
                        page_text = getattr(page, "text", "")
                    text_content += page_text + "\n\n"
            elif isinstance(result, list):
                for item in result:
                    text = getattr(item, "text", "")
                    if not text:
                        text = getattr(item, "page_content", "")
                    text_content += text + "\n\n"
            else:
                text_content = getattr(result, "text", "")
                if not text_content:
                    text_content = getattr(result, "page_content", "")
                if not text_content:
                    text_content = getattr(result, "md", "")

            # Save to cache
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            metadata = {"source": pdf_path}
            docs.append(Document(page_content=text_content, metadata=metadata))

    return docs


def split_docs(docs):
    """
    Split documents using markdown-aware chunking with LLM metadata extraction.

    Args:
        docs: List of Document objects

    Returns:
        List of split Document objects with metadata
    """
    # 1. Split by Markdown Headers first to preserve structure
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # Initialize LLM for metadata extraction
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    metadata_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at analyzing clinical documents. Extract the following metadata from the text: 'cancer_type' (e.g. Breast, Lung, General), 'year' (YYYY), and 'doc_type' (Guideline, Trial, Report). Return JSON only."),
        ("user", "Filename: {filename}\n\nText Preview: {text_preview}")
    ])

    md_header_splits = []

    # Cache metadata per file to avoid repeated API calls
    file_metadata_cache = {}

    for doc in docs:
        # Extract metadata if not already cached for this source
        source = doc.metadata.get("source", "")
        if source not in file_metadata_cache:
            try:
                # Use first 2000 chars for metadata extraction
                text_preview = doc.page_content[:2000]
                filename = os.path.basename(source)
                chain = metadata_prompt | llm
                response = chain.invoke({"filename": filename, "text_preview": text_preview})
                metadata = json.loads(response.content)
                file_metadata_cache[source] = metadata
                print(f"Extracted metadata for {filename}: {metadata}")
            except Exception as e:
                print(f"Error extracting metadata for {source}: {e}")
                file_metadata_cache[source] = {"cancer_type": "Unknown", "year": "Unknown", "doc_type": "Unknown"}

        # Split the content of each document
        splits = markdown_splitter.split_text(doc.page_content)
        # Preserve original metadata and merge with extracted metadata
        for split in splits:
            split.metadata.update(doc.metadata)
            split.metadata.update(file_metadata_cache[source])
            md_header_splits.append(split)

    # 2. Recursively split within header sections if still too large
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    return text_splitter.split_documents(md_header_splits)


def build_vectorstore(splits):
    """Build Chroma vectorstore from document splits."""
    embeddings = OpenAIEmbeddings(
        model="qwen/qwen3-embedding-8b",
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def ingest_docs(parser: str = DEFAULT_PARSER):
    """
    Main ingestion pipeline.

    Args:
        parser: Parser type to use ("docling" or "llamaparse")

    Returns:
        Chroma vectorstore
    """
    docs = load_pdfs("./data", parser=parser)

    if not docs:
        print("No valid documents found to ingest.")
        return None

    splits = split_docs(docs)

    if not splits:
        print("No splits created. Exiting.")
        return None

    return build_vectorstore(splits)


if __name__ == "__main__":
    # Use environment variable PDF_PARSER or default to docling
    parser_type = os.getenv("PDF_PARSER", ParserType.DOCLING.value)
    print(f"\nStarting ingestion with parser: {parser_type}")
    ingest_docs(parser=parser_type)
