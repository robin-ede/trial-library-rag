import os
import nest_asyncio
from llama_cloud_services import LlamaParse
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

CHROMA_DIR = "./chroma_db"

def load_pdfs(data_dir: str = "./data"):
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
        # Simple cache key: filename + .md (could use hash of content but filename is okay for now)
        cache_path = os.path.join(cache_dir, f"{f}.md")
        
        if os.path.exists(cache_path):
            files_to_load_from_cache.append((pdf_path, cache_path))
        else:
            files_to_parse.append(pdf_path)

    # Load cached files
    for pdf_path, cache_path in files_to_load_from_cache:
        # print(f"Loading cached parsed file for {pdf_path}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            text = f.read()
            metadata = {"source": pdf_path}
            docs.append(Document(page_content=text, metadata=metadata))

    # Parse new files
    if files_to_parse:
        print(f"Parsing {len(files_to_parse)} new files with LlamaParse...")
        
        # Initialize LlamaParse only if needed
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=True,
            language="en",
            num_workers=4,
        )

        try:
            results = parser.parse(files_to_parse)
        except Exception as e:
            print(f"Error during parsing: {e}")
            return docs # Return what we have so far

        # Check if result has 'pages' attribute (JobResult object)
        if hasattr(result, "pages"):
            for page in result.pages:
                # Prefer markdown content
                page_text = getattr(page, "md", "")
                if not page_text:
                    page_text = getattr(page, "text", "")
                text_content += page_text + "\n\n"
        # Fallback for other potential return types (e.g. list of documents)
        elif isinstance(result, list):
            for item in result:
                text = getattr(item, "text", "")
                if not text:
                    text = getattr(item, "page_content", "")
                text_content += text + "\n\n"
        else:
            # Try to get text directly
            text_content = getattr(result, "text", "")
            if not text_content:
                    text_content = getattr(result, "page_content", "")
            # If still empty, try 'md'
            if not text_content:
                    text_content = getattr(result, "md", "")
        
        # Save to cache
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        
        metadata = {"source": pdf_path}
        docs.append(Document(page_content=text_content, metadata=metadata))

    return docs

def split_docs(docs):
    # Markdown-aware splitting could be better, but RecursiveCharacterTextSplitter is robust
    # We might want to increase chunk size since markdown preserves structure better
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def build_vectorstore(splits):
    embeddings = OpenAIEmbeddings(
        # OpenRouter model id for OpenAI embeddings
        model="openai/text-embedding-3-small",
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore

def ingest_docs():
    docs = load_pdfs("./data")
    
    if not docs:
        print("No valid documents found to ingest.")
        return None

    splits = split_docs(docs)
    
    if not splits:
        print("No splits created. Exiting.")
        return None

    return build_vectorstore(splits)

if __name__ == "__main__":
    ingest_docs()
