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
        
        # Initialize LlamaParse with advanced settings from demo
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
            return docs # Return what we have so far

        # LlamaParse returns a list of JobResult objects (one per file)
        # We need to iterate through them and match them back to the files
        # Note: parser.parse(files_to_parse) returns results in the same order as input
        
        for i, result in enumerate(results):
            pdf_path = files_to_parse[i]
            cache_path = os.path.join(cache_dir, f"{os.path.basename(pdf_path)}.md")
            
            text_content = ""

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

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

def split_docs(docs):
    # 1. Split by Markdown Headers first to preserve structure
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Initialize LLM for metadata extraction
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    import json
    
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
    
    # Cache metadata per file to avoid repeated API calls for chunks of the same file
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
        # Preserve original metadata (e.g., source) and merge with header metadata AND extracted metadata
        for split in splits:
            split.metadata.update(doc.metadata)
            split.metadata.update(file_metadata_cache[source])
            md_header_splits.append(split)

    # 2. Recursively split within those header sections if they are still too large
    # We can use a slightly larger chunk size now that we have semantic boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    
    return text_splitter.split_documents(md_header_splits)

def build_vectorstore(splits):
    embeddings = OpenAIEmbeddings(
        # OpenRouter model id for OpenAI embeddings
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
