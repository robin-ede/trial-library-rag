import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "./chroma_db"

def load_pdfs(data_dir: str = "./data"):
    loaders = []
    docs = []
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return []
        
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(data_dir, filename)
            loaders.append(PyMuPDFLoader(path))

    for loader in loaders:
        docs.extend(loader.load())
    return docs

def split_docs(docs):
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
    # Chroma 0.4+ persists automatically, but explicit call doesn't hurt if using older versions
    # vectorstore.persist() 
    return vectorstore

def ingest_docs():
    docs = load_pdfs("./data")
    # Simple check for empty pages (scanned images, etc.)
    docs = [d for d in docs if d.page_content and len(d.page_content.strip()) > 10]
    
    if not docs:
        print("No valid documents found to ingest.")
        return None

    splits = split_docs(docs)
    return build_vectorstore(splits)

if __name__ == "__main__":
    ingest_docs()
