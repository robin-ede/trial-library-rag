import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

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
