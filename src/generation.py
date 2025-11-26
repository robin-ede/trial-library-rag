import os
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a careful assistant for oncology clinical trial recruiting.

Use ONLY the provided context to answer the question.
If you cannot fully answer from the context, say you do not know and suggest asking a clinician.

For any factual statement, cite the source document in parentheses at the end
of the sentence, e.g. (NCCN_Breast_2024.pdf).
"""

def format_docs(docs: List[Document]) -> str:
    """Format documents with metadata including source file and page numbers from dl_meta."""
    formatted = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown_source")
        source = os.path.basename(source)

        # Extract page number from dl_meta if available
        page_info = ""
        dl_meta = d.metadata.get("dl_meta", {})
        if dl_meta and "doc_items" in dl_meta:
            doc_items = dl_meta["doc_items"]
            if doc_items and len(doc_items) > 0:
                prov = doc_items[0].get("prov", [])
                if prov and len(prov) > 0:
                    page_no = prov[0].get("page_no")
                    if page_no:
                        page_info = f" (page {page_no})"

        # Format header with page number
        header = f"**Source {i}:** `{source}{page_info}`"

        formatted.append(f"{header}\n\n{d.page_content}\n\n---")

    return "\n\n".join(formatted)

def get_rag_chain():
    """
    Get RAG chain that accepts context and question directly.

    This allows the app to retrieve docs once and reuse them for both
    generation and source display, avoiding the issue where re-retrieving
    with MultiQueryRetriever returns more docs than the LLM actually saw.
    """
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    rag_chain = prompt | llm

    return rag_chain
