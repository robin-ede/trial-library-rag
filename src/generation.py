import os
from typing import List, Dict
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

QUERY_REWRITE_PROMPT = """Given a conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that can be used for semantic search.

The rewritten question should:
- Resolve pronouns (it, they, this, that) to specific entities
- Include relevant context from the conversation
- Be clear and specific for document retrieval

If the question is already standalone, return it as-is.

Conversation History:
{history}

Follow-up Question: {question}

Rewritten Standalone Question:"""

def rewrite_query_with_history(question: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Rewrite a question to be standalone using conversation history.

    Args:
        question: The current user question
        chat_history: List of previous messages [{"role": "user"/"assistant", "content": "..."}]

    Returns:
        Rewritten standalone question suitable for retrieval
    """
    # If no history or very first question, return as-is
    if not chat_history or len(chat_history) == 0:
        return question

    # Format recent history (last 3 exchanges = 6 messages max)
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    history_text = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in recent_history
    ])

    # If history is empty after formatting, return original
    if not history_text.strip():
        return question

    # Create LLM for query rewriting
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
    chain = prompt | llm

    response = chain.invoke({
        "history": history_text,
        "question": question
    })

    rewritten = response.content.strip()
    return rewritten


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
    Get RAG chain that accepts context, question, and conversation history.

    This allows the app to:
    - Retrieve docs once and reuse them for both generation and source display
    - Include conversation history for contextual responses
    - Handle follow-up questions that reference previous exchanges
    """
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create prompt template that includes conversation history
    # Note: history is optional and will be empty string if not provided
    prompt = ChatPromptTemplate.from_template(
        """You are a careful assistant for oncology clinical trial recruiting.

Use ONLY the provided context to answer the question.
If you cannot fully answer from the context, say you do not know and suggest asking a clinician.

For any factual statement, cite the source document in parentheses at the end
of the sentence, e.g. (NCCN_Breast_2024.pdf).

{history}

Context:
{context}

Question: {question}

Answer:"""
    )

    rag_chain = prompt | llm

    return rag_chain
