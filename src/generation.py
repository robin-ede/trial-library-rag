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

def get_rag_chain(retriever):
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

    def retrieve_context(question: str):
        docs = retriever.invoke(question)
        return {
            "context": format_docs(docs),
            "question": question,
            "raw_docs": docs,  # for introspection / UI
        }

    # We need to be careful with RunnablePassthrough here because we want to return the raw_docs as well.
    # The standard chain usually just returns the string answer.
    # Let's define a chain that returns the whole dict or object.
    
    # However, to keep it simple and compatible with the architecture doc:
    # The architecture doc had:
    # rag_chain = (
    #     RunnablePassthrough()
    #     | (lambda x: retrieve_context(x["question"]))
    #     | prompt
    #     | llm
    # )
    # This chain returns the LLM response (AIMessage).
    # The retrieval happens inside the lambda.
    # To get the docs out, we might need a slightly different structure or just re-retrieve in the app 
    # (as suggested in the architecture doc: "Retrieve sources again for UI").
    # I will stick to the architecture doc's suggestion of re-retrieving in the app for simplicity for now,
    # or I can make the chain return a dict.
    # Let's follow the architecture doc's pattern exactly for now to match the approved plan.
    
    rag_chain = (
        RunnablePassthrough()
        | (lambda x: retrieve_context(x["question"]))
        | prompt
        | llm
    )

    return rag_chain
