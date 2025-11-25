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
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "unknown_source")
        # PyMuPDFLoader usually puts full path in 'source', we might want just basename
        source = os.path.basename(source)
        page = d.metadata.get("page", "n/a")
        # Page numbers from PyMuPDF are 0-indexed, usually we want 1-indexed for humans
        try:
            page = int(page) + 1
        except:
            pass
            
        # Handle images
        images = d.metadata.get("images", [])
        image_text = ""
        if images:
            # Just list the filenames for now
            image_refs = ", ".join([os.path.basename(img) for img in images])
            image_text = f"\n[Images: {image_refs}]"

        formatted.append(f"[{source}, page {page}]{image_text}\n{d.page_content}")
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
