## 1. Strategic Approach: "Healthcare-Grade" RAG (with OpenRouter)

Everything about the original strategy still holds:

- Use **oncology / clinical guideline PDFs** as your corpus (NCCN excerpts, ASCO guidelines, trial protocols, etc.).
- Optimize for **low hallucination, strong traceability**, and **rigorous evaluation**.
- Spend ~40% on the app and **60% on evaluation** (Ragas, synthetic test set, LLM-as-a-judge).

The only major change: all model calls go through **OpenRouter’s OpenAI-compatible endpoint** instead of directly to OpenAI.

---

## 2. Tech Stack (Updated for OpenRouter)

- **Language:** Python 3.10+
- **LLM & Embeddings via OpenRouter:**
  - `ChatOpenAI` + `OpenAIEmbeddings` (from `langchain-openai`), pointing to OpenRouter.
  - Models (example):
    - Chat: `openai/gpt-4o-mini` (fast, cheap, strong reasoning).
    - Embeddings: `openai/text-embedding-3-small`.
- **Vector Store:** ChromaDB (`langchain-chroma`).
- **PDF Parsing:** PyMuPDF (`fitz`) via `PyMuPDFLoader`.
- **UI:** Streamlit.
- **Orchestration:** LangChain.
- **Evaluation:** Ragas (faithfulness, answer relevancy, context precision).

### OpenRouter Configuration

In your `.env` (used by both backend and evaluation):

```bash
# OpenRouter as OpenAI-compatible backend
OPENAI_API_KEY=sk-or-xxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://openrouter.ai/api/v1
# Optional but recommended by OpenRouter
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
```

LangChain’s `langchain-openai` will pick up `OPENAI_API_KEY` and `OPENAI_API_BASE` automatically, or you can pass `base_url` explicitly.

---

## 3. Project Structure

Same structure; only the provider changes.

```text
trial-library-rag/
├── data/                   # 3–5 oncology / clinical PDFs
├── src/
│   ├── __init__.py
│   ├── ingestion.py        # PDF loading, chunking, vector store build
│   ├── retrieval.py        # Retrieval logic over Chroma
│   ├── generation.py       # RAG chain, prompts, OpenRouter LLM calls
│   └── evaluation.py       # Ragas-based evaluation pipeline
├── app.py                  # Streamlit chat UI
├── requirements.txt
├── README.md               # Design, tradeoffs, evaluation results
└── .env                    # API keys (gitignored)
```

---

## 4. Component Architecture (Unchanged Conceptually, Updated for OpenRouter)

### Phase A: Ingestion & Indexing (`ingestion.py`)

1. **Load PDFs** via `PyMuPDFLoader` so you keep page-level metadata.
2. **Chunk** with `RecursiveCharacterTextSplitter`:
   - Chunk size: ~800–1000 characters (or tokens if you use a token splitter).
   - Overlap: ~150–200.
   - Ensure metadata includes: `source` (filename) and `page` (page number).
3. **Embed & Store:**
   - Use **OpenRouter** + `OpenAIEmbeddings` with `openai/text-embedding-3-small`.
   - Persist Chroma to disk (e.g., `./chroma_db`) to avoid re-embedding.

### Phase B: Retrieval & Generation (`retrieval.py`, `generation.py`)

1. **Retrieval:**
   - Use `vectorstore.as_retriever(search_type="similarity", k=5)`.
   - Optionally add a **minimum similarity threshold** for “no answer” behavior.

2. **Prompting / Generation:**
   - LLM: `openai/gpt-4o-mini` via OpenRouter.
   - System message should be **strict** about:
     - Using only provided context.
     - Explicitly saying “I don’t know” when context is insufficient.
     - Citing **document name and page number**.

   Example system message:

   > You are a careful assistant for oncology clinical trial recruiting.  
   > Use only the provided context to answer.  
   > If the answer is not contained in the context, say you do not know and suggest asking a clinician.  
   > For any factual statement, cite the source document name and page number in parentheses at the end of the sentence, e.g. (NCCN_Breast_2024.pdf, p.12).

### Phase C: UI (`app.py`)

- Use `st.chat_input`/`st.chat_message` and `st.session_state` for chat history.
- Display:
  - Main answer (with inline citations).
  - Below or in an `st.expander("View Sources")`, show:
    - The underlying chunks.
    - Their metadata (filename, page).
- This is critical for **explainability** in healthcare.

---

## 5. Evaluation Framework (LLM-as-a-Judge via OpenRouter)

Core idea remains:

1. **Synthetic QA dataset from your own chunks.**
2. **Run your RAG system on those questions.**
3. **Evaluate** with Ragas:
   - Faithfulness (hallucination control).
   - Answer Relevancy.
   - Context Precision (retrieval quality).

All LLM calls within Ragas (for metric scoring) will also go through **OpenRouter**.

---

## 6. Step-by-Step Implementation (Updated Code Skeletons)

### 6.1 `ingestion.py` – Data Prep with OpenRouter Embeddings

```python
# src/ingestion.py
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
    vectorstore.persist()
    return vectorstore

def ingest_docs():
    docs = load_pdfs("./data")
    # Simple check for empty pages (scanned images, etc.)
    docs = [d for d in docs if d.page_content and len(d.page_content.strip()) > 10]

    splits = split_docs(docs)
    return build_vectorstore(splits)
```

---

### 6.2 `retrieval.py` – Retriever Logic

```python
# src/retrieval.py
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
```

You can add a wrapper to filter by similarity score (if you expose scores).

---

### 6.3 `generation.py` – RAG Chain Using OpenRouter LLM

```python
# src/generation.py
import os
from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a careful assistant for oncology clinical trial recruiting.

Use ONLY the provided context to answer the question.
If you cannot fully answer from the context, say you do not know and suggest asking a clinician.

For any factual statement, cite the source document and page number in parentheses at the end
of the sentence, e.g. (NCCN_Breast_2024.pdf, p.12).
"""

def format_docs(docs: List[Document]) -> str:
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "unknown_source")
        page = d.metadata.get("page", "n/a")
        formatted.append(f"[{source}, page {page}]\n{d.page_content}")
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
        docs = retriever.get_relevant_documents(question)
        return {
            "context": format_docs(docs),
            "question": question,
            "raw_docs": docs,  # for introspection / UI
        }

    rag_chain = (
        RunnablePassthrough()
        | (lambda x: retrieve_context(x["question"]))
        | prompt
        | llm
    )

    return rag_chain
```

In your app, you’ll keep the `raw_docs` around for the “View Sources” panel.

---

### 6.4 `app.py` – Streamlit Chat UI

```python
# app.py
import streamlit as st
from src.retrieval import get_retriever
from src.generation import get_rag_chain, format_docs

st.set_page_config(page_title="Trial Library RAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("Oncology Trial Library RAG Demo")

retriever = get_retriever(k=5)
rag_chain = get_rag_chain(retriever)

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View Sources"):
                st.markdown(msg["sources"])

user_input = st.chat_input("Ask a question about these guidelines or trials")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Run RAG
            result = rag_chain.invoke({"question": user_input})
            answer = result.content if hasattr(result, "content") else str(result)
            st.markdown(answer)

            # Retrieve sources again for UI
            docs = retriever.get_relevant_documents(user_input)
            sources_text = format_docs(docs)
            with st.expander("View Sources"):
                st.markdown(sources_text)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources_text,
        }
    )
```

---

### 6.5 `evaluation.py` – Ragas Evaluation (Using OpenRouter)

Key pieces:

1. **Generate synthetic questions** from random chunks using the same `ChatOpenAI` (via OpenRouter).
2. **Run your RAG chain** on those questions.
3. **Evaluate** with Ragas metrics.

Skeleton below is simplified but shows the flow.

```python
# src/evaluation.py
import random
import os
from datasets import Dataset
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from src.retrieval import get_retriever
from src.generation import get_rag_chain

load_dotenv()

NUM_QUESTIONS = 5

def sample_chunks(vectorstore, n=NUM_QUESTIONS):
    # Chroma API: get all documents (you can also store docs separately)
    # For small corpora this is fine; for larger, maintain splits separately.
    docs = vectorstore._collection.get(include=["documents", "metadatas"])
    all_docs = list(
        zip(docs["documents"], docs["metadatas"])
    )
    return random.sample(all_docs, min(n, len(all_docs)))

def generate_question_from_chunk(llm, text: str) -> str:
    prompt = f"""
You are generating a question for RAG evaluation.

Given this text (from oncology guidelines or a clinical trial document):

\"\"\"{text}\"\"\"

Generate ONE specific, answerable question that can be fully answered using ONLY the given text.
The question should be factual and not vague.
Return only the question.
"""
    resp = llm.invoke(prompt)
    return resp.content.strip()

def main():
    # LLM via OpenRouter
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Load vectorstore & retriever
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    retriever = get_retriever(k=5)
    rag_chain = get_rag_chain(retriever)

    sampled = sample_chunks(vectorstore, NUM_QUESTIONS)

    questions = []
    ground_truths = []
    retrieved_contexts = []
    answers = []

    for text, meta in sampled:
        q = generate_question_from_chunk(llm, text)
        questions.append(q)
        ground_truths.append(text)

        # Run RAG
        result = rag_chain.invoke({"question": q})
        answer = result.content if hasattr(result, "content") else str(result)
        answers.append(answer)

        # Get retrieved docs to pass to Ragas as contexts
        docs = retriever.get_relevant_documents(q)
        ctxs = [d.page_content for d in docs]
        retrieved_contexts.append(ctxs)

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truths,
        }
    )

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print(results)
    results.to_pandas().to_csv("evaluation_results.csv", index=False)

if __name__ == "__main__":
    main()
```

You’ll then:
- Include `evaluation_results.csv` or a summarized table in `README.md`.
- Analyze failures: mismatch between retrieved context and ground truth, hallucinations, etc.

---

## 7. Handling Edge Cases (Still Important)

1. **Empty Retrieval / Low Similarity:**
   - If Chroma’s top similarity score is below a threshold, skip answering and return:
     > I cannot find information about that in the provided documents.  
     > Please consult a clinician or additional sources.
2. **Rate Limits / Timeouts:**
   - Wrap OpenRouter calls (`ChatOpenAI`, `OpenAIEmbeddings`) in `try/except`.
   - Optionally exponential backoff for transient errors.
3. **Scanned PDFs / Bad Text:**
   - During ingestion, skip pages with very short content.
   - Log warnings for empty or near-empty text; mention in README.

---

## 8. README Strategy (Updated for OpenRouter)

Your `README.md` should clearly state:

1. **Setup:**
   ```bash
   pip install -r requirements.txt
   cp .env.example .env  # then fill in your OpenRouter key
   python -m src.ingestion  # or python -c "from src.ingestion import ingest_docs; ingest_docs()"
   streamlit run app.py
   ```

2. **OpenRouter Configuration:**
   - Explain the `OPENAI_API_KEY` + `OPENAI_API_BASE` setup.
   - Mention chosen models:
     - Chat: `openai/gpt-4o-mini`.
     - Embeddings: `openai/text-embedding-3-small`.

3. **Design Decisions:**
   - Why chunk size ~1000 & overlap 200 (keep guideline paragraphs intact).
   - Why Chroma (simple, local, persistent).
   - Why strict system prompt + temperature 0 (hallucination control in healthcare).

4. **Evaluation Report:**
   - Include a Markdown table from `evaluation_results.csv`:
     - Columns: `question`, `faithfulness`, `answer_relevancy`, `context_precision`.
   - Brief analysis:
     - Where retrieval failed (low context_precision).
     - Any hallucinations (low faithfulness) and suspected reasons (tables, odd formatting, etc.).
   - Potential improvements:
     - Better chunking (hierarchical, section-aware).
     - Hybrid retrieval (BM25 + vectors).
     - Per-document retriever or metadata filters (cancer type, line of therapy).

---

This gives you a full, OpenRouter-based RAG setup with:

- Oncology-relevant PDFs.
- Strict, citation-heavy prompting.
- Streamlit chat UI with source transparency.
- A **Ragas evaluation pipeline** that uses LLM-as-a-judge via OpenRouter.

You can now adapt this into the actual repo, fill in any missing glue code, and tune the evaluation to showcase your rigor.