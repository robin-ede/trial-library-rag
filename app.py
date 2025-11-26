import streamlit as st
import os
import time
from dotenv import load_dotenv

# Load environment variables FIRST (including LangSmith config)
load_dotenv()

from langchain_community.callbacks import get_openai_callback
from src.retrieval import get_advanced_retriever
from src.generation import get_rag_chain, format_docs

st.set_page_config(page_title="Trial Library RAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "selected_sources" not in st.session_state:
    st.session_state["selected_sources"] = []

if "session_stats" not in st.session_state:
    st.session_state["session_stats"] = {
        "queries": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "total_time": 0.0,
    }

st.title("Oncology Trial Library RAG Demo")

# Example questions
st.markdown("**Try asking:**")
example_questions = [
    "What is the recommended maintenance immunotherapy duration?",
    "Which biomarkers should be tested for metastatic NSCLC?",
    "What are the FDA's diversity action plan requirements?",
]

cols = st.columns(len(example_questions))
for i, q in enumerate(example_questions):
    if cols[i].button(q, key=f"example_{i}", use_container_width=True):
        st.session_state["pending_question"] = q
        st.rerun()

st.divider()

with st.sidebar:
    st.header("Knowledge Base")
    try:
        pdf_files = [f for f in os.listdir("./data") if f.lower().endswith(".pdf")]
        if pdf_files:
            st.success(f"Loaded {len(pdf_files)} documents:")

            # Document filtering
            st.subheader("Filter Sources")
            selected_docs = []
            for f in pdf_files:
                if st.checkbox(f, value=True, key=f"doc_{f}"):
                    selected_docs.append(f)
                    st.markdown(f"📄 {f}")

            st.session_state["selected_sources"] = selected_docs

            if not selected_docs:
                st.warning("⚠️ No documents selected - select at least one to ask questions")
        else:
            st.warning("No PDF documents found in ./data")
    except Exception as e:
        st.error(f"Error listing documents: {e}")

    st.divider()

    # File Upload
    st.subheader("Add Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        save_path = os.path.join("./data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved {uploaded_file.name}")

        with st.spinner("Ingesting new document..."):
            from src.ingestion import ingest_docs
            ingest_docs()
            st.cache_resource.clear()
            st.success("Ingestion complete!")
            st.rerun()

    st.divider()

    # Session Stats
    st.subheader("Session Stats")
    st.caption("Includes LLM + Embedding costs")
    stats = st.session_state["session_stats"]
    col1, col2 = st.columns(2)
    col1.metric("Queries", stats["queries"])
    col2.metric("Total Cost", f"${stats['total_cost']:.4f}")
    if stats["queries"] > 0:
        col1.metric("Avg Time", f"{stats['total_time']/stats['queries']:.2f}s")
        col2.metric("Total Tokens", f"{stats['total_tokens']:,}")

    st.divider()

    col1, col2 = st.columns(2)
    if col1.button("Clear Chat History"):
        st.session_state["messages"] = []
        st.rerun()
    if col2.button("Reset Stats"):
        st.session_state["session_stats"] = {
            "queries": 0, "total_tokens": 0,
            "total_cost": 0.0, "total_time": 0.0
        }
        st.rerun()

# Initialize RAG components
@st.cache_resource
def load_rag_chain():
    return get_rag_chain()

@st.cache_resource
def load_retriever():
    """
    Load retriever once and cache it.
    Important: Milvus Lite doesn't support multiple connections to the same DB file.
    """
    return get_advanced_retriever(k=3)

try:
    rag_chain = load_rag_chain()
    base_retriever = load_retriever()
except Exception as e:
    st.error(f"Failed to load RAG components. Make sure you have set .env correctly. Error: {e}")
    st.stop()

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            # Count sources from formatted text (each source ends with ---)
            num_sources = msg["sources"].count("---") if msg["sources"] != "No sources found." else 0
            label = f"View Sources ({num_sources} chunks)" if num_sources > 0 else "View Sources"
            with st.expander(label):
                st.markdown(msg["sources"])

user_input = st.chat_input("Ask a question about these guidelines or trials")

# Check for pending question from example buttons
if "pending_question" in st.session_state:
    user_input = st.session_state["pending_question"]
    del st.session_state["pending_question"]

if user_input:
    # Check if any documents are selected
    if not st.session_state.get("selected_sources"):
        st.error("⚠️ Please select at least one document from the sidebar to ask questions.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                query_start_time = time.time()
                selected_sources = st.session_state["selected_sources"]

                # Retrieve documents ONCE (before generation)
                # This ensures UI shows exactly what the LLM saw
                # Use cached retriever to avoid Milvus Lite connection issues
                retrieval_start = time.time()
                try:
                    docs = base_retriever.invoke(user_input)
                    retrieval_time = time.time() - retrieval_start
                except Exception as e:
                    st.error(f"Retrieval error: {e}")
                    st.warning("This may be an embedding API issue. Check your OPENAI_API_KEY and OPENAI_API_BASE settings.")
                    raise

                # Filter docs by selected sources (post-retrieval filtering)
                if selected_sources:
                    filtered_docs = []
                    for doc in docs:
                        source = os.path.basename(doc.metadata.get("source", ""))
                        if source in selected_sources:
                            filtered_docs.append(doc)
                    docs = filtered_docs

                if not docs:
                    answer = "I could not find relevant information in the selected documents. Please try rephrasing your question or selecting different documents."
                    sources_text = "No sources found."

                    # No LLM call when no docs found
                    total_tokens = 0
                    total_cost = 0.0
                    llm_tokens = 0
                    llm_cost = 0.0
                else:
                    # Format context from retrieved docs
                    context = format_docs(docs)

                    # Run RAG with streaming and callback tracking
                    stream_handler = st.empty()
                    full_response = ""

                    # Use OpenAI callback to track token usage and cost
                    with get_openai_callback() as cb:
                        # Stream the LLM response
                        response_generator = rag_chain.stream({
                            "context": context,
                            "question": user_input
                        })

                        for chunk in response_generator:
                            content = chunk.content if hasattr(chunk, "content") else str(chunk)
                            full_response += content
                            stream_handler.markdown(full_response + "▌")

                        stream_handler.markdown(full_response)
                        answer = full_response

                        # Get LLM token and cost data from callback
                        llm_tokens = cb.total_tokens
                        llm_cost = cb.total_cost

                        # Embedding costs tracked in LangSmith (via TrackedOpenAIEmbeddings)
                        total_tokens = llm_tokens
                        total_cost = llm_cost

                    # Show sources (these are the ACTUAL docs the LLM saw)
                    sources_text = format_docs(docs)
                    with st.expander(f"View Sources ({len(docs)} chunks)"):
                        st.markdown(sources_text)

                # Calculate total query time
                total_time = time.time() - query_start_time

                # Update session stats
                st.session_state["session_stats"]["queries"] += 1
                st.session_state["session_stats"]["total_tokens"] += total_tokens
                st.session_state["session_stats"]["total_cost"] += total_cost
                st.session_state["session_stats"]["total_time"] += total_time

                # Show per-query metrics (embedding costs tracked in LangSmith)
                st.caption(
                    f"⏱️ {total_time:.2f}s (retrieval: {retrieval_time:.2f}s) | "
                    f"📊 {llm_tokens:,} LLM tokens | "
                    f"💰 ${llm_cost:.5f} LLM cost | "
                    f"📌 Embedding costs tracked in LangSmith"
                )

            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
                answer = "I apologize, but I encountered an error while processing your request."
                sources_text = ""

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources_text,
        }
    )
