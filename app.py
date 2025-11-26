import streamlit as st
import os
import time
from dotenv import load_dotenv

# Load environment variables FIRST (including LangSmith config)
load_dotenv()

from langchain_community.callbacks import get_openai_callback
from src.retrieval import get_advanced_retriever
from src.generation import get_rag_chain, format_docs, rewrite_query_with_history

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
        # Display metrics if available
        if "metrics" in msg:
            metrics = msg["metrics"]
            st.caption(
                f"⏱️ {metrics['total_time']:.2f}s (retrieval: {metrics['retrieval_time']:.2f}s) | "
                f"📊 {metrics['llm_tokens']:,} LLM tokens | "
                f"💰 ${metrics['llm_cost']:.5f} LLM cost"
            )

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

                # Get conversation history (excluding current question)
                chat_history = st.session_state["messages"][:-1]  # Exclude the just-added user message

                # Rewrite query with conversation history for better retrieval
                # Track the rewriting cost
                rewrite_start = time.time()
                with get_openai_callback() as rewrite_cb:
                    rewritten_query = rewrite_query_with_history(user_input, chat_history)
                rewrite_time = time.time() - rewrite_start

                # Calculate rewrite cost (gpt-4o-mini pricing) - only if there was a rewrite
                rewrite_tokens = rewrite_cb.total_tokens
                if rewrite_tokens > 0:
                    rewrite_input_tokens = getattr(rewrite_cb, 'prompt_tokens', int(rewrite_tokens * 0.8))
                    rewrite_output_tokens = getattr(rewrite_cb, 'completion_tokens', rewrite_tokens - rewrite_input_tokens)
                    rewrite_cost = (rewrite_input_tokens * 0.15 / 1_000_000) + (rewrite_output_tokens * 0.60 / 1_000_000)
                else:
                    rewrite_cost = 0.0

                # Retrieve documents ONCE using rewritten query
                # This ensures UI shows exactly what the LLM saw
                # Use cached retriever to avoid Milvus Lite connection issues
                retrieval_start = time.time()
                try:
                    docs = base_retriever.invoke(rewritten_query)
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

                    # No LLM call when no docs found, but include rewrite cost
                    total_tokens = rewrite_tokens
                    total_cost = rewrite_cost
                    llm_tokens = 0
                    llm_cost = 0.0
                else:
                    # Format context from retrieved docs
                    context = format_docs(docs)

                    # Format conversation history for generation prompt
                    # Include last 3 exchanges (6 messages) for context
                    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
                    if recent_history:
                        history_text = "Conversation History:\n" + "\n".join([
                            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                            for msg in recent_history
                        ]) + "\n"
                    else:
                        history_text = ""

                    # Run RAG with streaming and callback tracking
                    stream_handler = st.empty()
                    full_response = ""

                    # Use OpenAI callback to track token usage
                    with get_openai_callback() as cb:
                        # Stream the LLM response with conversation history
                        response_generator = rag_chain.stream({
                            "history": history_text,
                            "context": context,
                            "question": user_input
                        })

                        for chunk in response_generator:
                            content = chunk.content if hasattr(chunk, "content") else str(chunk)
                            full_response += content
                            stream_handler.markdown(full_response + "▌")

                        stream_handler.markdown(full_response)
                        answer = full_response

                        # Get LLM token data from callback
                        llm_tokens = cb.total_tokens

                        # Manual cost calculation for OpenRouter gpt-4o-mini
                        # Pricing: $0.15/1M input tokens, $0.60/1M output tokens
                        input_tokens = getattr(cb, 'prompt_tokens', int(llm_tokens * 0.8))
                        output_tokens = getattr(cb, 'completion_tokens', llm_tokens - input_tokens)
                        llm_cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

                        # Include query rewriting cost in totals
                        total_tokens = llm_tokens + rewrite_tokens
                        total_cost = llm_cost + rewrite_cost

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

                # Show per-query metrics
                st.caption(
                    f"⏱️ {total_time:.2f}s (retrieval: {retrieval_time:.2f}s) | "
                    f"📊 {llm_tokens:,} LLM tokens | "
                    f"💰 ${llm_cost:.5f} LLM cost"
                )

            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
                answer = "I apologize, but I encountered an error while processing your request."
                sources_text = ""
                # Set default values for metrics in case of error
                total_time = 0.0
                retrieval_time = 0.0
                llm_tokens = 0
                llm_cost = 0.0

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources_text,
            "metrics": {
                "total_time": total_time,
                "retrieval_time": retrieval_time,
                "llm_tokens": llm_tokens,
                "llm_cost": llm_cost,
            }
        }
    )

    # Force rerun to display the message with metrics from history
    st.rerun()
