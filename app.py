import streamlit as st
from src.retrieval import get_ensemble_retriever
from src.generation import get_rag_chain, format_docs

st.set_page_config(page_title="Trial Library RAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("Oncology Trial Library RAG Demo")

# Initialize RAG components
# We might want to cache these resources
@st.cache_resource
def load_rag_components():
    retriever = get_ensemble_retriever(k=5)
    rag_chain = get_rag_chain(retriever)
    return retriever, rag_chain

try:
    retriever, rag_chain = load_rag_components()
except Exception as e:
    st.error(f"Failed to load RAG components. Make sure you have ingested data and set .env correctly. Error: {e}")
    st.stop()

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

            # Retrieve sources again for UI (as per architecture doc strategy)
            docs = retriever.invoke(user_input)
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
