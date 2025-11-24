import streamlit as st
from src.retrieval import get_advanced_retriever
from src.generation import get_rag_chain, format_docs

st.set_page_config(page_title="Trial Library RAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("Oncology Trial Library RAG Demo")

with st.sidebar:
    st.header("Knowledge Base")
    try:
        import os
        pdf_files = [f for f in os.listdir("./data") if f.lower().endswith(".pdf")]
        if pdf_files:
            st.success(f"Loaded {len(pdf_files)} documents:")
            for f in pdf_files:
                st.markdown(f"- 📄 {f}")
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
    
    # Filters
    st.subheader("Filters")
    cancer_type_filter = st.multiselect("Cancer Type", ["Breast", "Lung", "General", "Unknown"])
    doc_type_filter = st.multiselect("Document Type", ["Guideline", "Trial", "Report", "Unknown"])
    
    filter_conditions = []
    if cancer_type_filter:
        filter_conditions.append({"cancer_type": {"$in": cancer_type_filter}})
    if doc_type_filter:
        filter_conditions.append({"doc_type": {"$in": doc_type_filter}})
    
    filter_dict = {}
    if len(filter_conditions) == 1:
        filter_dict = filter_conditions[0]
    elif len(filter_conditions) > 1:
        filter_dict = {"$and": filter_conditions}
        
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state["messages"] = []
        st.rerun()

# Initialize RAG components
# We might want to cache these resources
@st.cache_resource
def load_rag_components(filter_dict=None):
    retriever = get_advanced_retriever(k=5, filter=filter_dict)
    rag_chain = get_rag_chain(retriever)
    return retriever, rag_chain

try:
    # Note: st.cache_resource with dict argument might be tricky if dict is not hashable (it is mutable).
    # But Streamlit handles dicts in cache keys usually.
    # To be safe, we can turn it into a tuple of items for caching if needed, but let's try direct first.
    retriever, rag_chain = load_rag_components(filter_dict)
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
            try:
                # Run RAG with streaming
                # We use a placeholder to stream the response
                stream_handler = st.empty()
                full_response = ""
                
                # Create a generator for the stream
                response_generator = rag_chain.stream({"question": user_input})
                
                # Iterate through the stream
                for chunk in response_generator:
                    # Check if chunk is AIMessage or string (depending on chain output)
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_response += content
                    stream_handler.markdown(full_response + "▌")
                
                stream_handler.markdown(full_response)
                answer = full_response

                # Retrieve sources again for UI (as per architecture doc strategy)
                # We do this after streaming to not block the first token
                docs = retriever.invoke(user_input)
                sources_text = format_docs(docs)
                with st.expander("View Sources"):
                    st.markdown(sources_text)
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
