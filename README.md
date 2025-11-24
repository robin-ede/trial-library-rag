# Trial Library RAG

A "Healthcare-Grade" Retrieval-Augmented Generation (RAG) system designed for oncology clinical trial recruiting. This system uses OpenRouter (OpenAI-compatible) for LLM and embedding calls, ChromaDB for vector storage, and Streamlit for the user interface.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd trial-library-rag
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Copy `.env.example` to `.env` and fill in your OpenRouter API key.
    ```bash
    cp .env.example .env
    ```
    Edit `.env`:
    ```bash
    OPENAI_API_KEY=sk-or-your-key-here
    OPENAI_API_BASE=https://openrouter.ai/api/v1
    ```

4.  **Add Data:**
    Place your oncology/clinical PDF documents in the `data/` directory.

5.  **Ingest Data:**
    Run the ingestion script to process PDFs and build the vector database.
    ```bash
    python -m src.ingestion
    ```

6.  **Run the App:**
    Start the Streamlit application.
    ```bash
    streamlit run app.py
    ```

## Design Decisions

-   **OpenRouter**: Used as the LLM provider to access models like `gpt-4o-mini` and `text-embedding-3-small` in an OpenAI-compatible way.
-   **ChromaDB**: Chosen for its simplicity, local persistence, and ease of integration with LangChain.
-   **Chunking Strategy**: **Markdown-Aware Splitting**. We use `MarkdownHeaderTextSplitter` to respect document structure (headers), followed by `RecursiveCharacterTextSplitter` for inner content. This preserves tables and semantic sections better than naive splitting, **at the cost of losing page number boundaries**.
-   **Strict Prompting**: The system prompt is designed to be strict about using only the provided context and citing sources (document name) to minimize hallucinations, which is critical in healthcare.
-   **Evaluation**: Ragas is used for evaluation, leveraging "LLM-as-a-judge" to measure faithfulness, answer relevancy, and context precision.
-   **Advanced Retrieval**: Implemented using a pipeline of **Hybrid Search** (BM25 + Vector), **Multi-Query Expansion** (for recall), and **Flashrank Re-ranking** (for precision).

## Evaluation

To run the evaluation pipeline:
```bash
python -m src.evaluation
```
This will generate synthetic questions from your data, run the RAG system, and compute metrics. Results will be saved to `evaluation_results.csv`.

## Limitations

-   **PDF Parsing**: Uses **LlamaParse (Agentic Mode)** with `gpt-4o-mini`. This handles complex clinical trial layouts, tables, and multi-column text significantly better than standard loaders.
-   **Retrieval**: Uses Advanced Retrieval (Ensemble + Multi-Query + Re-ranking) for improved accuracy.
-   **No Page Numbers**: Due to the markdown-aware chunking strategy (which prioritizes table/section integrity), specific page numbers are not available for citations.
-   **No History Persistence**: Chat history is lost on page refresh.

## Future Improvements

-   Explore RBF (Radial Basis Function) or RRF (Reciprocal Rank Fusion) for advanced result fusion.
-   Implement metadata filtering (e.g., filter by cancer type).
-   Improve PDF parsing for tables.
-   Add user authentication and history persistence.
