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
-   **Chunking Strategy**: Recursive character splitting with ~1000 characters and 200 overlap. This size is chosen to keep paragraphs mostly intact, which is crucial for maintaining context in clinical guidelines.
-   **Strict Prompting**: The system prompt is designed to be strict about using only the provided context and citing sources (document name and page number) to minimize hallucinations, which is critical in healthcare.
-   **Evaluation**: Ragas is used for evaluation, leveraging "LLM-as-a-judge" to measure faithfulness, answer relevancy, and context precision.

## Evaluation

To run the evaluation pipeline:
```bash
python -m src.evaluation
```
This will generate synthetic questions from your data, run the RAG system, and compute metrics. Results will be saved to `evaluation_results.csv`.

## Limitations

-   **PDF Parsing**: Currently uses `PyMuPDFLoader`. Complex layouts (tables, multi-column) might need more advanced parsing strategies.
-   **Retrieval**: Basic similarity search. Could be improved with hybrid search (keyword + vector) or metadata filtering.
-   **No History Persistence**: Chat history is lost on page refresh.

## Future Improvements

-   Add hybrid search (BM25 + Dense).
-   Implement metadata filtering (e.g., filter by cancer type).
-   Improve PDF parsing for tables.
-   Add user authentication and history persistence.
