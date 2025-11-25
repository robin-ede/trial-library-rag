# Trial Library RAG

A "Healthcare-Grade" Retrieval-Augmented Generation (RAG) system designed for oncology clinical trial recruiting. This system uses OpenRouter (OpenAI-compatible) for LLM and embedding calls, Milvus for vector storage, and Streamlit for the user interface.

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

    Current dataset includes:
    - `nscl.pdf` - NCCN Non-Small Cell Lung Cancer Guidelines
    - `fda_guidance.pdf` - FDA Oncology Guidance Document
    - `diversity_study.pdf` - Cancer Research Diversity Study

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

-   **OpenRouter**: Used as the LLM provider to access models like `gpt-4o-mini` and `qwen/qwen3-embedding-8b` in an OpenAI-compatible way.
-   **Milvus**: Chosen for its robustness, scalability, and ability to run locally via Milvus Lite (embedded).
-   **Chunking Strategy**: **HybridChunker** from Docling. Uses tokenization-aware chunking that respects document structure, token limits, and semantic boundaries for optimal retrieval performance.
-   **Strict Prompting**: The system prompt is designed to be strict about using only the provided context and citing sources (document name) to minimize hallucinations, which is critical in healthcare.
-   **Evaluation**: Uses curated question-answer pairs with known ground truth for reliable measurement. Includes both Ragas metrics (faithfulness, answer relevancy, context precision) and custom domain-specific metrics (citation accuracy, retrieval recall).
-   **Advanced Retrieval**: Implemented using a pipeline of **Hybrid Search** (BM25 + Vector) and **Multi-Query Expansion** (for improved recall).

## Evaluation

To run the evaluation pipeline:
```bash
python -m src.evaluation
```
This runs a curated set of hand-crafted questions with known ground truth answers. Results are saved to `evaluation_results.csv`.

The evaluation includes:
- **Ragas Metrics**: faithfulness, answer_relevancy, context_precision
- **Custom Metrics**: citation_accuracy, retrieval_recall, refusal_appropriateness, ground_truth_match

To add more evaluation questions, edit `src/evaluation.py` and add entries to the `EVAL_QUESTIONS` list.

## Testing

Unit tests are available for the individual components (`src/retrieval.py` and `src/ingestion.py`). These tests mock external dependencies (Milvus, OpenAI) to ensure they are fast and reliable.

To run the tests:
```bash
python -m pytest tests/
```

## Limitations

-   **PDF Parsing**: Uses **Docling** with DoclingLoader and HybridChunker. This handles complex clinical trial layouts, tables, and multi-column text with layout-aware parsing and tokenization-aware chunking.
-   **Retrieval**: Uses Advanced Retrieval (Hybrid Search with BM25 + Vector, plus Multi-Query Expansion) for improved accuracy.
-   **No Reranking**: Originally implemented reranking, but removed due to local model constraints and lack of OpenRouter support.
-   **No Page Numbers**: Due to the markdown-aware chunking strategy (which prioritizes table/section integrity), specific page numbers are not available for citations.
-   **No History Persistence**: Chat history is lost on page refresh.

## Future Improvements

-   Explore RBF (Radial Basis Function) or RRF (Reciprocal Rank Fusion) for advanced result fusion.
-   Improve PDF parsing for tables.
-   Add user authentication and history persistence.
