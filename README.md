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

    **Optional - LangSmith for Observability:**
    To enable cost/token tracking and LangSmith tracing, add:
    ```bash
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=lsv2_pt_your-key-here
    LANGCHAIN_PROJECT=trial-library-rag
    ```
    Sign up for free at [smith.langchain.com](https://smith.langchain.com)

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
-   **Conversation History**: Implements query rewriting to handle follow-up questions by:
    - Resolving pronouns (it, they, this, that) to specific entities from conversation history
    - Rewriting follow-up questions as standalone queries for better retrieval
    - Including last 3 conversation exchanges (6 messages) in generation context for coherent multi-turn dialogue
    - Critical for clinical workflows where users explore complex topics through iterative questioning
-   **Observability**: Integrated LangSmith for comprehensive cost tracking:
    - **LLM costs**: Tracked via `get_openai_callback()` for ChatOpenAI calls (gpt-4o-mini)
    - **Embedding costs**: Estimated for OpenAIEmbeddings calls (qwen/qwen3-embedding-8b) based on query length
    - **Per-query breakdown**: Shows LLM vs embedding costs separately
    - **Session aggregates**: Cumulative costs, tokens, and timing in sidebar
    - **LangSmith tracing**: Embedding calls now visible in LangSmith traces
    - Critical for budget-conscious clinical trial operations where every API call counts.

## Logging and Observability

The system includes production-grade structured logging for performance monitoring and debugging.

### Log Configuration

Logging is controlled via the `LOG_LEVEL` environment variable in `.env`:

```bash
# Show timing data and progress (default)
LOG_LEVEL=INFO

# Show detailed step-by-step execution
LOG_LEVEL=DEBUG

# Show only warnings and errors
LOG_LEVEL=WARNING
```

### What Gets Logged

**INFO Level (Default):**
- Document ingestion progress and completion
- Retrieval timing breakdowns (BM25, Vector, Ensemble, Multi-Query)
- Embedding API call latency (single and batch)
- Evaluation pipeline progress and results
- Document counts at each retrieval stage

**Example Output:**
```
14:32:15 - trial_library.retrieval - INFO - BM25 retrieval completed in 0.234s, retrieved 147 documents
14:32:16 - trial_library.retrieval - INFO - Vector retrieval completed in 0.456s, retrieved 3 documents
14:32:16 - trial_library.retrieval - INFO - Ensemble retrieval completed in 0.734s (BM25: 0.234s, Vector: 0.456s, Merge: 0.044s)
```

### Timing Instrumentation

All critical operations are instrumented:
- Milvus vector queries: Per-query latency + document count
- BM25/Vector retrieval: Execution time + document count
- Ensemble merging: Result deduplication time
- Multi-Query expansion: Query generation + per-variation timing
- Embedding API calls: Single query and batch latency

## Technologies: Familiar vs New

Here is a breakdown of the technology stack based on my prior experience:

-   **Familiar Technologies**:
    -   **LangChain**: Used previously for basic chains, though this project required deeper usage of retrievers and custom chains.
    -   **Streamlit**: Used for building simple data apps.
    -   **OpenRouter**: Familiar with using it as an OpenAI-compatible API gateway.

-   **New Technologies**:
    -   **Milvus**: First time using Milvus (specifically Milvus Lite) for vector storage; previously used Chroma.
    -   **Docling**: New to this tool for PDF parsing; found it very robust for complex layouts compared to standard PyPDF.
    -   **Ragas**: First time implementing Ragas for automated RAG evaluation.
    -   **LangSmith**: New to using this for deep observability and cost tracking.

## Evaluation

To run the evaluation pipeline:
```bash
python -m src.evaluation
```
This runs a curated set of hand-crafted questions with known ground truth answers. Results are saved to `evaluation_results.csv`.

### Results Summary

The system was evaluated on a set of clinical trial questions ranging from easy to hard. Here are the aggregate metrics from the latest run:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Retrieval Recall** | **1.00** | The system successfully retrieved the relevant information for 100% of the questions. Hybrid search + Query Expansion is highly effective. |
| **Citation Accuracy** | **0.90** | The model correctly cited the source document 90% of the time, which is critical for clinical trust. |
| **Faithfulness** | **0.87** | High faithfulness indicates the answers are derived directly from the retrieved context, minimizing hallucinations. |
| **Answer Relevancy** | **0.74** | Answers are generally relevant to the user's query. |
| **Context Precision** | **0.39** | Lower precision is an expected trade-off for our aggressive recall strategy (Multi-Query Expansion + Hybrid Search). We intentionally omitted a reranker step to avoid additional API costs and latency, accepting that some irrelevant chunks will be retrieved alongside the correct ones. |
| **Ground Truth Match** | **0.46** | Exact string matching is low, which is expected for generative tasks. Semantic similarity (Answer Relevancy) is a better indicator of quality. |

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

## Limitations (Known Issues)

-   **PDF Parsing**: Uses **Docling** with DoclingLoader and HybridChunker. This handles complex clinical trial layouts, tables, and multi-column text with layout-aware parsing and tokenization-aware chunking.
-   **Retrieval**: Uses Advanced Retrieval (Hybrid Search with BM25 + Vector, plus Multi-Query Expansion) for improved accuracy.
-   **No Reranking**: Originally implemented reranking, but removed due to local model constraints and lack of OpenRouter support.
-   **No Page Numbers**: Due to the markdown-aware chunking strategy (which prioritizes table/section integrity), specific page numbers are not available for citations.
-   **Limited History Persistence**: Conversation history works within a session (follow-up questions, pronoun resolution), but is lost on page refresh. No cross-session memory.
-   **Milvus Lite Single Connection**: Milvus Lite only supports one connection per database file. The app caches the retriever to avoid connection issues. Don't run evaluation while Streamlit is running.

## Future Improvements (What I'd Change)

-   **Reranking Step**: Implement a Cross-Encoder (e.g., zerank-2) to re-score retrieved chunks. This would directly address the low Context Precision (0.39) by filtering out irrelevant chunks before generation.
-   **Agentic Workflow**: Move from a linear RAG chain to an agentic loop (e.g., LangGraph) that can decide to search again or ask clarifying questions if the initial retrieval is insufficient.
-   **Structured Output**: Use function calling to force the LLM to output structured data (e.g., JSON) for easier downstream processing.
-   **Persistent Session History**: Implement cross-session persistence (database or file-based) so conversation history survives page refreshes and can be resumed later.
-   **Advanced Fusion**: Explore RBF (Radial Basis Function) or RRF (Reciprocal Rank Fusion) for more sophisticated result merging.
