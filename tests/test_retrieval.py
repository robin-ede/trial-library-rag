import pytest
from unittest.mock import MagicMock, patch
from src.retrieval import (
    get_vectorstore,
    get_retriever,
    get_bm25_retriever,
    get_ensemble_retriever,
    get_advanced_retriever
)

@pytest.fixture
def mock_milvus():
    with patch("src.retrieval.Milvus") as mock:
        yield mock

@pytest.fixture
def mock_openai_embeddings():
    with patch("src.retrieval.OpenAIEmbeddings") as mock:
        yield mock

@pytest.fixture
def mock_bm25_retriever():
    with patch("src.retrieval.BM25Retriever") as mock:
        yield mock

@pytest.fixture
def mock_ensemble_retriever():
    with patch("src.retrieval.EnsembleRetriever") as mock:
        yield mock

@pytest.fixture
def mock_multi_query_retriever():
    with patch("src.retrieval.MultiQueryRetriever") as mock:
        yield mock

@pytest.fixture
def mock_chat_openai():
    with patch("src.retrieval.ChatOpenAI") as mock:
        yield mock

def test_get_vectorstore(mock_milvus, mock_openai_embeddings):
    vectorstore = get_vectorstore()
    
    mock_openai_embeddings.assert_called_once()
    mock_milvus.assert_called_once()
    assert vectorstore == mock_milvus.return_value

def test_get_retriever(mock_milvus, mock_openai_embeddings):
    mock_vectorstore = mock_milvus.return_value
    mock_retriever = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_retriever
    
    retriever = get_retriever(k=10, filter={"source": "test"})
    
    mock_vectorstore.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 10, "filter": {"source": "test"}}
    )
    assert retriever == mock_retriever

def test_get_bm25_retriever(mock_bm25_retriever):
    docs = [MagicMock(), MagicMock()]
    get_bm25_retriever(docs, k=3)
    
    mock_bm25_retriever.from_documents.assert_called_once_with(docs, k=3)

def test_get_ensemble_retriever_success(mock_milvus, mock_bm25_retriever, mock_ensemble_retriever, mock_openai_embeddings):
    # Mock vectorstore behavior
    mock_vectorstore = mock_milvus.return_value
    # Ensure docs have content > 10 chars to pass the filter
    mock_docs = [MagicMock(page_content="content_long_enough_1"), MagicMock(page_content="content_long_enough_2")]
    mock_vectorstore.similarity_search.return_value = mock_docs
    
    # Mock retrievers
    mock_bm25 = MagicMock()
    mock_bm25_retriever.from_documents.return_value = mock_bm25
    
    mock_vector_retriever = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_vector_retriever
    
    ensemble = get_ensemble_retriever(k=5)
    
    # Verify vectorstore interaction
    mock_vectorstore.similarity_search.assert_called_once()
    
    # Verify BM25 creation
    mock_bm25_retriever.from_documents.assert_called_once_with(mock_docs, k=5)
    
    # Verify Ensemble creation
    mock_ensemble_retriever.assert_called_once_with(
        retrievers=[mock_bm25, mock_vector_retriever],
        weights=[0.5, 0.5]
    )

def test_get_ensemble_retriever_fallback_no_docs(mock_milvus, mock_bm25_retriever, mock_ensemble_retriever, mock_openai_embeddings):
    """Test fallback to vector retriever when no valid docs found for BM25."""
    mock_vectorstore = mock_milvus.return_value
    
    # Return empty docs or short docs that get filtered out
    mock_vectorstore.similarity_search.return_value = [MagicMock(page_content="short")]
    
    mock_vector_retriever = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_vector_retriever
    
    retriever = get_ensemble_retriever(k=5)
    
    # Verify we got the vector retriever directly, not an ensemble
    assert retriever == mock_vector_retriever
    
    # Verify BM25 was NOT initialized
    mock_bm25_retriever.from_documents.assert_not_called()
    mock_ensemble_retriever.assert_not_called()

def test_get_ensemble_retriever_fallback_exception(mock_milvus, mock_bm25_retriever, mock_ensemble_retriever, mock_openai_embeddings):
    """Test fallback to vector retriever when Milvus query fails."""
    mock_vectorstore = mock_milvus.return_value
    mock_vectorstore.similarity_search.side_effect = Exception("Milvus error")
    
    mock_vector_retriever = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_vector_retriever
    
    retriever = get_ensemble_retriever(k=5)
    
    assert retriever == mock_vector_retriever
    mock_bm25_retriever.from_documents.assert_not_called()

def test_get_advanced_retriever(mock_multi_query_retriever, mock_chat_openai, mock_milvus, mock_ensemble_retriever):
    # Mock dependencies to avoid complex setup
    with patch("src.retrieval.get_ensemble_retriever") as mock_get_ensemble:
        mock_base_retriever = MagicMock()
        mock_get_ensemble.return_value = mock_base_retriever
        
        advanced_retriever = get_advanced_retriever(k=5)
        
        mock_get_ensemble.assert_called_once_with(k=5, filter=None)
        mock_chat_openai.assert_called_once()
        mock_multi_query_retriever.from_llm.assert_called_once()
