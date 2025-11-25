import pytest
import os
from unittest.mock import MagicMock, patch
from src.ingestion import load_pdfs, build_vectorstore, ingest_docs

@pytest.fixture
def mock_docling_loader():
    with patch("src.ingestion.DoclingLoader") as mock:
        yield mock

@pytest.fixture
def mock_hybrid_chunker():
    with patch("src.ingestion.HybridChunker") as mock:
        yield mock

@pytest.fixture
def mock_huggingface_tokenizer():
    with patch("src.ingestion.HuggingFaceTokenizer") as mock:
        yield mock

@pytest.fixture
def mock_milvus():
    with patch("src.ingestion.Milvus") as mock:
        yield mock

@pytest.fixture
def mock_openai_embeddings():
    with patch("src.ingestion.OpenAIEmbeddings") as mock:
        yield mock

def test_load_pdfs_no_files(tmp_path):
    # Create an empty directory
    data_dir = tmp_path / "empty_data"
    data_dir.mkdir()
    
    docs = load_pdfs(data_dir=str(data_dir))
    assert docs == []

def test_load_pdfs_with_files(tmp_path, mock_docling_loader, mock_huggingface_tokenizer, mock_hybrid_chunker):
    # Create a directory with some files
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create dummy files
    (data_dir / "doc1.pdf").touch()
    (data_dir / "doc2.pdf").touch()
    (data_dir / "image.png").touch() # Should be ignored
    
    mock_loader_instance = mock_docling_loader.return_value
    mock_loader_instance.load.return_value = ["chunk1", "chunk2"]
    
    docs = load_pdfs(data_dir=str(data_dir))
    
    # Verify DoclingLoader initialization with filtered list
    call_args = mock_docling_loader.call_args
    assert call_args is not None
    file_paths = call_args.kwargs['file_path']
    
    # Check that only PDFs are included
    assert len(file_paths) == 2
    assert any("doc1.pdf" in str(p) for p in file_paths)
    assert any("doc2.pdf" in str(p) for p in file_paths)
    assert not any("image.png" in str(p) for p in file_paths)
    
    assert docs == ["chunk1", "chunk2"]

def test_build_vectorstore(mock_milvus, mock_openai_embeddings):
    splits = ["split1", "split2"]
    
    vectorstore = build_vectorstore(splits)
    
    mock_openai_embeddings.assert_called_once()
    mock_milvus.from_documents.assert_called_once()
    assert vectorstore == mock_milvus.from_documents.return_value

@patch("src.ingestion.load_pdfs")
@patch("src.ingestion.build_vectorstore")
def test_ingest_docs_success(mock_build, mock_load):
    mock_load.return_value = ["doc1"]
    mock_build.return_value = "vectorstore"
    
    result = ingest_docs()
    
    mock_load.assert_called_once()
    mock_build.assert_called_once_with(["doc1"])
    assert result == "vectorstore"

@patch("src.ingestion.load_pdfs")
def test_ingest_docs_no_docs(mock_load):
    mock_load.return_value = []
    
    result = ingest_docs()
    
    mock_load.assert_called_once()
    assert result is None
