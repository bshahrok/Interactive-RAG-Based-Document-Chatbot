"""
Unit tests for the vector store module.

Tests cover ChromaVectorStore initialization, CRUD operations, querying,
and statistics retrieval.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

from src.vector_store import ChromaVectorStore
from src.chunk_types import Chunk, RetrievedChunk


def create_sample_chunks(num_chunks: int = 3, document_id: str = None) -> list:
    """Helper to create sample chunks for testing."""
    if document_id is None:
        document_id = str(uuid.uuid4())
    
    chunks = []
    for i in range(num_chunks):
        chunk = Chunk(
            chunk_id=f"{document_id}_chunk_{i}",
            document_id=document_id,
            filename="test_doc.txt",
            filetype="txt",
            chunk_index=i,
            location=i * 100,
            chunk_text=f"This is chunk {i} with some sample text.",
            text_preview=f"chunk {i}"
        )
        chunks.append(chunk)
    
    return chunks


def create_sample_embeddings(num_embeddings: int, dim: int = 384) -> np.ndarray:
    """Helper to create sample embeddings."""
    # Create normalized random embeddings
    embeddings = np.random.randn(num_embeddings, dim).astype(np.float32)
    # Normalize to unit length for more realistic similarity scores
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


class TempDirManager:
    """Context manager for temporary directory that handles Windows file locking."""
    
    def __init__(self):
        self.temp_dir = None
    
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        return self.temp_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except (OSError, PermissionError):
                # On Windows, Chroma may hold locks briefly
                import time
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception:
                    pass  # Best effort cleanup


def test_init_creates_collection():
    """Test that ChromaVectorStore initializes with a persistent client and collection."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir, collection_name="test_collection")
        
        assert store.persist_path == temp_dir
        assert store.collection_name == "test_collection"
        assert store.client is not None
        assert store.collection is not None
        assert store.collection.name == "test_collection"


def test_upsert_chunks_basic():
    """Test upserting chunks with embeddings."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        chunks = create_sample_chunks(3)
        embeddings = create_sample_embeddings(3)
        
        # Upsert should not raise
        store.upsert_chunks(chunks, embeddings)
        
        # Verify data was stored
        stats = store.get_stats()
        assert stats["total_chunks"] == 3
        assert stats["unique_documents"] == 1


def test_upsert_chunks_empty():
    """Test that upserting empty list does nothing."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        # Should not raise
        store.upsert_chunks([], np.array([]))
        
        stats = store.get_stats()
        assert stats["total_chunks"] == 0


def test_upsert_chunks_length_mismatch():
    """Test that mismatched chunks and embeddings raises ValueError."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        chunks = create_sample_chunks(3)
        embeddings = create_sample_embeddings(2)  # Wrong length
        
        try:
            store.upsert_chunks(chunks, embeddings)
            raise AssertionError("Expected ValueError for length mismatch")
        except ValueError as e:
            assert "Mismatch" in str(e)


def test_upsert_chunks_invalid_embedding_shape():
    """Test that 1D embeddings array raises ValueError."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        chunks = create_sample_chunks(3)
        embeddings = np.random.randn(3)  # 1D instead of 2D
        
        try:
            store.upsert_chunks(chunks, embeddings)
            raise AssertionError("Expected ValueError for invalid shape")
        except ValueError as e:
            assert "2D array" in str(e)


def test_upsert_overwrites_existing():
    """Test that upserting same chunk_id overwrites existing data."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        # Insert initial chunk
        chunks = create_sample_chunks(1)
        embeddings = create_sample_embeddings(1)
        store.upsert_chunks(chunks, embeddings)
        
        # Update with different text but same chunk_id
        chunks[0].chunk_text = "Updated text for chunk 0"
        new_embeddings = create_sample_embeddings(1)
        store.upsert_chunks(chunks, new_embeddings)
        
        # Should still have only 1 chunk
        stats = store.get_stats()
        assert stats["total_chunks"] == 1


def test_query_basic():
    """Test querying for similar chunks."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        # Insert chunks
        chunks = create_sample_chunks(5)
        embeddings = create_sample_embeddings(5)
        store.upsert_chunks(chunks, embeddings)
        
        # Query with first embedding (should match itself)
        query_embedding = embeddings[0]
        results = store.query(query_embedding, top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, RetrievedChunk) for r in results)
        
        # First result should be the exact match
        assert results[0].chunk_id == chunks[0].chunk_id
        assert results[0].score is not None
        assert 0 < results[0].score <= 1.0


def test_query_returns_sorted_by_score():
    """Test that query results are sorted by score (highest first)."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        chunks = create_sample_chunks(5)
        embeddings = create_sample_embeddings(5)
        store.upsert_chunks(chunks, embeddings)
        
        query_embedding = embeddings[0]
        results = store.query(query_embedding, top_k=5)
        
        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


def test_query_invalid_top_k():
    """Test that top_k <= 0 raises ValueError."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        query_embedding = create_sample_embeddings(1)[0]
        
        try:
            store.query(query_embedding, top_k=0)
            raise AssertionError("Expected ValueError for top_k=0")
        except ValueError as e:
            assert "positive" in str(e)


def test_query_invalid_embedding_shape():
    """Test that 2D query embedding raises ValueError."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        # 2D array instead of 1D
        query_embedding = create_sample_embeddings(1)
        
        try:
            store.query(query_embedding, top_k=5)
            raise AssertionError("Expected ValueError for 2D embedding")
        except ValueError as e:
            assert "1D array" in str(e)


def test_query_empty_store():
    """Test querying an empty store returns empty list."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        query_embedding = create_sample_embeddings(1)[0]
        results = store.query(query_embedding, top_k=5)
        
        assert results == []


def test_delete_document_basic():
    """Test deleting all chunks from a document."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        # Insert chunks from two documents
        doc1_id = str(uuid.uuid4())
        doc2_id = str(uuid.uuid4())
        
        chunks1 = create_sample_chunks(3, document_id=doc1_id)
        chunks2 = create_sample_chunks(2, document_id=doc2_id)
        
        all_chunks = chunks1 + chunks2
        embeddings = create_sample_embeddings(5)
        
        store.upsert_chunks(all_chunks, embeddings)
        
        # Delete first document
        deleted_count = store.delete_document(doc1_id)
        
        assert deleted_count == 3
        
        # Verify only doc2 chunks remain
        stats = store.get_stats()
        assert stats["total_chunks"] == 2
        assert stats["unique_documents"] == 1


def test_delete_document_nonexistent():
    """Test deleting a non-existent document returns 0."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        deleted_count = store.delete_document("nonexistent-id")
        assert deleted_count == 0


def test_list_documents_basic():
    """Test listing all documents in the store."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        # Insert chunks from two documents
        doc1_id = str(uuid.uuid4())
        doc2_id = str(uuid.uuid4())
        
        chunks1 = create_sample_chunks(3, document_id=doc1_id)
        chunks2 = create_sample_chunks(2, document_id=doc2_id)
        
        all_chunks = chunks1 + chunks2
        embeddings = create_sample_embeddings(5)
        
        store.upsert_chunks(all_chunks, embeddings)
        
        # List documents
        docs = store.list_documents()
        
        assert len(docs) == 2
        
        # Find each document
        doc1_info = next(d for d in docs if d["document_id"] == doc1_id)
        doc2_info = next(d for d in docs if d["document_id"] == doc2_id)
        
        assert doc1_info["filename"] == "test_doc.txt"
        assert doc1_info["filetype"] == "txt"
        assert doc1_info["chunk_count"] == 3
        
        assert doc2_info["chunk_count"] == 2


def test_list_documents_empty():
    """Test listing documents from an empty store."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        docs = store.list_documents()
        assert docs == []


def test_get_stats_basic():
    """Test getting store statistics."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        # Insert chunks from two documents
        doc1_id = str(uuid.uuid4())
        doc2_id = str(uuid.uuid4())
        
        chunks1 = create_sample_chunks(3, document_id=doc1_id)
        chunks2 = create_sample_chunks(4, document_id=doc2_id)
        
        all_chunks = chunks1 + chunks2
        embeddings = create_sample_embeddings(7, dim=384)
        
        store.upsert_chunks(all_chunks, embeddings)
        
        stats = store.get_stats()
        
        assert stats["total_chunks"] == 7
        assert stats["unique_documents"] == 2
        assert stats["embedding_dimension"] == 384


def test_get_stats_empty():
    """Test stats from empty store."""
    with TempDirManager() as temp_dir:
        store = ChromaVectorStore(persist_path=temp_dir)
        
        stats = store.get_stats()
        
        assert stats["total_chunks"] == 0
        assert stats["unique_documents"] == 0
        assert stats["embedding_dimension"] == 0


def test_persistence_across_instances():
    """Test that data persists across different store instances."""
    with TempDirManager() as temp_dir:
        # Create first instance and insert data
        store1 = ChromaVectorStore(persist_path=temp_dir, collection_name="persist_test")
        chunks = create_sample_chunks(3)
        embeddings = create_sample_embeddings(3)
        store1.upsert_chunks(chunks, embeddings)
        
        # Create second instance with same path
        store2 = ChromaVectorStore(persist_path=temp_dir, collection_name="persist_test")
        
        # Data should be available
        stats = store2.get_stats()
        assert stats["total_chunks"] == 3
        
        # Query should work
        query_embedding = embeddings[0]
        results = store2.query(query_embedding, top_k=1)
        assert len(results) == 1
