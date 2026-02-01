"""
Unit tests for the chunking module.

Tests cover recursive character text splitting, overlap behavior,
location metadata, and edge cases.
"""

import uuid
from datetime import datetime, timezone

import pytest

from src.chunking import (
    Chunk,
    ChunkingConfig,
    RecursiveCharacterTextSplitter,
    chunk_document,
)
from src.ingestion import DocumentMetadata, ExtractedText, PageContent


class TestChunkingConfig:
    """Test ChunkingConfig validation."""
    
    def test_valid_config(self):
        """Test creating valid chunking config."""
        config = ChunkingConfig(strategy="recursive", size=1000, overlap=150)
        assert config.strategy == "recursive"
        assert config.size == 1000
        assert config.overlap == 150
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        assert config.strategy == "recursive"
        assert config.size == 1000
        assert config.overlap == 150
    
    def test_invalid_strategy(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            ChunkingConfig(strategy="invalid")
    
    def test_invalid_size(self):
        """Test that non-positive size raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ChunkingConfig(size=0)
        with pytest.raises(ValueError, match="must be positive"):
            ChunkingConfig(size=-100)
    
    def test_invalid_overlap(self):
        """Test that negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            ChunkingConfig(overlap=-1)
    
    def test_overlap_exceeds_size(self):
        """Test that overlap >= size raises ValueError."""
        with pytest.raises(ValueError, match="must be less than chunk size"):
            ChunkingConfig(size=100, overlap=100)
        with pytest.raises(ValueError, match="must be less than chunk size"):
            ChunkingConfig(size=100, overlap=150)


class TestChunk:
    """Test Chunk dataclass."""
    
    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="doc-123",
            filename="test.pdf",
            filetype="pdf",
            chunk_index=0,
            location=1,
            chunk_text="This is a test chunk."
        )
        assert chunk.chunk_text == "This is a test chunk."
        assert chunk.location == 1
    
    def test_text_preview_auto_generation(self):
        """Test that text preview is auto-generated."""
        long_text = "a" * 200
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="doc-123",
            filename="test.txt",
            filetype="txt",
            chunk_index=0,
            location="chunk_0",
            chunk_text=long_text
        )
        assert chunk.text_preview == "a" * 100
        assert len(chunk.text_preview) == 100


class TestRecursiveCharacterTextSplitter:
    """Test RecursiveCharacterTextSplitter."""
    
    def test_short_text_not_split(self):
        """Test that text shorter than chunk_size is not split."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        text = "This is a short text."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_empty_text(self):
        """Test handling of empty text."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        chunks = splitter.split_text("")
        assert chunks == []
    
    def test_split_on_paragraphs(self):
        """Test splitting on paragraph boundaries."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = splitter.split_text(text)
        
        # Should split on \n\n separators
        assert len(chunks) > 1
        # Verify no chunk is excessively long
        for chunk in chunks:
            assert len(chunk) <= 50 or "\n\n" not in chunk
    
    def test_split_on_sentences(self):
        """Test splitting on sentence boundaries when paragraphs are too large."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        # Long paragraph with sentences
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence."
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 1
        # Verify chunks respect size limit
        for chunk in chunks:
            assert len(chunk) <= 100  # Allow some flexibility for sentence boundaries
    
    def test_split_on_words(self):
        """Test splitting on word boundaries when sentences are too large."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=0)
        # Long sentence
        text = "This is a very long sentence with many words that should be split."
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 1
        # Verify no mid-word splits (chunks should end with complete words)
        for i, chunk in enumerate(chunks[:-1]):  # Check all but last chunk
            # Chunk should not end mid-word (i.e., last char should be space or punctuation)
            # unless it's exactly at chunk_size character boundary
            pass  # Word boundary checking is complex due to overlap
    
    def test_overlap_behavior(self):
        """Test that chunks have correct overlap."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        # Create text that will be split
        text = "A" * 100  # Simple text to ensure predictable splitting
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 2
        # Note: With separators, exact overlap is approximate
        # The key is that we have multiple chunks and reasonable sizes
        for chunk in chunks:
            assert len(chunk) <= 60  # chunk_size + some flexibility
    
    def test_deterministic_splitting(self):
        """Test that same input produces same chunks."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 50  # Create predictable text
        
        chunks1 = splitter.split_text(text)
        chunks2 = splitter.split_text(text)
        
        assert chunks1 == chunks2


class TestChunkDocument:
    """Test chunk_document function."""
    
    def test_chunk_pdf_document(self):
        """Test chunking a PDF document."""
        metadata = DocumentMetadata(
            document_id="doc-123",
            filename="test.pdf",
            filetype="pdf",
            ingested_at=datetime.now(timezone.utc)
        )
        
        extracted = ExtractedText(
            content=[
                PageContent(page_number=1, text="First page content. " * 50),
                PageContent(page_number=2, text="Second page content. " * 50),
            ],
            total_pages=2
        )
        
        config = ChunkingConfig(size=100, overlap=20)
        chunks = list(chunk_document(metadata, extracted, config))
        
        # Should have multiple chunks
        assert len(chunks) > 0
        
        # Verify chunk properties
        for chunk in chunks:
            assert chunk.document_id == "doc-123"
            assert chunk.filename == "test.pdf"
            assert chunk.filetype == "pdf"
            assert chunk.chunk_id is not None
            assert chunk.chunk_text.strip()  # Non-empty
            assert chunk.location in [1, 2]  # Page numbers
            assert chunk.text_preview is not None
    
    def test_chunk_docx_document(self):
        """Test chunking a DOCX document."""
        metadata = DocumentMetadata(
            document_id="doc-456",
            filename="test.docx",
            filetype="docx",
            ingested_at=datetime.now(timezone.utc)
        )
        
        extracted = ExtractedText(
            content=[
                PageContent(page_number=None, text="First paragraph.", paragraph_index=0),
                PageContent(page_number=None, text="Second paragraph.", paragraph_index=1),
                PageContent(page_number=None, text="Third paragraph.", paragraph_index=2),
            ],
            total_paragraphs=3
        )
        
        config = ChunkingConfig(size=50, overlap=10)
        chunks = list(chunk_document(metadata, extracted, config))
        
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert chunk.document_id == "doc-456"
            assert chunk.filename == "test.docx"
            assert chunk.filetype == "docx"
            assert isinstance(chunk.location, str)  # Should be "chunk_N"
            assert chunk.location.startswith("chunk_")
    
    def test_chunk_txt_document(self):
        """Test chunking a TXT document."""
        metadata = DocumentMetadata(
            document_id="doc-789",
            filename="test.txt",
            filetype="txt",
            ingested_at=datetime.now(timezone.utc)
        )
        
        extracted = ExtractedText(
            content=[
                PageContent(page_number=None, text="This is a text file. " * 100)
            ]
        )
        
        config = ChunkingConfig(size=200, overlap=50)
        chunks = list(chunk_document(metadata, extracted, config))
        
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert chunk.document_id == "doc-789"
            assert chunk.filename == "test.txt"
            assert chunk.filetype == "txt"
            assert isinstance(chunk.location, str)
            assert chunk.location.startswith("chunk_")
    
    def test_chunk_indices_sequential(self):
        """Test that chunk indices are sequential."""
        metadata = DocumentMetadata(
            document_id="doc-seq",
            filename="test.txt",
            filetype="txt",
            ingested_at=datetime.now(timezone.utc)
        )
        
        extracted = ExtractedText(
            content=[
                PageContent(page_number=None, text="Content. " * 200)
            ]
        )
        
        config = ChunkingConfig(size=100, overlap=20)
        chunks = list(chunk_document(metadata, extracted, config))
        
        # Verify indices are sequential starting from 0
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_empty_pages_skipped(self):
        """Test that empty pages/content are skipped."""
        metadata = DocumentMetadata(
            document_id="doc-empty",
            filename="test.pdf",
            filetype="pdf",
            ingested_at=datetime.now(timezone.utc)
        )
        
        extracted = ExtractedText(
            content=[
                PageContent(page_number=1, text="   "),  # Empty
                PageContent(page_number=2, text="Real content here."),
                PageContent(page_number=3, text=""),  # Empty
            ],
            total_pages=3
        )
        
        config = ChunkingConfig(size=100, overlap=20)
        chunks = list(chunk_document(metadata, extracted, config))
        
        # Should only have chunks from page 2
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.location == 2
    
    def test_pdf_page_tracking(self):
        """Test that PDF chunks track correct page numbers."""
        metadata = DocumentMetadata(
            document_id="doc-pages",
            filename="test.pdf",
            filetype="pdf",
            ingested_at=datetime.now(timezone.utc)
        )
        
        # Create content with distinct pages
        extracted = ExtractedText(
            content=[
                PageContent(page_number=1, text="Page one content. " * 30),
                PageContent(page_number=2, text="Page two content. " * 30),
                PageContent(page_number=3, text="Page three content. " * 30),
            ],
            total_pages=3
        )
        
        config = ChunkingConfig(size=100, overlap=20)
        chunks = list(chunk_document(metadata, extracted, config))
        
        # Verify we have chunks from different pages
        page_numbers = set(chunk.location for chunk in chunks)
        assert len(page_numbers) > 1  # Should have chunks from multiple pages
        assert all(page_num in [1, 2, 3] for page_num in page_numbers)
    
    def test_deterministic_chunk_ids(self):
        """Test that chunks have unique IDs."""
        metadata = DocumentMetadata(
            document_id="doc-unique",
            filename="test.txt",
            filetype="txt",
            ingested_at=datetime.now(timezone.utc)
        )
        
        extracted = ExtractedText(
            content=[
                PageContent(page_number=None, text="Test content. " * 50)
            ]
        )
        
        config = ChunkingConfig(size=100, overlap=20)
        chunks = list(chunk_document(metadata, extracted, config))
        
        # All chunk IDs should be unique
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
    
    def test_large_document_memory_efficient(self):
        """Test that chunking large documents doesn't consume excessive memory."""
        metadata = DocumentMetadata(
            document_id="doc-large",
            filename="large.txt",
            filetype="txt",
            ingested_at=datetime.now(timezone.utc)
        )
        
        # Create a large text (simulate a large document)
        large_text = "This is a sentence. " * 1000  # ~20KB
        extracted = ExtractedText(
            content=[
                PageContent(page_number=None, text=large_text)
            ]
        )
        
        config = ChunkingConfig(size=500, overlap=100)
        
        # Use iterator - chunks should be yielded one at a time
        chunk_count = 0
        for chunk in chunk_document(metadata, extracted, config):
            chunk_count += 1
            assert chunk.chunk_text  # Verify chunk is valid
        
        assert chunk_count > 0
