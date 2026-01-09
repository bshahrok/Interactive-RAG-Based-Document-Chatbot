"""
Unit tests for the document ingestion module.

Tests cover PDF, DOCX, and TXT file extraction with metadata.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ingestion import (
    DocumentMetadata,
    ExtractedText,
    PageContent,
    ingest_file,
)


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
SAMPLE_DOCX = FIXTURES_DIR / "sample.docx"
SAMPLE_TXT = FIXTURES_DIR / "sample.txt"


class TestDocumentMetadata:
    """Test DocumentMetadata model."""
    
    def test_valid_metadata(self):
        """Test creating valid metadata."""
        metadata = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            filename="test.pdf",
            filetype="pdf",
            ingested_at=datetime.now(timezone.utc)
        )
        assert metadata.filetype == "pdf"
        assert metadata.filename == "test.pdf"
    
    def test_invalid_filetype(self):
        """Test that invalid filetype raises ValueError."""
        with pytest.raises(ValueError, match="Invalid filetype"):
            DocumentMetadata(
                document_id=str(uuid.uuid4()),
                filename="test.exe",
                filetype="exe",
                ingested_at=datetime.now(timezone.utc)
            )


class TestPDFIngestion:
    """Test PDF file ingestion."""
    
    def test_ingest_pdf_from_path(self):
        """Test ingesting PDF from file path."""
        metadata, extracted = ingest_file(
            source=str(SAMPLE_PDF),
            filename="sample.pdf"
        )
        
        # Check metadata
        assert metadata.filename == "sample.pdf"
        assert metadata.filetype == "pdf"
        assert metadata.document_id is not None
        assert isinstance(metadata.ingested_at, datetime)
        assert metadata.original_path == str(SAMPLE_PDF)
        
        # Check extracted text
        assert extracted.total_pages == 3
        assert len(extracted.content) == 3
        
        # Check page structure
        for idx, page in enumerate(extracted.content, start=1):
            assert page.page_number == idx
            assert page.text is not None
            assert isinstance(page.text, str)
    
    def test_pdf_page_content(self):
        """Test that PDF extraction includes expected content."""
        _, extracted = ingest_file(
            source=str(SAMPLE_PDF),
            filename="sample.pdf"
        )
        
        # Check first page contains expected text
        full_text = extracted.get_full_text()
        assert "Page 1" in full_text
        assert "Page 2" in full_text
        assert "Page 3" in full_text
    
    def test_ingest_pdf_from_bytes(self):
        """Test ingesting PDF from bytes."""
        with open(SAMPLE_PDF, 'rb') as f:
            pdf_bytes = f.read()
        
        metadata, extracted = ingest_file(
            source=pdf_bytes,
            filename="sample.pdf"
        )
        
        assert metadata.filetype == "pdf"
        assert extracted.total_pages == 3


class TestDOCXIngestion:
    """Test DOCX file ingestion."""
    
    def test_ingest_docx_from_path(self):
        """Test ingesting DOCX from file path."""
        metadata, extracted = ingest_file(
            source=str(SAMPLE_DOCX),
            filename="sample.docx"
        )
        
        # Check metadata
        assert metadata.filename == "sample.docx"
        assert metadata.filetype == "docx"
        assert metadata.document_id is not None
        assert isinstance(metadata.ingested_at, datetime)
        
        # Check extracted text
        assert extracted.total_paragraphs == 3
        assert len(extracted.content) == 3
        
        # Check paragraph structure
        for idx, para in enumerate(extracted.content):
            assert para.page_number is None
            assert para.paragraph_index == idx
            assert para.text is not None
            assert len(para.text) > 0
    
    def test_docx_paragraph_content(self):
        """Test that DOCX extraction includes expected content."""
        _, extracted = ingest_file(
            source=str(SAMPLE_DOCX),
            filename="sample.docx"
        )
        
        # Check paragraphs contain expected text
        full_text = extracted.get_full_text()
        assert "First paragraph" in full_text
        assert "Second paragraph" in full_text
        assert "Third paragraph" in full_text
    
    def test_ingest_docx_from_bytes(self):
        """Test ingesting DOCX from bytes."""
        with open(SAMPLE_DOCX, 'rb') as f:
            docx_bytes = f.read()
        
        metadata, extracted = ingest_file(
            source=docx_bytes,
            filename="sample.docx"
        )
        
        assert metadata.filetype == "docx"
        assert extracted.total_paragraphs > 0


class TestTXTIngestion:
    """Test TXT file ingestion."""
    
    def test_ingest_txt_from_path(self):
        """Test ingesting TXT from file path."""
        metadata, extracted = ingest_file(
            source=str(SAMPLE_TXT),
            filename="sample.txt"
        )
        
        # Check metadata
        assert metadata.filename == "sample.txt"
        assert metadata.filetype == "txt"
        assert metadata.document_id is not None
        assert isinstance(metadata.ingested_at, datetime)
        
        # Check extracted text
        assert len(extracted.content) == 1
        assert extracted.content[0].page_number is None
        assert extracted.content[0].text is not None
    
    def test_txt_content(self):
        """Test that TXT extraction includes expected content."""
        _, extracted = ingest_file(
            source=str(SAMPLE_TXT),
            filename="sample.txt"
        )
        
        full_text = extracted.get_full_text()
        assert "sample text file" in full_text
        assert "multiple lines" in full_text
        assert "line three" in full_text
    
    def test_ingest_txt_from_bytes(self):
        """Test ingesting TXT from bytes."""
        with open(SAMPLE_TXT, 'rb') as f:
            txt_bytes = f.read()
        
        metadata, extracted = ingest_file(
            source=txt_bytes,
            filename="sample.txt"
        )
        
        assert metadata.filetype == "txt"
        assert len(extracted.content) == 1


class TestErrorHandling:
    """Test error handling in ingestion."""
    
    def test_unsupported_file_type(self):
        """Test that unsupported file types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_file(
                source=b"test",
                filename="test.exe"
            )
    
    def test_file_not_found(self):
        """Test that missing files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ingest_file(
                source="/nonexistent/file.pdf",
                filename="file.pdf"
            )
    
    def test_invalid_source_type(self):
        """Test that invalid source types raise TypeError."""
        with pytest.raises(TypeError, match="Invalid source type"):
            ingest_file(
                source=12345,
                filename="test.pdf"
            )


class TestExtractedText:
    """Test ExtractedText utility methods."""
    
    def test_get_full_text(self):
        """Test concatenating all page content."""
        extracted = ExtractedText(
            content=[
                PageContent(page_number=1, text="Page 1 text"),
                PageContent(page_number=2, text="Page 2 text"),
                PageContent(page_number=3, text="Page 3 text"),
            ],
            total_pages=3
        )
        
        full_text = extracted.get_full_text()
        assert "Page 1 text" in full_text
        assert "Page 2 text" in full_text
        assert "Page 3 text" in full_text
