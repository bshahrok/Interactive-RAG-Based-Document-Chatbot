"""
Document ingestion module for extracting text from PDF, DOCX, and TXT files.

This module provides functionality to ingest documents and extract their text
content along with metadata. It supports PDF (with page-aware extraction),
DOCX (with paragraph-aware extraction), and TXT files.
"""

import io
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Union, Tuple, List, Optional

from pypdf import PdfReader
from docx import Document


@dataclass
class DocumentMetadata:
    """Metadata for an ingested document."""
    
    document_id: str
    filename: str
    filetype: str  # pdf | docx | txt
    ingested_at: datetime
    original_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate filetype."""
        if self.filetype not in ('pdf', 'docx', 'txt'):
            raise ValueError(f"Invalid filetype: {self.filetype}. Must be pdf, docx, or txt.")


@dataclass
class PageContent:
    """Content from a single page (PDF) or paragraph (DOCX)."""
    
    page_number: Optional[int]  # For PDF; None for DOCX/TXT
    text: str
    paragraph_index: Optional[int] = None  # For DOCX; None for PDF/TXT


@dataclass
class ExtractedText:
    """Extracted text with location information."""
    
    content: List[PageContent]
    total_pages: Optional[int] = None  # For PDF
    total_paragraphs: Optional[int] = None  # For DOCX
    
    def get_full_text(self) -> str:
        """Get all text concatenated."""
        return "\n".join(page.text for page in self.content)


def _extract_pdf(file_obj) -> ExtractedText:
    """
    Extract text from PDF with page awareness.
    
    Args:
        file_obj: File-like object containing PDF data
        
    Returns:
        ExtractedText with page-separated content
    """
    reader = PdfReader(file_obj)
    pages = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        pages.append(PageContent(
            page_number=page_num,
            text=text
        ))
    
    return ExtractedText(
        content=pages,
        total_pages=len(pages)
    )


def _extract_docx(file_obj) -> ExtractedText:
    """
    Extract text from DOCX with paragraph awareness.
    
    Args:
        file_obj: File-like object containing DOCX data
        
    Returns:
        ExtractedText with paragraph-separated content
    """
    doc = Document(file_obj)
    paragraphs = []
    
    for para_idx, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text.strip()
        # Only include non-empty paragraphs
        if text:
            paragraphs.append(PageContent(
                page_number=None,
                text=text,
                paragraph_index=para_idx
            ))
    
    return ExtractedText(
        content=paragraphs,
        total_paragraphs=len(paragraphs)
    )


def _extract_txt(file_obj) -> ExtractedText:
    """
    Extract text from TXT file (raw read).
    
    Args:
        file_obj: File-like object containing TXT data
        
    Returns:
        ExtractedText with raw content
    """
    # Read and decode the text
    if isinstance(file_obj, (io.BytesIO, io.BufferedReader)):
        text = file_obj.read().decode('utf-8')
    else:
        text = file_obj.read()
    
    return ExtractedText(
        content=[PageContent(
            page_number=None,
            text=text
        )]
    )


def ingest_file(
    source: Union[str, Path, bytes],
    filename: str,
    original_path: Optional[str] = None
) -> Tuple[DocumentMetadata, ExtractedText]:
    """
    Ingest a document file and extract its text with metadata.
    
    Args:
        source: File path (str/Path) or bytes content
        filename: Name of the file (used to determine type)
        original_path: Optional original path for metadata
        
    Returns:
        Tuple of (DocumentMetadata, ExtractedText)
        
    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If file path doesn't exist
    """
    # Determine file type from filename
    file_ext = Path(filename).suffix.lower()
    filetype_map = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.txt': 'txt'
    }
    
    if file_ext not in filetype_map:
        raise ValueError(
            f"Unsupported file type: {file_ext}. "
            f"Supported types: {', '.join(filetype_map.keys())}"
        )
    
    filetype = filetype_map[file_ext]
    
    # Open file based on source type
    if isinstance(source, bytes):
        file_obj = io.BytesIO(source)
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        file_obj = open(path, 'rb')
        if original_path is None:
            original_path = str(path)
    else:
        raise TypeError(f"Invalid source type: {type(source)}")
    
    try:
        # Extract text based on file type
        if filetype == 'pdf':
            extracted_text = _extract_pdf(file_obj)
        elif filetype == 'docx':
            extracted_text = _extract_docx(file_obj)
        elif filetype == 'txt':
            extracted_text = _extract_txt(file_obj)
        else:
            raise ValueError(f"Unsupported filetype: {filetype}")
        
        # Create metadata
        metadata = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            filename=filename,
            filetype=filetype,
            ingested_at=datetime.now(timezone.utc),
            original_path=original_path
        )
        
        return metadata, extracted_text
        
    finally:
        # Close file if we opened it
        if isinstance(source, (str, Path)):
            file_obj.close()
