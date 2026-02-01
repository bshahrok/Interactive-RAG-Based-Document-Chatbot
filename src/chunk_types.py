from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Union
@dataclass
class DocumentMetadata:
    document_id: str
    filename: str
    filetype: Literal["pdf", "docx", "txt"]
    original_path: Optional[str]
    ingested_at: datetime
    page_count: Optional[int] = None
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[List[str]] = None
    publication_date: Optional[datetime] = None


@dataclass
class PageContent:
    page_number: Optional[int]
    text: str
    paragraph_index: Optional[int] = None


ExtractedContent = Union[str, List[PageContent]]


@dataclass
class ChunkingConfig:
    strategy: str = "fixed" # only 'fixed' supported currently
    # todo @nb: add 'semantic', 'recursive' strategy later
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    filetype: str
    chunk_index: int
    location: Union[int, str]
    chunk_text: str
    text_preview: Optional[str] = None


@dataclass
class RetrievedChunk(Chunk):
    score: Optional[float] = None
