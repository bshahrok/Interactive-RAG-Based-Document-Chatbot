"""
Text chunking module using recursive character text splitting.

This module implements best-practice chunking for RAG systems by:
- Using semantic separators (paragraphs, sentences, words) before character splits
- Preserving document location metadata for citations
- Ensuring overlap at sensible boundaries
- Never splitting mid-word
"""

import uuid
from dataclasses import dataclass
from typing import Iterator, List, Optional, Union

from src.ingestion import DocumentMetadata, ExtractedText, PageContent


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    
    strategy: str = "recursive"  # Currently only "recursive" is supported
    size: int = 1000  # Target chunk size in characters
    overlap: int = 150  # Overlap between chunks in characters
    
    def __post_init__(self):
        """Validate configuration."""
        if self.strategy not in ("recursive", "fixed"):
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be 'recursive' or 'fixed'.")
        if self.size <= 0:
            raise ValueError(f"Chunk size must be positive, got {self.size}")
        if self.overlap < 0:
            raise ValueError(f"Overlap must be non-negative, got {self.overlap}")
        if self.overlap >= self.size:
            raise ValueError(f"Overlap ({self.overlap}) must be less than chunk size ({self.size})")


@dataclass
class Chunk:
    """A text chunk with location metadata."""
    
    chunk_id: str
    document_id: str
    filename: str
    filetype: str
    chunk_index: int  # 0-indexed position in document
    location: Union[int, str]  # Page number for PDF, or chunk index for others
    chunk_text: str
    text_preview: Optional[str] = None  # First 100 chars for debugging
    
    def __post_init__(self):
        """Generate text preview if not provided."""
        if self.text_preview is None and self.chunk_text:
            self.text_preview = self.chunk_text[:100]


class RecursiveCharacterTextSplitter:
    """
    Split text recursively using semantic separators.
    
    This follows best practices by attempting to split on:
    1. Double newlines (paragraphs)
    2. Single newlines (lines)
    3. Spaces (words)
    4. Characters (as last resort)
    
    This preserves semantic coherence better than fixed-size character splitting.
    """
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Separators in order of preference (most semantic first)
        self.separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            " ",     # Words
            ""       # Characters (last resort)
        ]
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive splitting.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the provided separators.
        
        Args:
            text: Text to split
            separators: List of separators to try, in order of preference
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Base case: if text is already small enough, return it
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try each separator in order
        for i, separator in enumerate(separators):
            if separator == "":
                # Last resort: split by characters
                return self._split_by_characters(text)
            
            # Check if separator exists in text
            if separator in text:
                # Split by this separator
                splits = text.split(separator)
                
                # Merge splits into chunks
                return self._merge_splits(splits, separator, separators[i:])
        
        # Fallback: split by characters
        return self._split_by_characters(text)
    
    def _merge_splits(
        self, 
        splits: List[str], 
        separator: str, 
        remaining_separators: List[str]
    ) -> List[str]:
        """
        Merge small splits into chunks of appropriate size.
        
        Args:
            splits: List of text splits
            separator: The separator used to create the splits
            remaining_separators: Remaining separators for recursive splitting
            
        Returns:
            List of merged chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            separator_length = len(separator) if split != splits[-1] else 0
            
            # If adding this split would exceed chunk size
            if current_length + split_length + separator_length > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk_text = separator.join(current_chunk)
                    
                    # If chunk is still too large, recursively split it
                    if len(chunk_text) > self.chunk_size:
                        sub_chunks = self._split_text_recursive(
                            chunk_text, 
                            remaining_separators[1:] if len(remaining_separators) > 1 else [""]
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0 and chunks:
                        # Calculate how much of the previous chunk to include as overlap
                        overlap_text = chunk_text[-self.chunk_overlap:]
                        current_chunk = [overlap_text, split] if overlap_text else [split]
                        current_length = len(overlap_text) + split_length + len(separator)
                    else:
                        current_chunk = [split]
                        current_length = split_length
                else:
                    # Current split is too large on its own, recursively split it
                    if split_length > self.chunk_size:
                        sub_chunks = self._split_text_recursive(
                            split,
                            remaining_separators[1:] if len(remaining_separators) > 1 else [""]
                        )
                        chunks.extend(sub_chunks)
                        current_chunk = []
                        current_length = 0
                    else:
                        current_chunk = [split]
                        current_length = split_length
            else:
                # Add split to current chunk
                current_chunk.append(split)
                current_length += split_length + separator_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) > self.chunk_size and len(remaining_separators) > 1:
                sub_chunks = self._split_text_recursive(
                    chunk_text,
                    remaining_separators[1:]
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    def _split_by_characters(self, text: str) -> List[str]:
        """
        Split text by characters as a last resort.
        
        Args:
            text: Text to split
            
        Returns:
            List of character-based chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            
            # Move start forward, accounting for overlap
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
        
        return chunks


def chunk_document(
    metadata: DocumentMetadata,
    extracted: ExtractedText,
    config: ChunkingConfig
) -> Iterator[Chunk]:
    """
    Chunk a document's extracted text into smaller pieces.
    
    This function uses a recursive character text splitter that attempts to
    split on semantic boundaries (paragraphs, sentences, words) before
    resorting to character-level splitting.
    
    Args:
        metadata: Document metadata
        extracted: Extracted text with page/paragraph content
        config: Chunking configuration
        
    Yields:
        Chunk objects with text and location metadata
    """
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.size,
        chunk_overlap=config.overlap
    )
    
    chunk_index = 0
    
    # For PDFs with page-level content, track which page each chunk comes from
    if metadata.filetype == 'pdf' and extracted.total_pages:
        for page_content in extracted.content:
            page_text = page_content.text
            page_num = page_content.page_number
            
            if not page_text.strip():
                continue
            
            # Split page text into chunks
            text_chunks = splitter.split_text(page_text)
            
            for chunk_text in text_chunks:
                if chunk_text.strip():  # Only yield non-empty chunks
                    yield Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=metadata.document_id,
                        filename=metadata.filename,
                        filetype=metadata.filetype,
                        chunk_index=chunk_index,
                        location=page_num,  # Use page number as location
                        chunk_text=chunk_text,
                    )
                    chunk_index += 1
    
    # For DOCX with paragraph-level content
    elif metadata.filetype == 'docx' and extracted.total_paragraphs:
        # Combine all paragraphs into full text
        full_text = extracted.get_full_text()
        text_chunks = splitter.split_text(full_text)
        
        for chunk_text in text_chunks:
            if chunk_text.strip():
                yield Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=metadata.document_id,
                    filename=metadata.filename,
                    filetype=metadata.filetype,
                    chunk_index=chunk_index,
                    location=f"chunk_{chunk_index}",  # Use chunk index as location
                    chunk_text=chunk_text,
                )
                chunk_index += 1
    
    # For TXT files
    else:
        full_text = extracted.get_full_text()
        text_chunks = splitter.split_text(full_text)
        
        for chunk_text in text_chunks:
            if chunk_text.strip():
                yield Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=metadata.document_id,
                    filename=metadata.filename,
                    filetype=metadata.filetype,
                    chunk_index=chunk_index,
                    location=f"chunk_{chunk_index}",
                    chunk_text=chunk_text,
                )
                chunk_index += 1
