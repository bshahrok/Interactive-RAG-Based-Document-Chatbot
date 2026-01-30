import math

from datetime import datetime

from src.chunking import chunk_document
from src.chunking_types import (
    ChunkingConfig,
    DocumentMetadata,
    PageContent,
)


def _make_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc-1",
        filename="test.txt",
        filetype="txt",
        original_path=None,
        ingested_at=datetime.utcnow(),
    )


def test_overlap_exact():
    text = "x" * 200
    cfg = ChunkingConfig(chunk_size=50, chunk_overlap=10)
    meta = _make_metadata()

    chunks = list(chunk_document(meta, text, cfg))
    assert len(chunks) == math.ceil(len(text) / (cfg.chunk_size - cfg.chunk_overlap))

    for a, b in zip(chunks, chunks[1:]):
        # using simple no-whitespace input ensures chunker doesn't trim
        assert a.chunk_text[-cfg.chunk_overlap :] == b.chunk_text[: cfg.chunk_overlap]


def test_chunk_size_bounds():
    # uses spaced words so trimming may occur; we assert no chunk exceeds size
    words = [f"W{i:04d}" for i in range(200)]
    text = " ".join(words)
    cfg = ChunkingConfig(chunk_size=80, chunk_overlap=20)
    meta = _make_metadata()

    for ch in chunk_document(meta, text, cfg):
        assert len(ch.chunk_text) <= cfg.chunk_size


def test_word_boundary_splitting():
    # unique words ensure we can locate chunk text inside original
    words = [f"WORD{i:05d}" for i in range(300)]
    full = " ".join(words)
    cfg = ChunkingConfig(chunk_size=100, chunk_overlap=10)
    meta = _make_metadata()

    for ch in chunk_document(meta, full, cfg):
        idx = full.find(ch.chunk_text)
        assert idx != -1, "chunk text must be substring of original"
        # chunk should not start mid-word (unless it is the very start)
        if idx > 0:
            assert full[idx - 1].isspace()
        end_idx = idx + len(ch.chunk_text) - 1
        if end_idx < len(full) - 1:
            assert full[end_idx + 1].isspace()


def test_pdf_page_association():
    # two pages with distinct markers
    p1 = PageContent(page=1, text=("P1_" * 40).strip())
    p2 = PageContent(page=2, text=("P2_" * 40).strip())
    pages = [p1, p2]
    # mimic chunker behavior that inserts a newline between pages
    full = p1.text + "\n" + p2.text + "\n"

    cfg = ChunkingConfig(chunk_size=50, chunk_overlap=5)
    meta = DocumentMetadata(
        document_id="doc-pages",
        filename="doc.pdf",
        filetype="pdf",
        original_path=None,
        ingested_at=datetime.utcnow(),
    )

    # compute page start positions same as chunker
    page_starts = [0]
    pos = 0
    for p in pages:
        page_starts.append(pos)
        pos += len(p.text) + 1

    for ch in chunk_document(meta, pages, cfg):
        # locate chunk text in full
        idx = full.find(ch.chunk_text)
        assert idx != -1

        # determine expected page by seeing where idx falls
        expected_page = 1
        if idx >= len(p1.text) + 1:
            expected_page = 2

        assert ch.location == expected_page
