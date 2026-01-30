import os
import sys
from datetime import datetime

# make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.chunking import chunk_document
from src.types import ChunkingConfig, DocumentMetadata, PageContent


def debug_word():
    words = [f"WORD{i:05d}" for i in range(300)]
    full = " ".join(words)
    cfg = ChunkingConfig(chunk_size=100, chunk_overlap=10)
    meta = DocumentMetadata(
        document_id="doc-1",
        filename="test.txt",
        filetype="txt",
        original_path=None,
        ingested_at=datetime.utcnow(),
    )

    for ch in chunk_document(meta, full, cfg):
        idx = full.find(ch.chunk_text)
        print("CHUNK_INDEX", ch.chunk_index)
        print("chunk_text repr:", repr(ch.chunk_text[:80]))
        print("found at idx:", idx)
        print("preceding char:", repr(full[idx-1] if idx>0 else None))
        end_idx = idx + len(ch.chunk_text) - 1
        print("following char:", repr(full[end_idx+1] if end_idx < len(full)-1 else None))
        print("len chunk_text", len(ch.chunk_text))
        print("---")


def debug_pdf():
    p1 = PageContent(page=1, text=("P1_" * 40).strip())
    p2 = PageContent(page=2, text=("P2_" * 40).strip())
    pages = [p1, p2]
    full = p1.text + "\n" + p2.text + "\n"

    cfg = ChunkingConfig(chunk_size=50, chunk_overlap=5)
    meta = DocumentMetadata(
        document_id="doc-pages",
        filename="doc.pdf",
        filetype="pdf",
        original_path=None,
        ingested_at=datetime.utcnow(),
    )

    for ch in chunk_document(meta, pages, cfg):
        idx = full.find(ch.chunk_text)
        print("CHUNK_INDEX", ch.chunk_index, "location", ch.location)
        print("chunk_text repr:", repr(ch.chunk_text))
        print("found at idx:", idx)
        print("---")


if __name__ == '__main__':
    print('--- WORD DEBUG ---')
    debug_word()
    print('\n--- PDF DEBUG ---')
    debug_pdf()
