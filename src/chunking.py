from __future__ import annotations

import uuid
from typing import Iterator, List, Tuple

from .types import (
    Chunk,
    ChunkingConfig,
    DocumentMetadata,
    ExtractedContent,
    PageContent,
)


def _build_text_and_page_map(content: ExtractedContent) -> Tuple[str, List[Tuple[int, int]]]:
    """Return concatenated text and list of (char_start, page_number).

    If input is plain text, returns a single mapping to page 1.
    """
    if isinstance(content, str):
        return content, [(0, 1)]

    full = []
    page_map: List[Tuple[int, int]] = []
    pos = 0
    for page in content:
        page_map.append((pos, page.page))
        text = page.text or ""
        full.append(text)
        # add single newline between pages to avoid accidental joins
        full.append("\n")
        pos += len(text) + 1

    return "".join(full), page_map


def _locate_page(page_map: List[Tuple[int, int]], char_index: int) -> int:
    # page_map is sorted by start position
    last_page = page_map[0][1]
    for start_pos, page_num in page_map:
        if char_index < start_pos:
            break
        last_page = page_num
    return last_page


def chunk_document(
    metadata: DocumentMetadata,
    content: ExtractedContent,
    config: ChunkingConfig,
) -> Iterator[Chunk]:
    """Yield `Chunk` objects from extracted content using fixed-size character windows.

    Behavior:
    - Uses fixed `chunk_size` and exact `chunk_overlap` by computing start positions
      at multiples of `chunk_size - chunk_overlap`.
    - Assigns `location` as the PDF page where the chunk starts (if page-aware input).
    - Deterministic: same input + config => same chunks.
    """
    if config.chunk_overlap >= config.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    full_text, page_map = _build_text_and_page_map(content)
    total = len(full_text)
    step = config.chunk_size - config.chunk_overlap

    index = 0
    start = 0
    while start < total:
        end = min(start + config.chunk_size, total)
        chunk_text = full_text[start:end]

        # Prefer not to break words when possible: trim leading/trailing partial
        # words inside the chunk text for nicer chunk contents. We keep the
        # original character-based start when computing `location` so that
        # location mapping remains deterministic and consistent with exact
        # character positions.
        orig_chunk_text = chunk_text

        # Determine trimmed boundaries within the original character window
        trimmed_start = start
        trimmed_end = end

        # Trim leading partial word: move trimmed_start forward to first whitespace
        if start > 0 and chunk_text and not chunk_text[0].isspace():
            for i, ch in enumerate(chunk_text):
                if ch.isspace():
                    trimmed_start = start + i + 1
                    break

        # Trim trailing partial word: move trimmed_end backward to last whitespace
        if end < total and chunk_text and not chunk_text[-1].isspace():
            last_ws = None
            for i in range(len(chunk_text) - 1, -1, -1):
                if chunk_text[i].isspace():
                    last_ws = i
                    break
            if last_ws is not None:
                trimmed_end = start + last_ws

        # Extract the final chunk_text slice based on trimmed indices
        chunk_text = full_text[trimmed_start:trimmed_end]

        # Remove surrounding whitespace so chunks don't start/end with spaces
        chunk_text = chunk_text.strip()
        if not chunk_text:
            # skip empty chunks that can arise from aggressive trimming
            start += step
            index += 1
            continue

        # strip leading/trailing whitespace for preview
        text_preview = chunk_text[:100]

        # Use the trimmed start position to map to the correct page
        location = _locate_page(page_map, trimmed_start)

        yield Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=metadata.document_id,
            filename=metadata.filename,
            filetype=metadata.filetype,
            chunk_index=index,
            location=location,
            chunk_text=chunk_text,
            text_preview=text_preview,
        )

        index += 1
        start += step
