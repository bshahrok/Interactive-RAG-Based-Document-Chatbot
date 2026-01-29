# RAG System Project Specification

** Modular RAG Implementation**

This document contains Issue bodies for building a local RAG (Retrieval-Augmented Generation) system. Each issue is scoped, AI-codable, with explicit inputs/outputs, constraints, and Definition of Done criteria.

---

## Table of Contents

1.  [Implementation Order Recommendation](#implementation-order-recommendation)
2. [Issue 1 — Ingestion: PDF/DOCX/TXT Text Extraction + Metadata](#issue-1--ingestion-pdfdocxtxt-text-extraction--metadata)
3. [Issue 2 — Chunking: Fixed-Size + Overlap + Location Mapping](#issue-2--chunking-fixed-size--overlap--location-mapping)
4. [Issue 3 — Vector Store: Chroma Persistent Index + Upsert/Delete](#issue-3--vector-store-chroma-persistent-index--upsertdelete)
5. [Issue 4 — Embeddings: Local Embedding Model Wrapper](#issue-4--embeddings-local-embedding-model-wrapper)
6. [Issue 5 — Indexing Pipeline: Ingest → Chunk → Embed → Store](#issue-5--indexing-pipeline-ingest--chunk--embed--store-incremental)
7. [Issue 6 — Retrieval: Top-k + Min Score Threshold + Filtering](#issue-6--retrieval-top-k--min-score-threshold--filtering)
8. [Issue 7 — LLM Backends: Local + Optional API Adapter](#issue-7--llm-backends-local-ollamallama-cpp--optional-api-adapter)
9. [Issue 8 — Prompting + RAG Orchestrator](#issue-8--prompting--rag-orchestrator-concise--expand--clarify)
10. [Issue 9 — Config System: YAML Schema + Runtime Overrides](#issue-9--config-system-yaml-schema--runtime-overrides--validation)
11. [Issue 10 — Streamlit UI: Upload + Chat + Evidence + Settings](#issue-10--streamlit-ui-upload--chat--evidence-expanders--settings-panel)
12. [Issue 11 — Document Management: List + Delete](#issue-11--document-management-list--delete-document-from-index)
13. [Issue 12 — Tests + CI: Acceptance Tests and GitHub Actions](#issue-12--tests--ci-acceptance-tests-and-github-actions)
14. [Issue 13 — Documentation: README + Setup Guide](#issue-13--documentation-readme--setup-guide--architecture-docs)

---
## Implementation Order Recommendation

### Phase 1: Core Infrastructure (Issues 1-4)
Build foundational components that don't depend on each other:
- Issue 1: Ingestion
- Issue 2: Chunking
- Issue 3: Vector Store
- Issue 4: Embeddings

### Phase 2: Pipeline Integration (Issues 5-6, 9)
Connect components into working pipelines:
- Issue 9: Config System (needed for all integration)
- Issue 5: Indexing Pipeline
- Issue 6: Retrieval

### Phase 3: Intelligence Layer (Issues 7-8)
Add LLM capabilities:
- Issue 7: LLM Backends
- Issue 8: RAG Orchestrator

### Phase 4: User Interface (Issues 10-11)
Build user-facing features:
- Issue 11: Document Management
- Issue 10: Streamlit UI

### Phase 5: Quality & Documentation (Issues 12-13)
Finalize testing and documentation:
- Issue 12: Tests + CI
- Issue 13: Documentation

---

## Issue 1 — Ingestion: PDF/DOCX/TXT Text Extraction + Metadata

**Title:** Ingestion module: extract text from PDF/DOCX/TXT with metadata

**Labels:** `module:ingestion`, `priority:high`, `complexity:medium`

**Body:**

**Spec references:** Sections 3, 5, 6.1

### Goal
Implement document ingestion that accepts locally-uploaded **PDF, DOCX, TXT** files and outputs normalized extracted text + document metadata.

### Requirements
* **Supported types:** PDF, DOCX, TXT only. Reject unsupported formats with clear error messages.
* **Produce `DocumentMetadata` dataclass:**
  * `document_id: str` (UUID4 format)
  * `filename: str`
  * `filetype: Literal["pdf", "docx", "txt"]`
  * `original_path: Optional[str]`
  * `ingested_at: datetime` (UTC timestamp)
  * `page_count: Optional[int]` (for PDFs)
* **Extraction output must preserve location:**
  * **PDF:** Page-aware extraction with page number markers (e.g., tuple of `(page_num, text)` or structured `PageContent` objects)
  * **DOCX:** Paragraph-aware extraction preferred (preserve paragraph boundaries)
  * **TXT:** Raw read with line number tracking optional
* **Error handling:** Return structured errors for corrupted files, empty documents, or unsupported formats

### Constraints
* **Local-only:** No external services or network calls
* **Memory efficient:** Stream/iterative extraction for large files (>10MB)
* **Dependencies:** Use `pypdf` or `PyPDF2` for PDF, `python-docx` for DOCX

### Implementation Notes
* Return type: `Tuple[DocumentMetadata, Union[str, List[PageContent]]]`
* For PDFs, prefer returning `List[PageContent]` where `PageContent = {"page": int, "text": str}`
* Handle edge cases: password-protected PDFs (reject), scanned PDFs (extract as-is, note OCR not required)

### Deliverables
* `src/ingestion.py` with public API:
  * `ingest_file(file_path: Union[str, Path], filename: Optional[str] = None) -> Tuple[DocumentMetadata, ExtractedContent]`
  * `ingest_bytes(file_bytes: bytes, filename: str, filetype: str) -> Tuple[DocumentMetadata, ExtractedContent]`
* Type definitions in `src/types.py` (DocumentMetadata, ExtractedContent, PageContent)
* Unit tests in `tests/test_ingestion.py`:
  * Test fixtures for each file type (< 1MB each)
  * Test empty file rejection
  * Test page extraction for PDF
  * Test metadata field population

### Definition of Done
- [ ] Can ingest sample PDF/DOCX/TXT and produce complete metadata + extracted text
- [ ] PDF extraction includes page numbers for each text segment
- [ ] All unit tests pass with >90% code coverage for ingestion module
- [ ] Type hints present and pass `mypy --strict`
- [ ] Error messages are user-friendly and actionable

**Estimated effort:** 6-8 hours

---

## Issue 2 — Chunking: Fixed-Size + Overlap + Location Mapping

**Title:** Chunking module: fixed-size chunks with overlap and location metadata

**Labels:** `module:chunking`, `priority:high`, `complexity:medium`

**Body:**

**Spec references:** Sections 5.2, 6.2

### Goal
Implement chunking that converts extracted text into fixed-size chunks with configurable overlap, preserving document location metadata for citations.

### Requirements
* **Configurable parameters** (from config system):
  * `chunking.strategy: str = "fixed"` (implement `fixed` now; design for extensibility)
  * `chunking.chunk_size: int` (characters, default 500)
  * `chunking.chunk_overlap: int` (characters, default 50)
* **Output `Chunk` dataclass with:**
  * `chunk_id: str` (UUID4)
  * `document_id: str` (from parent document)
  * `filename: str`
  * `filetype: str`
  * `chunk_index: int` (0-indexed position in document)
  * `location: Union[int, str]` (PDF page number as int, or "chunk_{index}" for others)
  * `chunk_text: str` (the actual text content)
  * `text_preview: Optional[str]` (first 100 chars for debugging/display)
* **Location mapping rules:**
  * **PDF:** If chunk spans multiple pages, use the page where chunk *starts*
  * **DOCX/TXT:** Use `chunk_index` as location (or optionally paragraph number for DOCX)
* **Chunking algorithm:**
  * Split on word boundaries when possible (don't break mid-word)
  * Overlap must be exact: if `chunk_size=500, overlap=50`, next chunk starts at position `450`
  * Handle edge cases: final chunk may be smaller than `chunk_size`

### Constraints
* **Deterministic:** Same input + config must produce identical chunks (stable for re-indexing)
* **Memory efficient:** Process iteratively, don't materialize all chunks at once for large documents
* **Character-based:** Use character counts, not tokens (token-based chunking is non-goal)

### Implementation Notes
* Consider using `langchain.text_splitter.RecursiveCharacterTextSplitter` as reference, but implement from scratch for full control
* For PDFs with `List[PageContent]` input, track page transitions

### Deliverables
* `src/chunking.py` with:
  * `chunk_document(metadata: DocumentMetadata, content: ExtractedContent, config: ChunkingConfig) -> Iterator[Chunk]`
  * `ChunkingConfig` dataclass
* `src/types.py` updated with `Chunk` dataclass
* Unit tests in `tests/test_chunking.py`:
  * Test overlap behavior (verify chunk N and N+1 share exactly `overlap` characters)
  * Test chunk size bounds (no chunk exceeds `chunk_size` except possibly last)
  * Test PDF page association with multi-page document fixture
  * Test word-boundary splitting
  * Test determinism (same input → same output)

### Definition of Done
- [ ] Given extracted content, produces iterator of chunks matching spec schema
- [ ] Chunk overlap verified correct (exactly `overlap` characters shared between consecutive chunks)
- [ ] PDF page numbers correctly assigned
- [ ] All tests pass with >85% coverage
- [ ] No memory leaks with 100-page test document

**Estimated effort:** 8-10 hours

---

## Issue 3 — Vector Store: Chroma Persistent Index + Upsert/Delete

**Title:** Vector store module: Chroma persistent index with metadata

**Labels:** `module:vector-store`, `priority:high`, `complexity:high`

**Body:**

**Spec references:** Sections 4, 6.4, 11

### Goal
Create a Chroma-backed vector store wrapper providing persistent on-disk storage with full CRUD operations for document chunks.

### Requirements
* **Persistence:** 
  * Store index at configurable `persist_path` (from config: `paths.persist_dir`)
  * Survive application restarts (re-init loads existing data)
* **Operations:**
  * `init_store()`: Create or load existing collection
  * `upsert_chunks()`: Add or update chunk embeddings + metadata (idempotent)
  * `query()`: Retrieve top-k most similar chunks with scores
  * `delete_document()`: Remove all chunks for a given `document_id`
  * `list_documents()`: Return metadata for all indexed documents (for UI)
  * `get_stats()`: Return index statistics (total chunks, total documents)
* **Storage schema:**
  * **Embeddings:** Vector embeddings (dimension determined by embedding model)
  * **Documents:** Store `chunk_text` as Chroma's document field
  * **Metadata:** Store all `Chunk` fields as metadata dict:
    * `chunk_id`, `document_id`, `filename`, `filetype`, `chunk_index`, `location`, `text_preview`
* **Collection management:**
  * Default collection name: `"rag_chunks"` (configurable)
  * Single collection per index for simplicity

### Constraints
* **Chroma version:** Use `chromadb >= 0.4.0`
* **Distance metric:** Cosine similarity (Chroma default)
* **Batching:** Support batch upsert (list of chunks at once) for performance
* **Thread safety:** Not required for MVP (single-user Streamlit app)

### Implementation Notes
* Use `chromadb.PersistentClient` (not deprecated `Client`)
* Handle Chroma migration warnings gracefully
* Consider lazy initialization (don't connect until first operation)

### Deliverables
* `src/vector_store.py` with class `ChromaVectorStore`:
  * `__init__(persist_path: str, collection_name: str = "rag_chunks")`
  * `upsert_chunks(chunks: List[Chunk], embeddings: np.ndarray) -> None`
  * `query(query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievedChunk]`
  * `delete_document(document_id: str) -> int` (returns number of chunks deleted)
  * `list_documents() -> List[DocumentMetadata]`
  * `get_stats() -> Dict[str, int]`
* `src/types.py` updated with `RetrievedChunk` dataclass (extends `Chunk` with `score: float`)
* Tests in `tests/test_vector_store.py`:
  * Test persistence across re-initialization
  * Test upsert overwrites existing chunk_id
  * Test delete removes all document chunks
  * Test query returns correct top-k with scores
  * Test empty query behavior

### Definition of Done
- [ ] Index persists to disk and survives restart (test by re-init)
- [ ] Query returns `chunk_text` + all metadata + similarity scores
- [ ] `delete_document()` removes all chunks and doesn't affect other documents
- [ ] `list_documents()` returns unique documents from index
- [ ] All tests pass with >80% coverage
- [ ] Performance: Can upsert 1000 chunks in <5 seconds on standard hardware

**Estimated effort:** 10-12 hours

---

## Issue 4 — Embeddings: Local Embedding Model Wrapper

**Title:** Embeddings module: local model loader + batch embedding API

**Labels:** `module:embeddings`, `priority:high`, `complexity:medium`

**Body:**

**Spec references:** Sections 4, 6.3, 11

### Goal
Implement embedding generation using a local transformer model with batching support and configurable model selection.

### Requirements
* **API:**
  * `embed_texts(texts: List[str]) -> np.ndarray` (batch embedding for chunks)
  * `embed_query(query: str) -> np.ndarray` (single query embedding)
  * Both return numpy arrays of shape `(n, embedding_dim)`
* **Configuration** (from config system):
  * `embeddings.model_name: str` (default: `"sentence-transformers/all-MiniLM-L6-v2"`)
  * `embeddings.device: str` (default: `"cpu"`, support `"cuda"`, `"mps"`)
  * `embeddings.batch_size: int` (default: 32, for batching large lists)
* **Model compatibility:**
  * Must use sentence-transformers compatible models
  * Default model embedding dimension: 384 (verify Chroma compatibility)
  * Cache model downloads in standard HuggingFace cache location
* **Performance:**
  * Batch processing for efficiency (don't call model 1000 times for 1000 chunks)
  * Progress indication for large batches (>100 texts) optional but nice

### Constraints
* **Local-only:** No API calls, fully offline after model download
* **Memory management:** 
  * Don't load multiple models simultaneously
  * Clear GPU memory if device switching occurs
* **Dependencies:** Use `sentence-transformers` library

### Implementation Notes
* Consider lazy loading model (don't instantiate until first embed call)
* Normalize embeddings if needed (some models require this for cosine similarity)
* Handle empty string input gracefully

### Deliverables
* `src/embeddings.py` with class `LocalEmbeddings`:
  * `__init__(model_name: str, device: str = "cpu")`
  * `embed_texts(texts: List[str], show_progress: bool = False) -> np.ndarray`
  * `embed_query(query: str) -> np.ndarray`
  * `get_embedding_dim() -> int`
* Tests in `tests/test_embeddings.py`:
  * Test consistent output shapes (n texts → n x dim array)
  * Test batch processing (verify batching doesn't change results)
  * Test empty input handling
  * Test determinism (same input → same embedding)
  * Test query vs texts consistency (query path should match batch path for single item)

### Definition of Done
- [ ] Embeddings computed for both chunks and queries with correct dimensions
- [ ] Batch processing works and is more efficient than sequential (benchmark >2x speedup)
- [ ] No external API required; runs fully offline
- [ ] Model downloads automatically on first run with clear progress
- [ ] All tests pass with >85% coverage
- [ ] CPU-only tests pass in CI (don't require GPU)

**Estimated effort:** 6-8 hours

---

## Issue 5 — Indexing Pipeline: Ingest → Chunk → Embed → Store (Incremental)

**Title:** Indexing pipeline: incremental document indexing with atomic operations

**Labels:** `module:indexing`, `priority:high`, `complexity:high`

**Body:**

**Spec references:** Sections 6, 6.4, 13

### Goal
Implement end-to-end indexing pipeline orchestrating ingestion, chunking, embedding, and storage with incremental add capability and error recovery.

### Requirements
* **Pipeline flow:**
  1. Ingest file → `DocumentMetadata` + `ExtractedContent`
  2. Chunk content → `List[Chunk]`
  3. Embed chunks → `np.ndarray`
  4. Upsert to vector store
* **Incremental indexing:**
  * Adding new documents does NOT rebuild existing index
  * Re-indexing same file (by filename) should update, not duplicate
  * `document_id` must be deterministic for same filename OR allow explicit replace
* **Metadata preservation:**
  * All chunk metadata must flow through to vector store
  * Sufficient metadata for citations: filename, location, filetype
* **Error handling:**
  * Per-file error isolation (one file failure doesn't stop batch)
  * Return detailed `IndexingReport`:
    * `successful_documents: List[DocumentMetadata]`
    * `failed_documents: List[Tuple[str, Exception]]` (filename + error)
    * `total_chunks_added: int`
    * `indexing_duration: float` (seconds)
* **Batch support:**
  * `index_documents(file_paths: List[Path]) -> IndexingReport`
  * Progress callback optional for UI integration

### Constraints
* **Atomicity:** If embedding fails, don't partially index document
* **Idempotency:** Re-running on same files should be safe (update, not error)
* **Performance:** Embed in batches across documents (not one doc at a time)

### Implementation Notes
* Consider transaction-like behavior: collect all chunks for a document before upserting
* For duplicate filename handling, decide strategy:
  - Option A: Delete old document first, then index new
  - Option B: Generate deterministic `document_id` from content hash
  - **Recommended:** Option A for simplicity (explicit replace)

### Deliverables
* `src/indexing.py` with:
  * `IndexingPipeline` class (or function-based API)
  * `index_document(file_path: Path, replace_existing: bool = True) -> IndexingReport`
  * `index_documents(file_paths: List[Path], replace_existing: bool = True) -> IndexingReport`
* `src/types.py` updated with `IndexingReport` dataclass
* Integration tests in `tests/test_indexing.py`:
  * Test full pipeline with multi-file batch
  * Test incremental add (index 2 files, then index 2 more, verify 4 total)
  * Test replace mode (re-index same file, verify only latest version exists)
  * Test error isolation (1 corrupted file doesn't stop others)
  * Test chunk count accuracy

### Definition of Done
- [ ] End-to-end indexing works for all supported file types
- [ ] Incremental add verified: existing documents remain after adding new ones
- [ ] Replace mode verified: re-indexing updates without duplicating
- [ ] Error handling works: corrupted file returns error but doesn't crash
- [ ] `IndexingReport` provides accurate statistics
- [ ] Integration tests pass with >75% coverage
- [ ] Performance benchmark: Can index 10 x 10-page PDFs in <60 seconds on CPU

**Estimated effort:** 10-12 hours

**Dependencies:** Issues #1, #2, #3, #4

---

## Issue 6 — Retrieval: Top-k + Min Score Threshold + Filtering

**Title:** Retrieval module: semantic search with score filtering and metadata

**Labels:** `module:retrieval`, `priority:high`, `complexity:medium`

**Body:**

**Spec references:** Sections 7, 8.3

### Goal
Implement retrieval orchestrating query embedding, vector search, and score-based filtering to support both high-confidence and low-evidence modes.

### Requirements
* **Configuration** (from config system):
  * `retrieval.top_k: int` (default: 5)
  * `retrieval.min_score: float` (default: 0.3, range 0.0-1.0)
  * `retrieval.include_metadata: bool` (default: True)
* **Retrieval flow:**
  1. Embed user query string
  2. Query vector store for top-k chunks
  3. Filter results by `min_score` threshold
  4. Return ranked results with full metadata
* **Output schema (`RetrievalResult`):**
  * `chunks: List[RetrievedChunk]` (may be empty if all below threshold)
  * `is_low_evidence: bool` (True if no chunks above threshold)
  * `query_embedding: Optional[np.ndarray]` (for debugging, optional)
* **Low-evidence detection:**
  * If all retrieved chunks have score < `min_score`, set `is_low_evidence=True`
  * Return empty `chunks` list in this case (don't pass low-confidence chunks to LLM)

### Constraints
* **Score semantics:** Assumes cosine similarity (0.0-1.0 range, higher is better)
* **No re-ranking:** Simple top-k retrieval only (advanced re-ranking is non-goal)

### Implementation Notes
* Score threshold should be tunable per-query if needed (advanced feature)
* Consider logging retrieval stats for analysis (query, top scores, etc.)

### Deliverables
* `src/retrieval.py` with:
  * `retrieve(query: str, top_k: Optional[int] = None, min_score: Optional[float] = None) -> RetrievalResult`
  * `RetrievalConfig` dataclass
* `src/types.py` updated with `RetrievalResult` dataclass
* Tests in `tests/test_retrieval.py`:
  * Test returns exactly top_k chunks (when sufficient high-score chunks exist)
  * Test min_score filtering (low-score chunks excluded)
  * Test low-evidence mode triggers correctly
  * Test empty index behavior
  * Test metadata preservation

### Definition of Done
- [ ] Retrieval returns structured `RetrievalResult` with chunks and scores
- [ ] Score threshold filtering works (chunks below threshold excluded)
- [ ] Low-evidence mode detected when no chunks meet threshold
- [ ] All metadata preserved from vector store through to output
- [ ] All tests pass with >85% coverage
- [ ] Performance: Retrieval completes in <500ms for typical query

**Estimated effort:** 6-8 hours

**Dependencies:** Issues #3, #4

---

## Issue 7 — LLM Backends: Local (Ollama/llama cpp) + Optional API Adapter

**Title:** LLM module: local backend + optional API backend interface

**Labels:** `module:llm`, `priority:high`, `complexity:high`

**Body:**

**Spec references:** Sections 4, 8, 11

### Goal
Implement LLM interface supporting local inference (Ollama or llama.cpp) as primary backend with optional API backend adapter for flexibility.

### Requirements
* **Unified interface:**
  * `generate(prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str`
  * `generate_stream(prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]` (optional, for UI streaming)
* **Configuration** (from config system):
  * `llm.backend: Literal["local", "api"]` (default: `"local"`)
  * `llm.local_runtime: Literal["ollama", "llama_cpp"]` (default: `"ollama"`)
  * `llm.model: str` (default: `"llama3.2:3b"` for Ollama, or path for llama.cpp)
  * `llm.temperature: float` (default: 0.1, range 0.0-1.0)
  * `llm.max_tokens: int` (default: 512)
  * `llm.api_base_url: Optional[str]` (for API backend, e.g., OpenAI-compatible endpoint)
  * `llm.api_key: Optional[str]` (for API backend, loaded from env var `LLM_API_KEY`)
* **Local backend implementation:**
  * **Ollama:** Use Ollama HTTP API (localhost:11434 by default)
  * **llama.cpp:** Use llama-cpp-python bindings
  * Verify model availability on init (fail fast with clear error if model missing)
* **API backend implementation:**
  * Support OpenAI-compatible API format
  * Handle authentication, rate limits gracefully
  * Optional: Support other providers (Anthropic, etc.) with adapter pattern
* **Error handling:**
  * Network errors for Ollama
  * Model loading errors for llama.cpp
  * API authentication/quota errors
  * Timeout handling (configurable timeout, default 30s)

### Constraints
* **Default local-only:** No API keys required for default configuration
* **No hardcoded secrets:** API keys must come from environment variables or config file (not in code)
* **Graceful degradation:** If local runtime unavailable, provide clear setup instructions

### Implementation Notes
* Consider factory pattern: `LLMBackend.create(config) -> Union[OllamaBackend, LlamaCppBackend, APIBackend]`
* For Ollama, use `/api/generate` endpoint
* For streaming, use `/api/generate` with `stream: true`

### Deliverables
* `src/llm.py` with:
  * Abstract base `LLMBackend` class
  * `OllamaBackend` implementation
  * `LlamaCppBackend` implementation
  * `APIBackend` implementation (can be minimal/stubbed)
  * Factory function `create_llm(config: LLMConfig) -> LLMBackend`
* `src/types.py` updated with `LLMConfig` dataclass
* Tests in `tests/test_llm.py`:
  * Test Ollama backend with mock server (or skip if Ollama not installed)
  * Test llama.cpp backend with tiny model (or skip)
  * Test API backend with mock HTTP responses
  * Test error handling (model not found, timeout, etc.)
  * Test parameter passing (temperature, max_tokens)

### Definition of Done
- [ ] Local backend (Ollama or llama.cpp) works and generates text
- [ ] API backend interface exists and can be configured (even if stubbed)
- [ ] Configuration properly selects backend based on `llm.backend`
- [ ] Clear error messages when model/runtime unavailable
- [ ] No secrets hardcoded; all sensitive config from env vars
- [ ] Tests pass (with appropriate skip decorators for unavailable runtimes)
- [ ] README includes setup instructions for Ollama and llama.cpp

**Estimated effort:** 10-14 hours

---

## Issue 8 — Prompting + RAG Orchestrator: Concise + Expand + Clarify

**Title:** RAG orchestrator: grounded prompting with concise/expanded + clarifying mode

**Labels:** `module:rag`, `priority:critical`, `complexity:high`

**Body:**

**Spec references:** Sections 8, 8.1–8.3, 9

### Goal
Implement RAG response generation orchestrating retrieval and LLM prompting with strict grounding, multiple response modes, and citation generation.

### Requirements
* **Response modes:**
  * **Concise** (default): Brief, direct answer in 2-4 sentences
  * **Expanded**: Detailed explanation with examples, 1-2 paragraphs
  * **Clarifying**: Ask user a question when evidence insufficient
* **Grounding rules (CRITICAL):**
  * **MUST answer ONLY from retrieved chunks**
  * **MUST NOT use LLM's prior knowledge**
  * **MUST cite specific chunks for all claims**
  * If retrieved chunks don't contain answer, MUST ask clarifying question (don't make up answer)
* **Citation format:**
  * Inline references: `"According to [filename, page X]..."`
  * Output includes `citations: List[Citation]` with:
    * `chunk_id: str`
    * `filename: str`
    * `location: Union[int, str]`
    * `text_snippet: str` (relevant excerpt from chunk, max 200 chars)
* **Mode selection logic:**
  * Use `retrieval.is_low_evidence` to trigger clarifying mode
  * User can explicitly request expanded mode
  * Default to concise mode
* **Prompt engineering:**
  * System prompt enforces grounding constraint strongly
  * Include retrieved chunks in context with clear formatting
  * Use few-shot examples to teach citation format

### Constraints
* **Token budget:** Manage context window (retrieval chunks + prompt + response < model's context limit)
* **Determinism:** Use low temperature (0.1) for consistency
* **Evidence preservation:** Must return retrieved chunks to UI for evidence panel display

### Implementation Notes
* Prompt template structure:
  ```
  System: You are a precise Q&A assistant. Answer ONLY using the provided context chunks. Do not use prior knowledge. Cite sources using [filename, location] format.
  
  Context chunks:
  [Chunk 1: filename.pdf, page 5]
  {chunk_text_1}
  
  [Chunk 2: report.docx, chunk 3]
  {chunk_text_2}
  
  User question: {query}
  
  Answer (concise, with citations):
  ```
* Consider prompt variants for each mode (concise/expanded/clarifying)
* Extract citations from LLM response using regex or structured output

### Deliverables
* `src/rag.py` with:
  * `RAGOrchestrator` class
  * `generate_response(query: str, mode: Literal["concise", "expanded", "clarifying"] = "concise") -> RAGResponse`
  * Helper: `extract_citations(response_text: str, retrieved_chunks: List[RetrievedChunk]) -> List[Citation]`
* Prompt templates in `src/prompts.py`:
  * `CONCISE_PROMPT_TEMPLATE`
  * `EXPANDED_PROMPT_TEMPLATE`
  * `CLARIFYING_PROMPT_TEMPLATE`
* `src/types.py` updated with:
  * `RAGResponse` dataclass (answer_text, citations, mode_used, retrieved_chunks)
  * `Citation` dataclass
* Tests in `tests/test_rag.py`:
  * Test concise mode produces short answer with citations
  * Test expanded mode produces longer answer
  * Test clarifying mode triggers on low evidence
  * Test grounding: verify answer doesn't include facts not in chunks (manual review test)
  * Test citation extraction accuracy

### Definition of Done
- [ ] Concise mode produces 2-4 sentence answers with inline citations
- [ ] Expanded mode produces 1-2 paragraph detailed answers with citations
- [ ] Clarifying mode asks relevant question when `is_low_evidence=True`
- [ ] All answers verifiably grounded in retrieved chunks (manual spot-check + test)
- [ ] Citations include correct filename, location, and text snippet
- [ ] Retrieved chunks passed through to response for evidence display
- [ ] Integration tests pass with >70% coverage
- [ ] Manual test with sample queries produces high-quality answers

**Estimated effort:** 14-16 hours

**Dependencies:** Issues #6, #7

---

## Issue 9 — Config System: YAML Schema + Runtime Overrides + Validation

**Title:** Config system: YAML defaults + runtime overrides with schema validation

**Labels:** `module:config`, `priority:high`, `complexity:medium`

**Body:**

**Spec references:** Section 11

### Goal
Implement configuration system providing YAML-based defaults with runtime override capability, type safety, and validation.

### Requirements
* **Config file:** `config/default.yaml` with all hyperparameters:
  * `chunking.*` (strategy, chunk_size, chunk_overlap)
  * `retrieval.*` (top_k, min_score)
  * `embeddings.*` (model_name, device, batch_size)
  * `llm.*` (backend, local_runtime, model, temperature, max_tokens, api_base_url, api_key)
  * `paths.*` (persist_dir, upload_dir)
  * `grounding_strict: bool` (enforce grounding in prompts, default: true)
  * `ui.*` (optional, e.g., theme, page_title)
* **Schema validation:**
  * Type checking (int, float, str, bool, Literal types)
  * Range validation (e.g., temperature 0.0-1.0, min_score 0.0-1.0)
  * Required vs optional fields
  * Default values for optional fields
* **Runtime overrides:**
  * Environment variables: `RAG_LLM_MODEL`, `RAG_PERSIST_DIR`, etc.
  * Programmatic overrides: `config.set("llm.temperature", 0.5)`
  * UI overrides: Settings panel modifies config in session state
* **Config access:**
  * Dot notation: `config.llm.temperature`
  * Dictionary access: `config["llm"]["temperature"]`
  * Immutable by default (require explicit `.set()` for changes)

### Constraints
* **No secrets in default.yaml:** API keys must use env var placeholders
* **Backward compatibility:** Adding new config keys shouldn't break existing configs
* **Type safety:** Use dataclasses or Pydantic for strong typing

### Implementation Notes
* Consider using Pydantic for validation (automatic type checking + nice errors)
* Support config inheritance: user config overrides default config
* Provide `config.to_dict()` for serialization (e.g., saving to file)

### Deliverables
* `config/default.yaml` with comprehensive defaults (see spec)
* `src/config.py` with:
  * `Config` class (or Pydantic BaseModel)
  * `load_config(config_path: Optional[Path] = None, overrides: Optional[Dict] = None) -> Config`
  * `Config.validate()` method
  * `Config.set(key_path: str, value: Any)` method
* Tests in `tests/test_config.py`:
  * Test default config loads successfully
  * Test validation catches invalid values (wrong types, out of range)
  * Test environment variable overrides
  * Test programmatic overrides
  * Test missing required fields raise error
  * Test serialization round-trip

### Definition of Done
- [ ] `config/default.yaml` contains all spec'd configuration keys with sensible defaults
- [ ] Config loads and validates successfully
- [ ] Type errors caught at load time (not runtime)
- [ ] Environment variables override YAML values
- [ ] Programmatic overrides work for UI integration
- [ ] All tests pass with >90% coverage
- [ ] README documents config system and common overrides

**Estimated effort:** 6-8 hours

---

## Issue 10 — Streamlit UI: Upload + Chat + Evidence Expanders + Settings Panel

**Title:** Streamlit UI: local web app with upload, chat, evidence, settings

**Labels:** `module:ui`, `priority:critical`, `complexity:high`

**Body:**

**Spec references:** Section 12, 9.2

### Goal
Build Streamlit web interface providing document upload, indexed document management, chat interface, citation display, evidence expanders, and settings panel.

### Requirements
* **Page layout:**
  * **Sidebar:**
    * Document upload widget (file uploader, PDF/DOCX/TXT)
    * "Index Document" button
    * Indexed documents list (filename, page count, date added)
    * Delete document button per item
    * Settings expander (collapsible)
  * **Main area:**
    * Chat interface (messages display)
    * Query input box
    * Response display with:
      * Answer text
      * "Show Evidence" expander per response
      * Citations displayed inline or as footnotes
* **Upload & indexing:**
  * Upload button triggers indexing pipeline
  * Show progress spinner during indexing
  * Display success/error message with details (# chunks added, errors)
  * Update indexed documents list immediately after indexing
* **Chat interface:**
  * Text input for user query
  * "Send" button (or Enter key)
  * Display user messages and assistant responses in chat bubbles
  * Response mode selector: radio buttons for Concise/Expanded
  * Handle low-evidence mode: display clarifying question differently (e.g., special icon/color)
* **Evidence expanders:**
  * One expander per response, labeled "Show Evidence (N chunks)"
  * Inside expander, display each retrieved chunk as:
    * **Header:** `[filename.pdf, page 5] - Score: 0.87`
    * **Body:** Chunk text (formatted, max height with scrollbar if long)
    * **Footer:** Optional "View full document" link (nice-to-have)
* **Settings panel:**
  * Editable fields for key hyperparameters:
    * Chunk size, chunk overlap
    * Top-k, min score threshold
    * LLM temperature, max tokens
  * "Apply Settings" button (updates config, re-initializes components if needed)
  * "Reset to Defaults" button
* **Session state management:**
  * Persist chat history across interactions (session state)
  * Persist config overrides (session state)
  * Indexed documents list persists (loaded from vector store metadata)

### Constraints
* **No drag-and-drop required:** Simple file uploader is sufficient
* **Responsive design:** Not required, but should work on desktop browsers
* **Accessibility:** Basic (keyboard navigation not critical for MVP)

### Implementation Notes
* Use `st.file_uploader` for upload
* Use `st.chat_message` for chat bubbles (Streamlit native)
* Use `st.expander` for evidence display
* Store chat history in `st.session_state.messages`
* For indexing, run in background if possible (or show blocking spinner)
* Consider `st.form` for settings panel to batch updates

### Deliverables
* `app.py` (main Streamlit entrypoint) OR `src/ui/` module:
  * `app.py`
  * `ui/components/` (optional, for organized code)
* Tests (optional for UI, but consider):
  * Integration test: upload fixture → verify indexed
  * Integration test: query → verify response format
* README section: "Running the Streamlit App"

### Definition of Done
- [ ] User can upload PDF/DOCX/TXT files and see them in indexed list
- [ ] Index persists across app restarts (vector store loads correctly)
- [ ] Chat interface works: query → response with citations
- [ ] Evidence expanders show chunk text with filename, location, score
- [ ] Settings panel allows editing hyperparameters and applying changes
- [ ] Concise/Expanded mode selector works
- [ ] Low-evidence mode displays clarifying question differently
- [ ] Delete document removes from index and updates UI
- [ ] Manual testing checklist completed (see below)

**Manual testing checklist:**
- [ ] Upload 3 PDFs, verify indexed list shows all 3
- [ ] Restart app, verify indexed list still shows 3 PDFs
- [ ] Delete 1 PDF, verify removed from list and retrieval
- [ ] Ask query with good evidence, verify concise answer + citations
- [ ] Click "Show Evidence", verify chunk text displays
- [ ] Switch to Expanded mode, ask same query, verify longer answer
- [ ] Ask query with no evidence, verify clarifying question
- [ ] Change chunk size in settings, re-index doc, verify different chunk count
- [ ] Change temperature in settings, verify LLM responses differ

**Estimated effort:** 16-20 hours

**Dependencies:** Issues #5, #8, #9, #11

---

## Issue 11 — Document Management: List + Delete Document from Index

**Title:** Document management: list indexed docs and delete selected doc

**Labels:** `module:doc-management`, `priority:medium`, `complexity:low`

**Body:**

**Spec references:** Storage/persistence + chunk metadata; supports non-goals and usability

### Goal
Implement document listing and deletion functionality enabling users to manage their indexed document library.

### Requirements
* **List documents:**
  * Query vector store for all unique documents
  * Return list of `DocumentMetadata` (document_id, filename, filetype, ingested_at, page_count/chunk_count)
  * Sort by `ingested_at` descending (most recent first)
* **Delete document:**
  * Given `document_id`, remove all associated chunks from vector store
  * Return deletion report: number of chunks deleted
  * Handle errors gracefully (document not found, deletion failure)
* **UI integration:**
  * Display indexed documents in sidebar as list items
  * Each item shows: filename, filetype icon, date added, chunk count
  * Each item has delete button (trash icon or "X")
  * Clicking delete shows confirmation dialog
  * After deletion, immediately update UI list (remove deleted item)

### Constraints
* **Cascade delete:** Deleting document must remove ALL chunks (no orphans)
* **Idempotency:** Deleting non-existent document should not error (return 0 chunks deleted)
* **No undo:** Deletion is permanent (document must be re-uploaded to re-index)

### Implementation Notes
* `list_documents()` implemented in Issue #3, but may need enhancement for metadata aggregation
* Consider caching document list in UI session state (refresh on index/delete operations)
* For UI, use `st.button` with unique key per document (e.g., `key=f"delete_{doc_id}"`)

### Deliverables
* `src/doc_manager.py` (optional wrapper) with:
  * `list_indexed_documents() -> List[DocumentSummary]` (extends DocumentMetadata with chunk_count)
  * `delete_document(document_id: str) -> DeletionReport`
* `src/types.py` updated with:
  * `DocumentSummary` dataclass
  * `DeletionReport` dataclass (document_id, chunks_deleted, success)
* UI integration in `app.py`:
  * Sidebar document list with delete buttons
  * Confirmation dialog before deletion
  * Refresh list after deletion
* Tests in `tests/test_doc_manager.py`:
  * Test list returns all indexed documents
  * Test delete removes all chunks for document
  * Test delete does not affect other documents
  * Test delete non-existent document returns 0 without error

### Definition of Done
- [ ] `list_indexed_documents()` returns accurate document summaries
- [ ] `delete_document()` removes all chunks and returns correct count
- [ ] UI displays indexed documents with metadata
- [ ] UI delete button works and updates list immediately
- [ ] After delete, retrieval does not return chunks from deleted document
- [ ] All tests pass with >85% coverage
- [ ] Manual test: delete document, verify gone from list and retrieval

**Estimated effort:** 4-6 hours

**Dependencies:** Issue #3

---

## Issue 12 — Tests + CI: Acceptance Tests and GitHub Actions

**Title:** Testing + CI: implement acceptance tests and GitHub Actions workflow

**Labels:** `infrastructure`, `priority:high`, `complexity:medium`

**Body:**

**Spec references:** Section 13

### Goal
Create comprehensive automated testing suite and CI/CD pipeline enforcing spec compliance and preventing regressions.

### Requirements
* **Test coverage targets:**
  * Unit tests: >85% coverage per module
  * Integration tests: >70% coverage for pipelines (indexing, retrieval, RAG)
  * Acceptance tests: All spec'd acceptance criteria
* **Acceptance test scenarios:**
  1. **Persistence:** Index documents, restart app (re-init vector store), verify documents still indexed and queryable
  2. **Citations:** Query returns answer with citations including correct filename and location
  3. **Evidence display:** Retrieved chunks returned with answer and displayable in UI
  4. **Low-evidence handling:** Query with no relevant docs triggers clarifying question mode
  5. **Expanded mode grounding:** Expanded answer remains grounded (no hallucinated facts)
  6. **Incremental indexing:** Add documents incrementally, verify existing docs unaffected
  7. **Document deletion:** Delete document, verify chunks removed from retrieval
* **Test organization:**
  * `tests/unit/` - Unit tests per module
  * `tests/integration/` - Integration tests (multi-module pipelines)
  * `tests/acceptance/` - End-to-end acceptance tests
  * `tests/fixtures/` - Test documents (small PDFs, DOCX, TXT)
* **CI workflow (GitHub Actions):**
  * Trigger on: Push to main, pull requests
  * Steps:
    1. Checkout code
    2. Setup Python 3.10+
    3. Install dependencies (pip install -r requirements.txt + dev dependencies)
    4. Run linting (ruff or flake8)
    5. Run type checking (mypy)
    6. Run tests with coverage (pytest --cov)
    7. Upload coverage report (codecov or similar, optional)
  * Matrix testing: Python 3.10, 3.11, 3.12 (optional)
  * OS matrix: Ubuntu (required), macOS/Windows (optional)

### Constraints
* **No external dependencies in CI:** Don't require Ollama or GPU (use mocks/stubs)
* **Fast CI:** Tests should complete in <10 minutes
* **Deterministic:** No flaky tests (fix or skip flaky tests)

### Implementation Notes
* Use pytest fixtures for reusable test data (indexed vector store, sample docs)
* Use `pytest-mock` for mocking LLM/embedding responses in unit tests
* For acceptance tests, consider using real (tiny) models or mock responses
* CI should cache pip dependencies and model downloads (if any)

### Deliverables
* Enhanced test suite in `tests/`:
  * Acceptance tests in `tests/acceptance/test_acceptance.py`
  * Integration tests in `tests/integration/`
  * Comprehensive unit tests (already partially delivered in earlier issues)
* `.github/workflows/ci.yml`:
  * Defines CI pipeline as specified above
* `tests/conftest.py`:
  * Shared fixtures (e.g., `vector_store`, `sample_pdf`, `config`)
* `requirements-dev.txt`:
  * Test dependencies (pytest, pytest-cov, pytest-mock, mypy, ruff)
* Coverage configuration:
  * `.coveragerc` or `pyproject.toml` with coverage settings
  * Target: >85% overall coverage

### Definition of Done
- [ ] All 7 acceptance criteria have corresponding automated tests
- [ ] Acceptance tests pass consistently
- [ ] CI workflow runs on every PR and main branch push
- [ ] CI tests pass on GitHub Actions (verified with real PR)
- [ ] Coverage report shows >85% overall coverage
- [ ] No flaky tests (tests pass 10/10 times locally and in CI)
- [ ] README documents how to run tests locally

**Estimated effort:** 12-16 hours

**Dependencies:** All prior issues (final integration)

---

## Issue 13 — Documentation: README + Setup Guide + Architecture Docs

**Title:** Documentation: comprehensive README with setup, usage, and architecture

**Labels:** `documentation`, `priority:medium`, `complexity:low`

**Body:**

### Goal
Create comprehensive documentation enabling new users to understand, install, and run the system, plus architecture documentation for contributors.

### Requirements
* **README.md sections:**
  1. **Overview:** Brief description of project (RAG system with local LLM)
  2. **Features:** Bullet list of key features (grounded QA, citations, evidence display, etc.)
  3. **Prerequisites:** Python version, Ollama/llama.cpp installation instructions
  4. **Installation:**
     * Clone repo
     * Create virtual environment
     * Install dependencies (`pip install -r requirements.txt`)
     * Download embedding model (automatic on first run)
     * Setup Ollama (install + pull model) OR llama.cpp (download model)
  5. **Configuration:**
     * How to edit `config/default.yaml`
     * How to use environment variables for overrides
     * How to switch between local and API backends
  6. **Usage:**
     * Running the Streamlit app (`streamlit run app.py`)
     * Uploading documents
     * Querying the system
     * Viewing evidence and citations
  7. **Testing:**
     * Running unit tests (`pytest tests/unit`)
     * Running integration tests (`pytest tests/integration`)
     * Running acceptance tests (`pytest tests/acceptance`)
     * Coverage report (`pytest --cov`)
  8. **Project Structure:** Directory tree with brief descriptions
  9. **Architecture:** High-level system diagram (optional, ASCII art or link to diagram)
  10. **Troubleshooting:** Common issues and solutions
  11. **Contributing:** Guidelines for contributors (optional for MVP)
  12. **License:** License information
* **Additional documentation:**
  * `docs/ARCHITECTURE.md`: Detailed system architecture, data flow diagrams, module interactions
  * `docs/CONFIG.md`: Complete configuration reference (all keys with descriptions)
  * `docs/DEVELOPMENT.md`: Development setup, coding standards, testing guidelines

### Constraints
* **Clarity:** Instructions must be executable by someone unfamiliar with the codebase
* **Accuracy:** All commands must be tested and work as documented
* **Maintainability:** Use includes/links to avoid duplication (e.g., link to config keys instead of copy-paste)

### Implementation Notes
* Test setup instructions on fresh VM/container to verify completeness
* Use code blocks with syntax highlighting for commands
* Include example config snippets
* Link to external resources (Ollama docs, model sources)

### Deliverables
* `README.md` (comprehensive, covering all sections above)
* `docs/ARCHITECTURE.md`
* `docs/CONFIG.md`
* `docs/DEVELOPMENT.md`
* `docs/diagrams/` (optional, if using images)
* All documentation reviewed for accuracy and completeness

### Definition of Done
- [ ] README includes all required sections with accurate content
- [ ] Installation instructions tested on fresh environment (Linux + macOS)
- [ ] Configuration documentation covers all config keys with examples
- [ ] Architecture documentation includes system diagram and data flow
- [ ] Troubleshooting section addresses common issues (model not found, Ollama connection error, etc.)
- [ ] All documentation links work (no 404s)
- [ ] New contributor can follow README to run app successfully

**Estimated effort:** 8-10 hours

---

## Appendix: Supporting Artifacts

### A. Repository Scaffold

Recommended directory structure:

```
rag-system/
├── .github/
│   └── workflows/
│       └── ci.yml
├── config/
│   └── default.yaml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CONFIG.md
│   └── DEVELOPMENT.md
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── llm.py
│   ├── rag.py
│   ├── indexing.py
│   ├── doc_manager.py
│   ├── config.py
│   ├── types.py
│   └── prompts.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   ├── acceptance/
│   └── fixtures/
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
├── README.md
└── LICENSE
```

### B. Config Template (`config/default.yaml`)

```yaml
# Chunking configuration
chunking:
  strategy: "fixed"
  chunk_size: 500
  chunk_overlap: 50

# Retrieval configuration
retrieval:
  top_k: 5
  min_score: 0.3
  include_metadata: true

# Embeddings configuration
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32

# LLM configuration
llm:
  backend: "local"  # "local" or "api"
  local_runtime: "ollama"  # "ollama" or "llama_cpp"
  model: "llama3.2:3b"
  temperature: 0.1
  max_tokens: 512
  api_base_url: null
  api_key: null  # Use env var LLM_API_KEY

# Paths
paths:
  persist_dir: "./data/vector_store"
  upload_dir: "./data/uploads"

# Grounding
grounding_strict: true

# UI configuration
ui:
  page_title: "Local RAG Q&A System"
  theme: "light"
```

### C. CI Workflow Template (`.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with ruff
      run: |
        ruff check src/ tests/
    
    - name: Type check with mypy
      run: |
        mypy src/
    
    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=src --cov-report=xml --cov-report=term
    
    - name: Upload coverage to Codecov (optional)
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: false
```

---

## Total Estimated Effort

| Issue | Module | Hours |
|-------|--------|-------|
| 1 | Ingestion | 6-8 |
| 2 | Chunking | 8-10 |
| 3 | Vector Store | 10-12 |
| 4 | Embeddings | 6-8 |
| 5 | Indexing Pipeline | 10-12 |
| 6 | Retrieval | 6-8 |
| 7 | LLM Backends | 10-14 |
| 8 | RAG Orchestrator | 14-16 |
| 9 | Config System | 6-8 |
| 10 | Streamlit UI | 16-20 |
| 11 | Document Management | 4-6 |
| 12 | Tests + CI | 12-16 |
| 13 | Documentation | 8-10 |
| **TOTAL** | | **116-148 hours** |

**Note:** Estimates assume one developer working full-time. Parallelization across multiple developers can significantly reduce wall-clock time.




**End of Project Specification**
