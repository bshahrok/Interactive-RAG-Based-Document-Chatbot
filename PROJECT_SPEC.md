# PROJECT_SPEC.md

## Project Title
Interactive RAG-Based Document Chatbot (Local, Single-User)

---

## 1. Project Goal

Build a **single-user local web application** that allows a user to upload documents
(**PDF, DOCX, TXT**) and query them via natural-language chat.

The system must use **retrieval-augmented generation (RAG)** and produce:
- grounded answers
- chunk-level citations
- an evidence view showing retrieved chunk text

The default configuration must be:
- **fully local**
- **free/open-source**
- **concise by default**
while allowing all major hyperparameters and backends to be configurable.

This document is **authoritative**.  
Code must not introduce features, defaults, or behaviors not specified here.

---

## 2. Non-Functional Constraints

- Local-first (no required external services)
- Single-user only
- No authentication
- Low memory usage:
  - avoid holding full documents in RAM after indexing
  - prefer disk-backed storage
- Runs on a typical laptop CPU
- No drag-and-drop UI required

---

## 3. Supported Inputs

### File Types
- PDF
- DOCX
- TXT

### Upload Behavior
- Files are uploaded via a local web UI.
- Uploaded files are **not duplicated** into an internal workspace by default.
- The system stores extracted chunk text and metadata required for retrieval.

---

## 4. Architecture Overview

### Default Stack (Free)
- UI: Streamlit
- Vector store: Chroma (disk-persisted)
- Embeddings: local embedding model
- LLM (default): local runtime (Ollama or llama.cpp)
- Optional: API-based LLM (user-configurable, disabled by default)

---

## 5. Data Model

### 5.1 Document Metadata
Each ingested document must have:
- document_id (UUID)
- filename
- filetype (pdf | docx | txt)
- original_path (optional, informational)
- ingested_at (timestamp)

### 5.2 Chunk Metadata
Each chunk must have:
- chunk_id (UUID)
- document_id
- filename
- filetype
- chunk_index (int)
- location:
  - PDF: page number
  - DOCX/TXT: chunk_index-based locator
- chunk_text (full text)
- optional text_preview (short excerpt)

---

## 6. Processing Pipeline

### 6.1 Ingestion
- PDF: extract text with page numbers when possible
- DOCX: extract paragraphs (heading awareness optional)
- TXT: raw text read

### 6.2 Chunking
Chunking must be configurable.

**Default behavior:**
- strategy: fixed-size
- chunk_size: configurable (default ~800–1200 chars)
- chunk_overlap: configurable (default ~100–200 chars)

PDF chunks must preserve page association.

### 6.3 Embeddings
- Compute embeddings for each chunk.
- Default model must be local.
- Store embeddings and metadata in Chroma.

### 6.4 Persistence
- Vector index must persist on disk.
- Restarting the app must not require re-embedding.

---

## 7. Retrieval

Given a user query:
1. Embed the query
2. Retrieve top-k chunks via similarity search
3. Return chunks with metadata and similarity scores

Configuration options must include:
- top_k
- similarity metric
- minimum retrieval score threshold

---

## 8. RAG Answer Generation

### 8.1 Grounding Rules
- Answers must be grounded **only** in retrieved chunks.
- No external knowledge or guessing.

### 8.2 Default Response Style
- Concise by default
- Expanded answer available on user request

### 8.3 Failure / Low-Evidence Mode
If:
- no chunks are retrieved, or
- retrieval scores are below threshold

Then the system must:
- NOT answer substantively
- Ask a clarifying question
- Optionally suggest which documents appear closest

---

## 9. Explainability & Evidence

### 9.1 Citations (Required)
Each answer must include:
- filename
- location (page or chunk index)
- chunk identifier

### 9.2 Evidence View (Required)
The UI must allow users to view:
- retrieved chunk text
- associated metadata
- similarity score (optional)

---

## 10. Conversation Behavior

- Multi-turn chat supported
- Lightweight memory:
  - last N turns or rolling summary (configurable)
- User must be able to:
  - expand an answer
  - continue asking follow-ups grounded in same documents

---

## 11. Configuration System

All of the following must be configurable (via config file and/or UI):

### Chunking
- strategy
- chunk_size
- chunk_overlap

### Embeddings
- model_name
- device (CPU default)

### Vector Store
- persist_path
- top_k
- min_score

### LLM
- backend (local | api)
- local_runtime (ollama | llama_cpp)
- model
- temperature
- max_output_tokens
- concise_mode (default true)

### Grounding Policy
- grounding_strict (default true)

Defaults must favor:
- free
- local
- low memory usage

---

## 12. UI Requirements (Streamlit)

### Required UI Elements
- Document upload panel
- Indexed document list
- Chat interface
- Evidence expanders per answer
- Settings panel

---

## 13. Acceptance Tests (Definition of Done)

1. Index persists across app restarts.
2. Answers include chunk-level citations.
3. Evidence view shows retrieved chunk text.
4. Queries without support trigger clarifying questions.
5. Expanded answers remain grounded and cited.

---

## 14. Explicit Non-Goals

The system must NOT:
- perform internet search
- fine-tune models
- learn from chat history across sessions
- silently hallucinate answers
- require authentication or cloud services


