"""Vector store implementation using Chroma for persistent semantic search.

Provides ChromaVectorStore class for managing document chunks with embedding-based
retrieval, supporting upsert, delete, and query operations.
"""

from typing import List, Dict, Optional, Set, Any
import numpy as np
import chromadb

from src.chunk_types import Chunk, RetrievedChunk


class ChromaVectorStore:
    """Persistent vector store using Chroma for semantic search over document chunks.
    
    Handles embedding storage, retrieval, and document management with full
    metadata preservation for citations and filtering.
    """
    
    def __init__(self, persist_path: str, collection_name: str = "rag_chunks"):
        """Initialize Chroma vector store with persistent storage.
        Stores:
            - ids: chunk_id
            - documents: chunk_text
            - metadatas: document_id, filename, filetype, chunk_index (int), location (int), text_preview (optional)
        Args:
            persist_path: Directory path where Chroma database will be stored
            collection_name: Name of the collection to use (default: "rag_chunks")
        Raises:
            chromadb.errors.ChromaException: If Chroma client or collection initialization fails
        
        Notes:
        - Chroma distances depend on collection metric and embedding normalization.
        (default is L2 distance)
        - We return both `distance` (from Chroma) and a monotonic `score` for convenience.
        - Do NOT treat `score` as cosine similarity unless you explicitly configure for it
        """
        self.persist_path = persist_path
        self.collection_name = collection_name
        
        # Initialize Chroma persistent client
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # Get or create collection (Chroma uses L2 distance by default)
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
    
    def upsert_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Insert or update chunks with their embeddings in the vector store.
        
        Batch operation for performance. Overwrites existing chunk_ids without
        creating duplicates.
        
        Args:
            chunks: List of Chunk objects to store
            embeddings: NumPy array of shape (len(chunks), embedding_dim)
                       Embeddings corresponding to each chunk's text
                       
        Raises:
            ValueError: If chunks and embeddings have mismatched lengths
        """
        if not chunks:
            return

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings)

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D array (n, d), got shape {embeddings.shape}")

        
        # Prepare data for Chroma upsert
        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.chunk_text for chunk in chunks]
        metadatas = []
        
        # Build metadata for each chunk
        for chunk in chunks:
            metadata = {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "filetype": chunk.filetype,
                "chunk_index": int(chunk.chunk_index),
                "location": int(chunk.location) if chunk.location is not None else 0,

            }
            if chunk.text_preview:
                metadata["text_preview"] = chunk.text_preview
            metadatas.append(metadata)
        
        # Convert embeddings to list format for Chroma
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
        # Batch upsert to vector store
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )
    
    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievedChunk]:
        """Retrieve top-k most similar chunks to a query embedding.
        
        Args:
            query_embedding: NumPy array of shape (embedding_dim,) representing the query
            top_k: Number of top results to return (default: 5, must be > 0)
            
        Returns:
            List of RetrievedChunk objects with similarity scores, ordered by score (highest first)
            
        Raises:
            ValueError: If top_k is not positive or embedding has invalid shape
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        if query_embedding.ndim != 1:
            raise ValueError(f"query_embedding must be 1D array, got shape {query_embedding.shape}")
        
        # Convert embedding to list for Chroma
        query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Query Chroma for similar embeddings
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_k
        )
        
        retrieved_chunks = []
        
        # Process results (Chroma returns lists of results for each query)
        if (
            results
            and results.get("ids")
            and len(results["ids"]) > 0
            and results.get("metadatas") is not None
            and results.get("documents") is not None
            and results.get("distances") is not None
        ):
            metadatas_list = results["metadatas"][0] if results["metadatas"] and results["metadatas"][0] is not None else []
            for i, chunk_id in enumerate(results["ids"][0]):
                # Extract metadata and content
                metadata = metadatas_list[i] if i < len(metadatas_list) and metadatas_list[i] is not None else {}
                text = results["documents"][0][i]
                distance = results["distances"][0][i]
                
                # Chroma uses L2 (Euclidean) distance internally
                # Convert L2 distance to similarity-like score in (0, 1]
                # NOTE: This is NOT cosine similarity. 1.0 = perfect match (distance=0)
                # For L2 distance: score = 1 / (1 + distance)
                similarity_score = 1 / (1 + float(distance))
                
                # Reconstruct RetrievedChunk
                retrieved_chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    filename=metadata.get("filename", ""),
                    filetype=str(metadata.get("filetype", "")),
                    chunk_index= metadata.get("chunk_index", 0),
                    location=int(metadata.get("location", 0)) if metadata.get("location") is not None else 0,
                    chunk_text=text,
                    text_preview=metadata.get("text_preview"),
                    score=similarity_score
                )
                retrieved_chunks.append(retrieved_chunk)
        
        return retrieved_chunks
    
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document.
        
        Args:
            document_id: The document ID to delete all chunks for
            
        Returns:
            Number of chunks deleted
        """
        # Query all chunks for this document
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        chunk_ids_to_delete = results["ids"] if results and results["ids"] else []
        
        # Delete chunks if any found
        if chunk_ids_to_delete:
            self.collection.delete(ids=chunk_ids_to_delete)
        
        return len(chunk_ids_to_delete)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """Retrieve unique documents currently in the vector store.
        
        Returns:
            List of dicts containing document metadata:
            [{document_id, filename, filetype, chunk_count}, ...]
        """
        # Get all data to extract unique documents
        all_data = self.collection.get(include=["metadatas"])
        
        if not all_data or not all_data.get("metadatas"):
            return []
        
        # Track unique documents
        documents = {}
        for metadata in all_data["metadatas"]:
            doc_id = metadata["document_id"]
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "filename": metadata["filename"],
                    "filetype": metadata["filetype"],
                    "chunk_count": 0
                }
            documents[doc_id]["chunk_count"] += 1
        
        return list(documents.values())
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary with stats:
            {
                "total_chunks": int,
                "unique_documents": int,
                "embedding_dimension": int
            }
        """
        # Get collection count
        total_chunks = self.collection.count()
        
        # Get all data including embeddings
        all_data = self.collection.get(include=["embeddings", "metadatas"])
        unique_docs = set()
        embedding_dim = 0
        
        # Extract unique documents from metadata
        if all_data and "metadatas" in all_data and all_data["metadatas"]:
            for metadata in all_data["metadatas"]:
                unique_docs.add(metadata["document_id"])
        
        # Get embedding dimension from first embedding
        if all_data and "embeddings" in all_data:
            embeddings = all_data["embeddings"]
            if embeddings is not None and len(embeddings) > 0:
                embedding_dim = len(embeddings[0]) if embeddings[0] is not None else 0
        
        return {
            "total_chunks": total_chunks,
            "unique_documents": len(unique_docs),
            "embedding_dimension": embedding_dim
        }
