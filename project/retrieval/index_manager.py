import pickle
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import faiss

from ..utils.logger import Logger
from ..utils.config import RetrieverConfig

logger = Logger.get_logger(__name__)


class IndexManager:
    """Manages FAISS index creation, storage, and retrieval."""
    
    def __init__(self, config: RetrieverConfig):
        """
        Initialize index manager.
        
        Args:
            config: RetrieverConfig object
        """
        self.config = config
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[dict] = []
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure knowledge base directory exists."""
        self.config.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Knowledge base directory ready: {self.config.knowledge_base_dir}")
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """
        Create a new FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings (n, embedding_dim)
        """
        if embeddings.shape[0] == 0:
            logger.error("Cannot build index with empty embeddings")
            raise ValueError("Embeddings array is empty")
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        logger.info(f"Building FAISS index for {embeddings.shape[0]} papers...")
        
        try:
            # Create IndexFlatIP (inner product for cosine similarity on normalized vectors)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
            logger.info(f"Index built successfully with {self.index.ntotal} papers")
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise
    
    def add_to_index(
        self,
        embeddings: np.ndarray,
        metadata: List[dict]
    ) -> None:
        """
        Add embeddings and metadata to existing index (incremental).
        
        Args:
            embeddings: New embeddings array
            metadata: Corresponding metadata list
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Embeddings and metadata length mismatch")
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        if self.index is None:
            logger.warning("Index doesn't exist, creating new one...")
            self.build_index(embeddings)
        else:
            logger.info(f"Adding {embeddings.shape[0]} papers to existing index...")
            self.index.add(embeddings)
        
        self.metadata.extend(metadata)
        logger.info(f"Index now contains {self.index.ntotal} papers")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[dict]:
        """
        Search the index for similar papers.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity score
            
        Returns:
            List of results with scores and metadata
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not initialized")
            return []
        
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        try:
            # Search
            distances, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for idx, (distance, index) in enumerate(zip(distances[0], indices[0])):
                # Skip invalid indices
                if index < 0 or index >= len(self.metadata):
                    continue
                
                # Apply threshold
                if distance < threshold:
                    continue
                
                result = {
                    "rank": len(results) + 1,
                    "score": float(distance),
                    **self.metadata[index]
                }
                results.append(result)
            
            logger.debug(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[List[dict]]:
        """
        Search for multiple queries.
        
        Args:
            query_embeddings: Multiple query embeddings
            top_k: Number of top results per query
            threshold: Minimum similarity score
            
        Returns:
            List of result lists
        """
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        
        logger.info(f"Batch search for {query_embeddings.shape[0]} queries...")
        
        batch_results = []
        for i, query_emb in enumerate(query_embeddings):
            results = self.search(query_emb, top_k, threshold)
            batch_results.append(results)
            logger.debug(f"Processed query {i+1}/{query_embeddings.shape[0]}")
        
        return batch_results
    
    def save_index(self) -> None:
        """Save index and metadata to disk."""
        if self.index is None:
            logger.warning("Index is not initialized, nothing to save")
            return
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.config.index_path))
            logger.info(f"Index saved to {self.config.index_path}")
            
            # Save metadata
            with open(self.config.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Metadata saved to {self.config.metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load_index(self) -> bool:
        """
        Load index and metadata from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.config.index_path.exists():
                logger.warning(f"Index file not found: {self.config.index_path}")
                return False
            
            self.index = faiss.read_index(str(self.config.index_path))
            logger.info(f"Index loaded from {self.config.index_path}")
            
            if not self.config.metadata_path.exists():
                logger.warning(f"Metadata file not found: {self.config.metadata_path}")
                self.metadata = []
                return True
            
            with open(self.config.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Metadata loaded from {self.config.metadata_path} ({len(self.metadata)} records)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_index_info(self) -> dict:
        """Get information about the current index."""
        if self.index is None:
            return {"status": "not initialized"}
        
        return {
            "status": "ready",
            "total_papers": self.index.ntotal,
            "embedding_dim": self.index.d,
            "metadata_records": len(self.metadata),
            "index_type": self.config.index_type
        }
