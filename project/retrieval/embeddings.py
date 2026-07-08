from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

from ..utils.logger import Logger
from ..utils.config import RetrieverConfig

logger = Logger.get_logger(__name__)


class EmbeddingGenerator:
    """Handles embedding generation using SentenceTransformer."""
    
    def __init__(self, config: RetrieverConfig):
        """
        Initialize embedding generator.
        
        Args:
            config: RetrieverConfig object
        """
        self.config = config
        self.model = self._load_model()
        logger.info(f"Embedding model initialized: {config.embedding_model}")
    
    def _load_model(self) -> SentenceTransformer:
        """
        Load the SentenceTransformer model.
        
        Returns:
            Loaded model
        """
        try:
            model = SentenceTransformer(
                self.config.embedding_model,
                device="cuda" if self.config.use_gpu else "cpu"
            )
            logger.info(f"Model loaded on {'GPU' if self.config.use_gpu else 'CPU'}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_embeddings(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            normalize: Whether to normalize embeddings (L2 norm)
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([]).reshape(0, self.config.embedding_dim)
        
        logger.debug(f"Generating embeddings for {len(texts)} texts...")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            if normalize:
                embeddings = self._normalize_embeddings(embeddings)
                logger.debug("Embeddings normalized (L2)")
            
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings using L2 norm.
        
        Args:
            embeddings: Embeddings array
            
        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        return embeddings / norms
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.config.embedding_dim
