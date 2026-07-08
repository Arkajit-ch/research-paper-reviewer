from typing import List, Optional, Iterator
from pathlib import Path
import numpy as np

from ..utils.logger import Logger
from ..utils.config import RetrieverConfig
from ..utils.data_loader import DataLoader, PaperData
from .embeddings import EmbeddingGenerator
from .index_manager import IndexManager

logger = Logger.get_logger(__name__)


class PaperRetriever:
    """
    Main retrieval engine for querying similar research papers.
    
    This class orchestrates the entire pipeline:
    1. Data loading from JSON files
    2. Embedding generation using SentenceTransformer
    3. FAISS index management
    4. Similarity search
    """
    
    def __init__(self, config: Optional[RetrieverConfig] = None):
        """
        Initialize the paper retriever.
        
        Args:
            config: RetrieverConfig object (uses default if None)
        """
        self.config = config or RetrieverConfig()
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.index_manager = IndexManager(self.config)
        logger.info("PaperRetriever initialized")
    
    def build_retrieval_index(self, papers_dir: Optional[Path] = None) -> int:
        """
        Build a complete retrieval index from paper JSON files.
        
        This is the main entry point for index creation:
        - Loads all papers from directory
        - Generates embeddings
        - Creates FAISS index
        - Saves to disk
        
        Args:
            papers_dir: Directory containing paper JSON files
            
        Returns:
            Number of papers indexed
        """
        papers_dir = papers_dir or self.config.papers_dir
        logger.info(f"Building retrieval index from {papers_dir}")
        
        # Load papers
        papers = list(DataLoader.load_all_papers(papers_dir))
        if not papers:
            logger.error("No papers loaded from directory")
            return 0
        
        logger.info(f"Loaded {len(papers)} papers")
        
        # Validate papers
        valid_papers = [p for p in papers if DataLoader.validate_paper(p)]
        logger.info(f"Validated {len(valid_papers)} papers")
        
        if not valid_papers:
            logger.error("No valid papers to index")
            return 0
        
        # Extract texts and metadata
        texts = [p.combined_text for p in valid_papers]
        metadata = self._create_metadata_list(valid_papers)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(texts, normalize=True)
        
        # Build index
        logger.info("Building FAISS index...")
        self.index_manager.build_index(embeddings)
        self.index_manager.metadata = metadata
        
        # Save index
        logger.info("Saving index to disk...")
        self.index_manager.save_index()
        
        logger.info(f"Index building complete: {len(valid_papers)} papers indexed")
        return len(valid_papers)
    
    def incremental_index(
        self,
        papers_dir: Optional[Path] = None,
        load_existing: bool = True
    ) -> int:
        """
        Add new papers to existing index incrementally.
        
        Args:
            papers_dir: Directory with new papers
            load_existing: Whether to load existing index first
            
        Returns:
            Number of papers added
        """
        papers_dir = papers_dir or self.config.papers_dir
        
        if load_existing:
            logger.info("Loading existing index...")
            self.index_manager.load_index()
        
        logger.info(f"Loading new papers from {papers_dir}")
        papers = list(DataLoader.load_all_papers(papers_dir))
        
        if not papers:
            logger.warning("No new papers found")
            return 0
        
        # Filter out duplicates
        existing_ids = {m["paper_id"] for m in self.index_manager.metadata}
        new_papers = [p for p in papers if p.paper_id not in existing_ids]
        
        logger.info(f"Found {len(new_papers)} new papers (skipped {len(papers) - len(new_papers)} duplicates)")
        
        if not new_papers:
            return 0
        
        # Process new papers
        valid_papers = [p for p in new_papers if DataLoader.validate_paper(p)]
        if not valid_papers:
            logger.warning("No valid new papers")
            return 0
        
        texts = [p.combined_text for p in valid_papers]
        metadata = self._create_metadata_list(valid_papers)
        
        embeddings = self.embedding_generator.generate_embeddings(texts, normalize=True)
        self.index_manager.add_to_index(embeddings, metadata)
        
        # Save updated index
        self.index_manager.save_index()
        logger.info(f"Incremental indexing complete: {len(valid_papers)} papers added")
        
        return len(valid_papers)
    
    def search_by_text(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        threshold: float = 0.0
    ) -> List[dict]:
        """
        Search for similar papers by text query.
        
        Args:
            query_text: Query text (title, abstract, etc.)
            top_k: Number of results (uses config default if None)
            threshold: Minimum similarity score
            
        Returns:
            List of matching papers with scores
        """
        top_k = top_k or self.config.default_top_k
        
        logger.info(f"Searching by text: '{query_text[:100]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings(
            [query_text],
            normalize=True
        )[0]
        
        # Search index
        results = self.index_manager.search(query_embedding, top_k, threshold)
        logger.info(f"Found {len(results)} similar papers")
        
        return results
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        threshold: float = 0.0
    ) -> List[dict]:
        """
        Search using a pre-computed embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            threshold: Minimum similarity score
            
        Returns:
            List of matching papers
        """
        top_k = top_k or self.config.default_top_k
        logger.debug(f"Searching by embedding (top_k={top_k})")
        return self.index_manager.search(query_embedding, top_k, threshold)
    
    def batch_search(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        threshold: float = 0.0
    ) -> List[List[dict]]:
        """
        Search for multiple queries at once.
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
            threshold: Minimum similarity score
            
        Returns:
            List of result lists
        """
        top_k = top_k or self.config.default_top_k
        logger.info(f"Batch search for {len(queries)} queries")
        
        # Generate embeddings for all queries
        query_embeddings = self.embedding_generator.generate_embeddings(
            queries,
            normalize=True
        )
        
        # Batch search
        results = self.index_manager.batch_search(query_embeddings, top_k, threshold)
        logger.info(f"Batch search complete")
        
        return results
    
    def load_retrieval_index(self) -> bool:
        """
        Load pre-built index from disk.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Loading retrieval index from disk...")
        success = self.index_manager.load_index()
        
        if success:
            info = self.index_manager.get_index_info()
            logger.info(f"Index loaded: {info['total_papers']} papers")
        else:
            logger.error("Failed to load index")
        
        return success
    
    def save_retrieval_index(self) -> None:
        """Save current index to disk."""
        logger.info("Saving retrieval index...")
        self.index_manager.save_index()
    
    def get_index_stats(self) -> dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index information
        """
        info = self.index_manager.get_index_info()
        info["config"] = {
            "model": self.config.embedding_model,
            "embedding_dim": self.config.embedding_dim,
            "default_top_k": self.config.default_top_k,
        }
        return info
    
    @staticmethod
    def _create_metadata_list(papers: List[PaperData]) -> List[dict]:
        """
        Create metadata dictionaries from papers.
        
        Args:
            papers: List of PaperData objects
            
        Returns:
            List of metadata dictionaries
        """
        metadata = []
        for paper in papers:
            metadata.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "authors": paper.authors,
                "reference_count": paper.reference_count,
                "file_path": paper.file_path,
                "abstract": paper.abstract
            })
        return metadata
