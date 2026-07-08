from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RetrieverConfig:
    """Configuration for the retrieval engine."""
    
    # Model configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Index configuration
    index_type: str = "IndexFlatIP"  # Cosine similarity
    batch_size: int = 32
    
    # Paths
    papers_dir: Path = Path("sample_data/papers")
    knowledge_base_dir: Path = Path("project/knowledge_base")
    index_path: Path = Path("project/knowledge_base/faiss.index")
    metadata_path: Path = Path("project/knowledge_base/metadata.pkl")
    log_file: Optional[Path] = Path("logs/retriever.log")
    
    # Retrieval
    default_top_k: int = 5
    similarity_threshold: float = 0.0
    
    # Performance
    use_gpu: bool = False
    num_workers: int = 4
    
    def __post_init__(self):
        """Ensure all paths are Path objects."""
        if isinstance(self.papers_dir, str):
            self.papers_dir = Path(self.papers_dir)
        if isinstance(self.knowledge_base_dir, str):
            self.knowledge_base_dir = Path(self.knowledge_base_dir)
        if isinstance(self.index_path, str):
            self.index_path = Path(self.index_path)
        if isinstance(self.metadata_path, str):
            self.metadata_path = Path(self.metadata_path)
        if isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)
