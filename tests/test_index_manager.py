import pytest
import tempfile
from pathlib import Path
import numpy as np

from project.retrieval.index_manager import IndexManager
from project.utils.config import RetrieverConfig


@pytest.fixture
def config():
    """Create test configuration."""
    tmpdir = tempfile.mkdtemp()
    return RetrieverConfig(
        knowledge_base_dir=Path(tmpdir),
        index_path=Path(tmpdir) / "faiss.index",
        metadata_path=Path(tmpdir) / "metadata.pkl"
    )


@pytest.fixture
def manager(config):
    """Create index manager."""
    return IndexManager(config)


@pytest.fixture
def sample_embeddings():
    """Create sample normalized embeddings."""
    np.random.seed(42)
    embeddings = np.random.randn(5, 384).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return [
        {"paper_id": f"paper_{i}", "title": f"Title {i}", "year": 2020 + i}
        for i in range(5)
    ]


class TestIndexManager:
    """Test suite for IndexManager."""
    
    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert manager.config is not None
        assert manager.index is None
        assert manager.metadata == []
    
    def test_build_index(self, manager, sample_embeddings):
        """Test index building."""
        manager.build_index(sample_embeddings)
        
        assert manager.index is not None
        assert manager.index.ntotal == 5
        assert manager.index.d == 384
    
    def test_add_to_index(self, manager, sample_embeddings, sample_metadata):
        """Test adding to index."""
        manager.build_index(sample_embeddings[:3])
        manager.add_to_index(sample_embeddings[3:], sample_metadata[3:])
        
        assert manager.index.ntotal == 5
        assert len(manager.metadata) == 5
    
    def test_search(self, manager, sample_embeddings, sample_metadata):
        """Test search."""
        manager.build_index(sample_embeddings)
        manager.metadata = sample_metadata
        
        query = sample_embeddings[0]
        results = manager.search(query, top_k=3)
        
        assert len(results) <= 3
        assert all("score" in r for r in results)
        assert all("title" in r for r in results)
    
    def test_save_and_load(self, manager, sample_embeddings, sample_metadata):
        """Test saving and loading."""
        manager.build_index(sample_embeddings)
        manager.metadata = sample_metadata
        manager.save_index()
        
        # Create new manager and load
        manager2 = IndexManager(manager.config)
        success = manager2.load_index()
        
        assert success
        assert manager2.index.ntotal == 5
        assert len(manager2.metadata) == 5
    
    def test_get_index_info(self, manager, sample_embeddings):
        """Test index info retrieval."""
        manager.build_index(sample_embeddings)
        info = manager.get_index_info()
        
        assert info["total_papers"] == 5
        assert info["embedding_dim"] == 384
        assert info["status"] == "ready"
