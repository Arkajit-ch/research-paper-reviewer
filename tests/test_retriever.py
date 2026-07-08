import pytest
from pathlib import Path
import tempfile
import json
import numpy as np

from project.retrieval.retriever import PaperRetriever
from project.utils.config import RetrieverConfig


@pytest.fixture
def sample_papers():
    """Create sample paper data for testing."""
    papers = [
        {
            "name": "paper_1.json",
            "metadata": {
                "title": "Deep Learning Fundamentals",
                "authors": ["Alice", "Bob"],
                "year": 2020
            },
            "abstractText": "This paper discusses the fundamentals of deep learning.",
            "sections": [],
            "references": ["ref1", "ref2"]
        },
        {
            "name": "paper_2.json",
            "metadata": {
                "title": "Neural Networks for NLP",
                "authors": ["Charlie"],
                "year": 2021
            },
            "abstractText": "An exploration of neural networks in natural language processing.",
            "sections": [],
            "references": ["ref1"]
        },
        {
            "name": "paper_3.json",
            "metadata": {
                "title": "Computer Vision Applications",
                "authors": ["Diana", "Eve"],
                "year": 2022
            },
            "abstractText": "Survey of computer vision techniques and applications.",
            "sections": [],
            "references": []
        }
    ]
    return papers


@pytest.fixture
def temp_papers_dir(sample_papers):
    """Create temporary directory with sample papers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        for paper in sample_papers:
            filepath = tmppath / paper["name"]
            with open(filepath, 'w') as f:
                json.dump(paper, f)
        yield tmppath


@pytest.fixture
def config(temp_papers_dir):
    """Create test configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield RetrieverConfig(
            papers_dir=temp_papers_dir,
            knowledge_base_dir=Path(tmpdir) / "kb",
            index_path=Path(tmpdir) / "kb" / "faiss.index",
            metadata_path=Path(tmpdir) / "kb" / "metadata.pkl",
            embedding_model="all-MiniLM-L6-v2",
            batch_size=32,
            default_top_k=2
        )


class TestPaperRetriever:
    """Test suite for PaperRetriever."""
    
    def test_initialization(self, config):
        """Test retriever initialization."""
        retriever = PaperRetriever(config)
        assert retriever is not None
        assert retriever.config == config
        assert retriever.embedding_generator is not None
        assert retriever.index_manager is not None
    
    def test_build_index(self, config):
        """Test index building."""
        retriever = PaperRetriever(config)
        count = retriever.build_retrieval_index()
        
        assert count == 3
        assert retriever.index_manager.index is not None
        assert retriever.index_manager.index.ntotal == 3
        assert len(retriever.index_manager.metadata) == 3
    
    def test_save_and_load_index(self, config):
        """Test saving and loading index."""
        # Build and save
        retriever1 = PaperRetriever(config)
        count1 = retriever1.build_retrieval_index()
        retriever1.save_retrieval_index()
        
        # Load
        retriever2 = PaperRetriever(config)
        success = retriever2.load_retrieval_index()
        
        assert success
        assert retriever2.index_manager.index.ntotal == count1
        assert len(retriever2.index_manager.metadata) == count1
    
    def test_search_by_text(self, config):
        """Test text-based search."""
        retriever = PaperRetriever(config)
        retriever.build_retrieval_index()
        
        query = "neural networks"
        results = retriever.search_by_text(query, top_k=2)
        
        assert len(results) > 0
        assert len(results) <= 2
        assert all("score" in r for r in results)
        assert all("title" in r for r in results)
        assert all(r["score"] >= 0 for r in results)
    
    def test_search_by_embedding(self, config):
        """Test embedding-based search."""
        retriever = PaperRetriever(config)
        retriever.build_retrieval_index()
        
        query_embedding = retriever.embedding_generator.generate_embeddings(
            ["deep learning"],
            normalize=True
        )[0]
        
        results = retriever.search_by_embedding(query_embedding, top_k=2)
        
        assert len(results) > 0
        assert all("score" in r for r in results)
    
    def test_batch_search(self, config):
        """Test batch search."""
        retriever = PaperRetriever(config)
        retriever.build_retrieval_index()
        
        queries = [
            "deep learning",
            "natural language processing",
            "computer vision"
        ]
        results = retriever.batch_search(queries, top_k=2)
        
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
    
    def test_index_stats(self, config):
        """Test index statistics."""
        retriever = PaperRetriever(config)
        retriever.build_retrieval_index()
        
        stats = retriever.get_index_stats()
        
        assert "total_papers" in stats
        assert stats["total_papers"] == 3
        assert "config" in stats
    
    def test_incremental_indexing(self, config):
        """Test incremental indexing."""
        # Build initial index
        retriever = PaperRetriever(config)
        count1 = retriever.build_retrieval_index()
        
        # Add more papers
        new_paper = {
            "name": "paper_4.json",
            "metadata": {
                "title": "Reinforcement Learning",
                "authors": ["Frank"],
                "year": 2023
            },
            "abstractText": "Deep reinforcement learning applications.",
            "sections": [],
            "references": []
        }
        
        paper_path = config.papers_dir / "paper_4.json"
        with open(paper_path, 'w') as f:
            json.dump(new_paper, f)
        
        # Incremental index
        count2 = retriever.incremental_index(config.papers_dir, load_existing=True)
        
        assert count2 == 1
        assert retriever.index_manager.index.ntotal == count1 + count2


class TestSearchQuality:
    """Test search result quality."""
    
    def test_similar_papers_ranking(self, config):
        """Test that similar papers rank higher."""
        retriever = PaperRetriever(config)
        retriever.build_retrieval_index()
        
        # Query about deep learning - should match first paper well
        results = retriever.search_by_text("deep learning fundamentals", top_k=3)
        
        assert len(results) > 0
        # Top result should be about deep learning
        assert any("deep" in r["title"].lower() for r in results[:1])
    
    def test_threshold_filtering(self, config):
        """Test similarity threshold filtering."""
        retriever = PaperRetriever(config)
        retriever.build_retrieval_index()
        
        query = "abc xyz qwerty"  # Nonsensical query
        results = retriever.search_by_text(query, top_k=10, threshold=0.8)
        
        # With high threshold, should get few or no results
        assert all(r["score"] >= 0.8 for r in results)
