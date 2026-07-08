import pytest
import numpy as np

from project.retrieval.embeddings import EmbeddingGenerator
from project.utils.config import RetrieverConfig


@pytest.fixture
def config():
    """Create test configuration."""
    return RetrieverConfig(embedding_model="all-MiniLM-L6-v2")


@pytest.fixture
def generator(config):
    """Create embedding generator."""
    return EmbeddingGenerator(config)


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator."""
    
    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert generator.model is not None
    
    def test_generate_single_embedding(self, generator):
        """Test single embedding generation."""
        text = "Machine learning is a subset of artificial intelligence"
        embeddings = generator.generate_embeddings([text])
        
        assert embeddings.shape == (1, 384)
        assert embeddings.dtype == np.float32
    
    def test_generate_batch_embeddings(self, generator):
        """Test batch embedding generation."""
        texts = [
            "Deep learning",
            "Neural networks",
            "Computer vision"
        ]
        embeddings = generator.generate_embeddings(texts)
        
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32
    
    def test_normalization(self, generator):
        """Test embedding normalization."""
        texts = ["sample text"]
        embeddings = generator.generate_embeddings(texts, normalize=True)
        
        # Check L2 norm is 1
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
    
    def test_empty_texts(self, generator):
        """Test handling of empty text list."""
        embeddings = generator.generate_embeddings([])
        assert embeddings.shape == (0, 384)
    
    def test_similarity_computation(self, generator):
        """Test that similar texts have higher similarity."""
        texts = [
            "Deep neural networks for image classification",
            "Deep neural networks for image recognition",
            "How to cook a pizza"
        ]
        embeddings = generator.generate_embeddings(texts, normalize=True)
        
        # Compute cosine similarities
        sim_1_2 = np.dot(embeddings[0], embeddings[1])
        sim_1_3 = np.dot(embeddings[0], embeddings[2])
        
        # Similar texts should have higher similarity
        assert sim_1_2 > sim_1_3
