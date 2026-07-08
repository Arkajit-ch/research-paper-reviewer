"""
Autonomous AI Paper Reviewer - Literature Retrieval Engine

A production-grade retrieval system for querying similar research papers
using embeddings and FAISS similarity search.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"

from .retrieval.retriever import PaperRetriever
from .utils.config import RetrieverConfig

__all__ = ["PaperRetriever", "RetrieverConfig"]
