# Research Paper Reviewer - Literature Retrieval Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade literature retrieval engine for the **Autonomous AI Paper Reviewer** project. This module automatically retrieves the most similar research papers using semantic embeddings and FAISS similarity search.

## 🎯 Overview

This project focuses **ONLY** on building a robust retrieval engine for AI/ML/CS papers. It does NOT implement:
- Novelty evaluation
- LLM-based reviewers
- UI/Frontend
- Paper modification

## 🏗️ Architecture

```
Research Paper Retrieval Pipeline:

┌──────────────────────────────────────┐
│  JSON Paper Files   │
└──────────────────────┬────────────────┘
           ↓
┌──────────────────────────────────────┐
│  Data Loader        │  Extract: title, abstract, authors, year, refs
│  (Validation)       │
└──────────────────────┬────────────────┘
           ↓
┌──────────────────────────────────────┐
│ Embedding Generator │  Model: all-MiniLM-L6-v2 (384-dim)
│ (SentenceTransform) │
└──────────────────────┬────────────────┘
           ↓
┌──────────────────────────────────────┐
│  FAISS Index        │  IndexFlatIP (Cosine Similarity)
│  (Build/Update)     │  Normalized embeddings
└──────────────────────┬────────────────┘
           ↓
┌──────────────────────────────────────┐
│  Similarity Search  │  Top-K retrieval with scoring
│  (Query Processing) │
└──────────────────────────────────────┘
```

## ✨ Features

✅ **Recursive Data Loading**: Automatically loads all JSON papers from nested directories

✅ **Semantic Embeddings**: Uses SentenceTransformer (all-MiniLM-L6-v2) for 384-dim embeddings

✅ **FAISS Indexing**: IndexFlatIP for efficient cosine similarity search

✅ **Batch Processing**: Supports batch embedding and batch search operations

✅ **Incremental Indexing**: Add new papers without rebuilding entire index

✅ **Metadata Management**: Stores paper metadata (title, authors, year, references)

✅ **Comprehensive Logging**: Detailed logging at every step

✅ **Production Ready**: Full type hints, error handling, and unit tests

✅ **Threshold Filtering**: Filter results by similarity threshold

## 📦 Installation

### Prerequisites
- Python 3.11+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/Arkajit-ch/research-paper-reviewer.git
cd research-paper-reviewer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Prepare Your Data

Place JSON paper files in `sample_data/papers/`:

```json
{
  "name": "paper_1.json",
  "metadata": {
    "title": "Deep Learning Fundamentals",
    "authors": ["Alice", "Bob"],
    "year": 2020
  },
  "abstractText": "This paper discusses the fundamentals of deep learning.",
  "sections": [{"heading": "...", "text": "..."}],
  "references": ["ref1", "ref2"]
}
```

### 2. Build Index

```bash
python project/main.py --mode build --papers-dir ./sample_data/papers
```

### 3. Search Papers

```bash
python project/main.py --mode search --query "neural networks" --top-k 5
```

### 4. View Statistics

```bash
python project/main.py --mode stats
```

## 💻 API Usage

### Basic Usage

```python
from project import PaperRetriever, RetrieverConfig
from pathlib import Path

# Initialize with custom config
config = RetrieverConfig(
    papers_dir=Path("sample_data/papers"),
    embedding_model="all-MiniLM-L6-v2",
    default_top_k=5
)

retriever = PaperRetriever(config)

# Build index
count = retriever.build_retrieval_index()
print(f"Indexed {count} papers")
```

### Search by Text

```python
# Search for similar papers
query = "attention mechanisms in transformer models"
results = retriever.search_by_text(query, top_k=5)

for result in results:
    print(f"[{result['rank']}] {result['title']}")
    print(f"    Score: {result['score']:.4f}")
    print(f"    Year: {result['year']}")
    print(f"    Authors: {', '.join(result['authors'])}")
    print()
```

### Batch Search

```python
# Search for multiple queries
queries = [
    "deep learning architectures",
    "natural language processing",
    "computer vision applications"
]

results = retriever.batch_search(queries, top_k=3)

for query, query_results in zip(queries, results):
    print(f"Query: {query}")
    for r in query_results:
        print(f"  - {r['title']} ({r['score']:.4f})")
```

### Incremental Indexing

```python
# Load existing index
retriever.load_retrieval_index()

# Add new papers
new_count = retriever.incremental_index(
    papers_dir=Path("new_papers"),
    load_existing=True
)
print(f"Added {new_count} new papers")
```

### Search by Embedding

```python
# Generate custom query embedding
query_text = "machine learning algorithms"
query_embedding = retriever.embedding_generator.generate_embeddings(
    [query_text],
    normalize=True
)[0]

# Search with embedding
results = retriever.search_by_embedding(query_embedding, top_k=5)
```

## 📊 Output Format

Search results are returned as a list of dictionaries:

```python
[
    {
        "rank": 1,
        "score": 0.8234,
        "paper_id": "123.json",
        "title": "Deep Learning for NLP",
        "year": 2022,
        "authors": ["John Doe", "Jane Smith"],
        "reference_count": 42,
        "file_path": "/path/to/paper.json",
        "abstract": "This paper explores..."
    },
    # ... more results
]
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_retriever.py -v

# Run with coverage
pytest tests/ --cov=project --cov-report=html
```

### Test Coverage

- ✅ Retriever initialization and configuration
- ✅ Index building and persistence
- ✅ Text-based and embedding-based search
- ✅ Batch search operations
- ✅ Incremental indexing
- ✅ Embedding generation and normalization
- ✅ FAISS index operations
- ✅ Search quality and ranking
- ✅ Threshold filtering

## 📁 Project Structure

```
research-paper-reviewer/
├── project/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py        # Main retrieval engine
│   │   ├── embeddings.py       # Embedding generation
│   │   └── index_manager.py    # FAISS index management
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   ├── logger.py           # Logging setup
│   │   └── data_loader.py      # Paper data loading
│   └── knowledge_base/         # Indexed data (generated)
│       ├── faiss.index
│       └── metadata.pkl
├── tests/
│   ├── __init__.py
│   ├── test_retriever.py       # Retriever tests
│   ├── test_embeddings.py      # Embedding tests
│   └── test_index_manager.py   # Index manager tests
├── sample_data/
│   └── papers/                 # Sample papers (user-provided)
├── logs/                       # Log files (generated)
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## 🔧 Configuration

Customize behavior via `RetrieverConfig`:

```python
from project import RetrieverConfig
from pathlib import Path

config = RetrieverConfig(
    # Model configuration
    embedding_model="all-MiniLM-L6-v2",
    embedding_dim=384,
    batch_size=32,
    
    # Paths
    papers_dir=Path("sample_data/papers"),
    knowledge_base_dir=Path("project/knowledge_base"),
    index_path=Path("project/knowledge_base/faiss.index"),
    metadata_path=Path("project/knowledge_base/metadata.pkl"),
    log_file=Path("logs/retriever.log"),
    
    # Retrieval parameters
    default_top_k=5,
    similarity_threshold=0.0,
    
    # Performance
    use_gpu=False,
    num_workers=4
)
```

## 📄 Supported Paper Format

The engine expects papers in the following JSON structure:

```json
{
  "name": "string",
  "metadata": {
    "title": "string",
    "authors": ["string"],
    "year": integer
  },
  "abstractText": "string",
  "sections": [
    {
      "heading": "string",
      "text": "string"
    }
  ],
  "references": ["string"]
}
```

## 🎯 Use Cases

1. **Literature Review**: Find related papers on a topic
2. **Paper Recommendation**: Suggest similar papers to authors
3. **Duplicate Detection**: Identify papers with similar content
4. **Research Gap Analysis**: Find areas not well covered
5. **Citation Analysis**: Discover highly-referenced topics

## ⚙️ Performance Optimization

### Embedding Generation
- Batch processing: Process multiple papers in parallel
- GPU acceleration: Enable with `use_gpu=True` in config
- Model selection: Use smaller models for faster inference

### FAISS Indexing
- IndexFlatIP: Exact similarity search (optimal for < 1M papers)
- Normalized vectors: Ensures cosine similarity in [0, 1] range
- In-memory index: Fast retrieval with lower latency

### Search Optimization
- Threshold filtering: Skip low-similarity results
- Top-K limiting: Reduce memory usage for large result sets
- Batch queries: Process multiple searches efficiently

## 🔐 Error Handling

The engine includes comprehensive error handling:

- Invalid JSON files are logged and skipped
- Missing required fields are handled gracefully
- Network errors during model download are caught
- Index corruption is detected and reported

## 📝 Logging

All operations are logged with detailed information:

```
2024-01-15 10:30:45 - project.retrieval.retriever - INFO - PaperRetriever initialized
2024-01-15 10:30:46 - project.utils.data_loader - INFO - Loading papers from ./papers...
2024-01-15 10:30:47 - project.retrieval.embeddings - INFO - Generating embeddings for 100 texts...
2024-01-15 10:31:05 - project.retrieval.index_manager - INFO - Building FAISS index for 100 papers...
```

## 🚀 CLI Commands

### Build Index
```bash
python project/main.py --mode build \
  --papers-dir ./sample_data/papers \
  --config-model all-MiniLM-L6-v2
```

### Search
```bash
python project/main.py --mode search \
  --query "neural networks in computer vision" \
  --top-k 10
```

### Incremental Update
```bash
python project/main.py --mode incremental \
  --papers-dir ./new_papers
```

### Show Statistics
```bash
python project/main.py --mode stats
```

## 📈 Scalability

The engine is designed to handle:
- ✅ Thousands of papers
- ✅ Multiple concurrent searches
- ✅ Incremental updates
- ✅ Large-scale batch operations

## 🔄 Future Enhancements

- [ ] GPU-accelerated FAISS indexing
- [ ] Distributed indexing for large corpora
- [ ] Advanced filtering (year, authors, venues)
- [ ] Multi-field search (title vs. abstract weighting)
- [ ] Index compression and approximate search
- [ ] Real-time index updates

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For issues and questions:
- Open an [Issue](https://github.com/Arkajit-ch/research-paper-reviewer/issues)
- Check existing discussions
- Review documentation

## 🙏 Acknowledgments

- [SentenceTransformers](https://www.sbert.net/) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [PeerRead](https://github.com/allenai/PeerRead) for inspiration

---

**Note**: This is the foundation for the Autonomous AI Paper Reviewer project. Future modules will build upon this retrieval engine to add review generation, novelty evaluation, and other capabilities.
