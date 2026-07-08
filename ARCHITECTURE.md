# Literature Retrieval Engine - Architecture Guide

## System Overview

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     PaperRetriever                                 │
│                  (Main Interface)                                  │
└────────────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   ┌─────────┐ ┌──────────────┐ ┌──────────────┐
   │DataLoader│ │Embedding    │ │IndexManager  │
   │          │ │Generator    │ │              │
   └─────────┘ └──────────────┘ └──────────────┘
       │            │              │
       │            │              ▼
       │            │         ┌──────────────┐
       │            │         │FAISS Index   │
       │            │         │(IndexFlatIP) │
       │            │         └──────────────┘
       ▼            ▼
   [JSON Files]  [ST Model]
```

## Component Details

### 1. DataLoader (`utils/data_loader.py`)

**Responsibility**: Load and validate paper data

**Key Classes**:
- `PaperData`: Dataclass holding parsed paper information
- `DataLoader`: Static methods for loading and validation

**Methods**:
```python
load_paper(file_path) -> Optional[PaperData]
  # Load single JSON file, handle errors gracefully

load_all_papers(papers_dir, recursive=True) -> Iterator[PaperData]
  # Recursively load all papers from directory
  # Returns iterator for memory efficiency

validate_paper(paper) -> bool
  # Validate year and text fields
```

**Features**:
- Recursive directory traversal
- JSON parsing with error handling
- Field validation and defaults
- Combined title + abstract text

### 2. EmbeddingGenerator (`retrieval/embeddings.py`)

**Responsibility**: Generate semantic embeddings

**Key Class**:
- `EmbeddingGenerator`: Manages SentenceTransformer model

**Methods**:
```python
generate_embeddings(texts, normalize=True) -> np.ndarray
  # Generate batch embeddings (384-dim)
  # Normalize using L2 norm for cosine similarity

_normalize_embeddings(embeddings) -> np.ndarray
  # L2 normalization to unit vectors

get_embedding_dim() -> int
  # Return embedding dimension (384)
```

**Features**:
- Batch processing for efficiency
- L2 normalization for cosine similarity
- GPU/CPU device selection
- Error handling for model loading
- Progress tracking

### 3. IndexManager (`retrieval/index_manager.py`)

**Responsibility**: Manage FAISS index lifecycle

**Key Class**:
- `IndexManager`: FAISS index operations

**Methods**:
```python
build_index(embeddings) -> None
  # Create new IndexFlatIP from embeddings

add_to_index(embeddings, metadata) -> None
  # Add embeddings incrementally

search(query_embedding, top_k, threshold) -> List[dict]
  # Search single query, apply threshold

batch_search(query_embeddings, top_k, threshold) -> List[List[dict]]
  # Search multiple queries

save_index() -> None
  # Persist index and metadata to disk

load_index() -> bool
  # Load index from disk

get_index_info() -> dict
  # Return index statistics
```

**Features**:
- FAISS IndexFlatIP (exact similarity)
- Incremental indexing support
- Metadata management with pickle
- Threshold filtering
- Batch search capabilities
- Disk persistence

### 4. PaperRetriever (`retrieval/retriever.py`)

**Responsibility**: Orchestrate entire retrieval pipeline

**Key Class**:
- `PaperRetriever`: Main API interface

**Methods**:
```python
build_retrieval_index(papers_dir) -> int
  # Complete pipeline: load -> embed -> index -> save

incremental_index(papers_dir, load_existing) -> int
  # Add new papers to existing index

search_by_text(query_text, top_k, threshold) -> List[dict]
  # Search with text query

search_by_embedding(query_embedding, top_k, threshold) -> List[dict]
  # Search with pre-computed embedding

batch_search(queries, top_k, threshold) -> List[List[dict]]
  # Multiple queries at once

load_retrieval_index() -> bool
  # Load index from disk

save_retrieval_index() -> None
  # Save index to disk

get_index_stats() -> dict
  # Return statistics
```

**Features**:
- High-level API
- Error handling and logging
- Configuration management
- Incremental updates
- Batch operations

## Data Flow

### Building Index

```
JSON Files
    ↓
[Load Papers]
    ↓ (PaperData)
[Validate]
    ↓ (Valid Papers)
[Extract Text]
    ↓ (Combined Text)
[Generate Embeddings]
    ↓ (Normalized Embeddings)
[Build FAISS Index]
    ↓ (IndexFlatIP)
[Save to Disk]
    ↓
(index.faiss, metadata.pkl)
```

### Searching

```
Query Text
    ↓
[Generate Query Embedding]
    ↓ (Normalized Vector)
[Load FAISS Index]
    ↓ (IndexFlatIP)
[Search (top_k, threshold)]
    ↓ (Indices & Scores)
[Fetch Metadata]
    ↓ (Paper Info)
[Format Results]
    ↓
List[Dict] Results
```

## Configuration System

### RetrieverConfig (`utils/config.py`)

```python
@dataclass
class RetrieverConfig:
    # Model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Index
    index_type: str = "IndexFlatIP"
    batch_size: int = 32
    
    # Paths
    papers_dir: Path
    knowledge_base_dir: Path
    index_path: Path
    metadata_path: Path
    log_file: Path
    
    # Retrieval
    default_top_k: int = 5
    similarity_threshold: float = 0.0
    
    # Performance
    use_gpu: bool = False
    num_workers: int = 4
```

## Logging Architecture

### Logger (`utils/logger.py`)

- Centralized logger creation
- Console + optional file output
- Singleton pattern for logger instances
- Configurable log levels

**Usage**:
```python
logger = Logger.get_logger(__name__)
logger.info("Message")
logger.debug("Debug info")
logger.error("Error occurred")
```

## Search Algorithm

### Similarity Computation

1. **Normalization**: All embeddings L2-normalized to unit vectors
2. **Distance Metric**: Inner product (IndexFlatIP) = cosine similarity
3. **Score Range**: [0, 1] for normalized vectors
4. **Ranking**: Results sorted by similarity score (descending)

### Search Process

```
Query: "neural networks"
    ↓
1. Embed: text → 384-dim vector
2. Normalize: ||vector|| = 1
3. Search: FAISS returns top_k matches
4. Score: similarity ∈ [0, 1]
5. Filter: score >= threshold
6. Return: top_k results with metadata
```

## Memory Efficiency

### Design Patterns

1. **Streaming**: DataLoader yields papers (iterator pattern)
2. **Batch Processing**: Process embeddings in batches
3. **Lazy Loading**: Load index only when needed
4. **Incremental Updates**: Don't rebuild entire index

### Storage

- **FAISS Index**: Binary format, memory-mapped access
- **Metadata**: Pickle format, efficient serialization
- **Papers**: Not stored after indexing

## Error Handling Strategy

### Levels

1. **File Level**: Try/except for file operations
2. **Data Level**: Validation and defaults
3. **Model Level**: Fallback and error logging
4. **Index Level**: Graceful degradation

### Approach

```python
try:
    # Operation
except SpecificException as e:
    logger.warning(f"Specific error: {e}")
    # Handle gracefully
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Re-raise for upstream handling
```

## Testing Strategy

### Test Organization

1. **test_retriever.py**: Integration and end-to-end tests
2. **test_embeddings.py**: Embedding generation and normalization
3. **test_index_manager.py**: FAISS operations and persistence

### Test Types

- **Unit Tests**: Individual components
- **Integration Tests**: Component interaction
- **Quality Tests**: Search ranking and filtering

## Performance Characteristics

### Complexity

- **Index Building**: O(n * d) where n = papers, d = embedding_dim
- **Search**: O(n) for exact search (IndexFlatIP)
- **Memory**: O(n * d) for embeddings + O(n * m) for metadata

### Optimization

- Batch embedding: ~32 papers/batch
- FAISS in-memory: <5ms per query
- Incremental: O(k) where k = new papers

## Future Extensions

### Index Types

- **IndexIVFFlat**: Approximate search for 1M+ papers
- **IndexHNSW**: Hierarchical graph for faster search
- **IndexPQ**: Product quantization for memory efficiency

### Features

- Multi-field search (title weight ≠ abstract weight)
- Filtering by metadata (year, authors, venue)
- Hybrid search (keyword + semantic)
- Real-time index updates

---

For detailed implementation, see the source code with comprehensive docstrings.
