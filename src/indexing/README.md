# Document Indexing Module

## Overview

This module provides document indexing implementations for creating searchable indexes from text documents. It demonstrates two fundamental indexing approaches used in LLM applications:

1. **Vector Store Indexing**: Similarity search using embeddings
2. **Property Graph Indexing**: Knowledge graphs for relationship queries

## Core Concepts

### Vector Store Index
Converts documents into high-dimensional vectors (embeddings) that capture semantic meaning. Enables finding similar documents based on meaning rather than exact keyword matches.

**Use when:**
- You need semantic similarity search
- Finding documents "similar in meaning" to a query
- Building RAG (Retrieval-Augmented Generation) pipelines
- Questions like "What documents discuss similar topics?"

### Property Graph Index
Extracts entities and relationships from documents to build a knowledge graph. Enables traversing relationships between concepts.

**Use when:**
- You need to understand relationships between entities
- Answering "who knows whom" or "what relates to what"
- Building knowledge bases with connected information
- Questions like "What companies did this person work for?"

## Usage Examples

### Vector Store Indexing

```python
from src.indexing import VectorStoreIndexer
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding

# Create documents
documents = [
    Document(text="RAG combines retrieval with generation"),
    Document(text="Embeddings capture semantic meaning"),
    Document(text="Vector databases enable similarity search")
]

# Create indexer
indexer = VectorStoreIndexer(
    show_progress=False,
    insert_batch_size=2048  # Batch size for memory efficiency
)

# Build index
embed_model = MockEmbedding(embed_dim=256)
index = indexer.create_index(documents, embed_model=embed_model)

# Use index for retrieval
retriever = index.as_retriever()
results = retriever.retrieve("What is RAG?")
```

### Property Graph Indexing

```python
from src.indexing import PropertyGraphIndexer
from src.mocks import TrajectoryMockLLMLlamaIndex

# Create documents with relationships
documents = [
    Document(text="Alice is the CEO of TechCorp"),
    Document(text="Bob reports to Alice"),
    Document(text="TechCorp acquired StartupAI in 2023")
]

# Without LLM: Basic relationship extraction
basic_indexer = PropertyGraphIndexer(
    show_progress=False,
    embed_kg_nodes=False
)
basic_index = basic_indexer.create_index(documents)

# With LLM: Enhanced entity extraction
mock_llm = TrajectoryMockLLMLlamaIndex(chain=["Entities: Alice, CEO, TechCorp"])
enhanced_indexer = PropertyGraphIndexer(
    llm=mock_llm,
    show_progress=False
)
enhanced_index = enhanced_indexer.create_index(documents)
```

## Testing Strategies

### Testing Without External Services

Both indexers work with mock embeddings and LLMs for testing:

```python
from llama_index.core.embeddings import MockEmbedding
from src.mocks import TrajectoryMockLLMLlamaIndex

# Mock embedding for vector indexing
mock_embed = MockEmbedding(embed_dim=256)

# Mock LLM for entity extraction
mock_llm = TrajectoryMockLLMLlamaIndex(
    chain=["Expected LLM response 1", "Response 2"]
)

# Test indexing without API calls
indexer = VectorStoreIndexer()
index = indexer.create_index(documents, embed_model=mock_embed)
```

### Input Validation

Both indexers use Pydantic's `@validate_call` decorator for automatic validation:

```python
# This will raise ValidationError
indexer.create_index("not a list")  # ❌ String instead of list
indexer.create_index(None)          # ❌ None instead of list

# These work correctly
indexer.create_index([])            # ✅ Empty list is valid
indexer.create_index([doc1, doc2])  # ✅ List of documents
```

## Architecture

### Class Hierarchy

```
DocumentIndexer (Abstract Base)
├── VectorStoreIndexer
└── PropertyGraphIndexer
```

### Key Design Decisions

1. **Abstract Base Class**: `DocumentIndexer` defines the interface all indexers must implement
2. **Pydantic Validation**: Automatic input validation without manual checks
3. **Mock Support**: All indexers work with mock LLMs and embeddings for testing
4. **Helper Functions**: Complex logic extracted into testable functions (e.g., `_select_extractors`)

## Configuration Options

### VectorStoreIndexer

- `show_progress`: Display indexing progress bar
- `insert_batch_size`: Documents per batch (affects memory usage)

### PropertyGraphIndexer

- `llm`: Optional LLM for entity extraction
- `kg_extractors`: Custom extractors for relationship extraction
- `embed_kg_nodes`: Whether to create embeddings for graph nodes
- `show_progress`: Display indexing progress

## Performance Considerations

### Memory Usage

- **VectorStoreIndexer**: Use smaller `insert_batch_size` for large document sets
- **PropertyGraphIndexer**: Set `embed_kg_nodes=False` to reduce memory usage

### Processing Time

- Vector indexing: O(n) with document count
- Graph extraction with LLM: Slower but richer relationships
- Graph extraction without LLM: Faster but basic relationships

## References

- [LlamaIndex Vector Store Index](https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/)
- [LlamaIndex Property Graph Index](https://docs.llamaindex.ai/en/stable/module_guides/indexing/property_graph_index/)
- [Understanding Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Knowledge Graphs Explained](https://neo4j.com/blog/what-is-knowledge-graph/)

## Summary

This module provides two complementary indexing approaches:

- **VectorStoreIndexer**: For semantic similarity search
- **PropertyGraphIndexer**: For relationship and entity queries

Both implementations:
- Follow the same `DocumentIndexer` interface
- Support testing without external services
- Include automatic input validation
- Provide clear, concrete examples

Choose based on your use case: similarity search (Vector) or relationship queries (Graph).