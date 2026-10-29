# Changelog

## 0.1.0rc2

- Document metadata filters, sync/async graph-tool calls, parallel-edge traversal,
  and ownership/cleanup across graph, parent and media stores.
- Replace the Community graph dependency with Antfly-owned `GraphDocument`,
  `Node` and `Relationship` input types. Graph storage and tools use the base
  install; the `[graph]` extra remains accepted for existing install commands.
- Continue accepting structurally compatible graph documents; the graph store
  no longer inherits Community's `GraphStore`. Cypher remains unsupported.

## 0.1.0rc1

Initial release candidate for the Python Antfly integration packages.

- Vector CRUD, LangChain factories/retrievers, distance/relevance scores, filters,
  MMR and external or managed embedding modes.
- Dense text/image embeddings, full-text/hybrid retrieval, reranking and graph
  context.
- Persistent JSON, document and byte stores for parents, media and caches.
- Source-owned graph ingestion, scoped queries/traversals and bounded tools.
- LangGraph long-term memory with tuple namespaces, semantic search, ordered
  batches, stable timestamps and sync/async operation.
- A persistent PDF/image RAG example with strict evidence validation, transform
  recovery and restart support.
- Published `antfly-sdk>=0.2.1` dependency and checked source/wheel distributions.

See [Testing and compatibility](docs/validation.md) for supported versions and
[validation results](docs/validation-report.md) for scope and limitations.
