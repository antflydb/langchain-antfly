# Integration interfaces

| Component | Public interface | Examples and tests |
| --- | --- | --- |
| `AntflyVectorStore` | LangChain vector store: CRUD, factories, scores, filters, MMR, sync/async | [Quickstart](../examples/quickstart.py), [contracts](../tests/test_vectorstores.py) |
| `AntflyRetriever` | Full-text/hybrid retrieval, reranking and bounded graph expansion | [Recipes](recipes.md), [contracts](../tests/test_retrievers.py) |
| `AntflyEmbeddings` | Standard text embeddings plus explicit image extensions | [Native image example](../examples/native_multimodal.py), [contracts](../tests/test_embeddings.py) |
| `AntflyReranker` | Reranker, document compressor and runnable | [Recipes](recipes.md), [contracts](../tests/test_rerankers.py) |
| JSON/document/byte stores | LangChain key/value storage, parent documents, media and embedding caches | [Multimodal walkthrough](../examples/multimodal/README.md), [contracts](../tests/test_stores.py) |
| `AntflyGraphStore` and graph tools | GraphDocument ingestion, schema, scoped Antfly JSON graph operations | [API reference](api-reference.md), [contracts](../tests/test_graphs.py) |
| `AntflyStore` | LangGraph long-term memory: namespaces, batches, filtering/search, sync/async | [Agent example](../examples/agent_memory.py), [contracts](../tests/test_memory.py) |

Client ownership, write outcomes, serialization, graph scope and cancellation
semantics are described in the [API reference](api-reference.md). For tested
versions and execution profiles, see [Testing and compatibility](validation.md).
