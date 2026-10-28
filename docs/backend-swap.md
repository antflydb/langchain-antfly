# Replace in-memory stores with Antfly

The backend-swap example compares LangChain's built-in `InMemoryVectorStore`
and `InMemoryStore` with persistent `AntflyVectorStore` and
`AntflyDocumentStore`. Both paths consume the same documents, questions and
standard `Embeddings` object. A task-aware cache supplies identical vectors to
both, keeping retrieval differences separate from embedding differences.

The application change is resource construction:

```python
# In-memory baseline
vectors = InMemoryVectorStore.from_documents(children, embedding)
parents = InMemoryStore()

# Persistent Antfly backend
vectors = AntflyVectorStore.from_documents(
    children, embedding, client=client, table=prefix + "_vectors",
    distance_metric="cosine",
)
parents = AntflyDocumentStore.create(client=client, table=prefix + "_parents")
```

Both then call `as_retriever().invoke(...)` and the same parent-store operations.
The comparison checks exact child/parent identities, content and source metadata.
An Antfly constructor attaches to persisted data on a later process; the in-memory
baseline must rebuild its state.

## Run the exercise

Start the server and install the dependencies as described in the
[quickstart](../README.md#first-result). Use a fresh prefix beginning
`backend_swap_`; the example refuses to overwrite existing tables.

```bash
mkdir -p .state/backend-swap
uv run python -m examples.backend_swap compare \
  --url http://localhost:8080 --model BAAI/bge-small-en-v1.5:onnx \
  --prefix backend_swap_example \
  --cache .state/backend-swap/cache.json \
  --output .state/backend-swap/compare.json

# Fresh process; optionally restart Antfly with the same data directory first.
uv run python -m examples.backend_swap attach \
  --url http://localhost:8080 --prefix backend_swap_example \
  --cache .state/backend-swap/cache.json \
  --output .state/backend-swap/attach.json

uv run python -m examples.backend_swap cleanup \
  --url http://localhost:8080 --prefix backend_swap_example
```

`ANTFLY_URL` and `ANTFLY_EMBEDDING_MODEL` can supply defaults; authentication uses
`ANTFLY_TOKEN`. The cache contains vectors and synthetic text, not credentials.
Attach uses the cached query vectors without making inference calls. Missing
parents or cache entries fail explicitly.

The in-memory baseline uses no external storage service; Antfly introduces one
server. This exercise demonstrates persistence and API migration, not reduced
service count, production retrieval quality or a general latency benchmark.
