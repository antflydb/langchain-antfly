# Public API reference

Python 3.11+; use the exact supported dependency resolution in `uv.lock` for
examples. [Compatibility and validation](validation.md) distinguishes tested
versions from declared ranges. [Recipes](recipes.md) contain executable examples.
All constructors attach without provisioning. Caller-supplied clients remain
caller-owned. Unsupported or misspelled configuration fails rather than being ignored.

## langchain-antfly

| Class | Provisioning and principal methods |
| --- | --- |
| `AntflyVectorStore` | `create(client=..., table=..., ...)`; `from_texts`, `from_documents`, `afrom_texts`, `afrom_documents`; `validate`; add/delete/get; similarity and MMR search; `as_retriever`; explicit image methods |
| `AntflyEmbeddings` | `AntflyEmbeddings(client=..., model=..., dimensions=None, batch_size=64)`; `embed_documents`, `embed_query`; explicit `embed_images`, `embed_image_query`; async counterparts |
| `AntflyRetriever` | Supply `vectorstore=...` or `client=..., table=...`; `mode="dense"`, `"text"` or `"hybrid"`; optional `reranker`, `graph_store` plus `graph_expansion`; `invoke`/`ainvoke` |
| `AntflyReranker` | `AntflyReranker(client=..., model=..., top_n=None)`; `rerank`, `compress_documents`, `invoke({"query": ..., "documents": ...})`, `as_runnable`; async counterparts |
| `AntflyJSONStore` | `create(client=..., table=..., namespace="")`; `mget`, `mset`, `mdelete`, `yield_keys(prefix=...)`; async counterparts |
| `AntflyDocumentStore` | Same store methods; values are LangChain `Document` objects |
| `AntflyByteStore` | Same store methods; values are bytes; default `max_value_bytes=8388608` per decoded value |
| `AntflyGraphStore` | `create(client, table, namespace="", ...)`; `add_graph_documents`, `delete_source`, schema properties/refresh, `query`, `query_graph`, `traverse`, `expand_documents` |

Vector factories create missing resources by default and reuse compatible existing
resources; `create_if_missing=False` gives an attach-only factory. `create()` is
create-only. Explicit `ids` override `Document.id`; absent IDs are generated.
Stable IDs update existing records. `get_by_ids` returns found unique documents;
use their IDs rather than assuming LangChain guarantees requested order.
`delete(ids=[...])` is bounded explicit deletion, not a table-drop operation.

External embeddings require only the standard LangChain interface. Vector shape
and finite values are checked; unknown model provenance stays unknown. Managed
indexes attach without `embedding`; creating one requires an explicit `embedder`
configuration. Never assume equal dimensions prove semantic/model compatibility.

`similarity_search_with_score` preserves raw distance. Relevance scores use the
index metric's defined conversion; hybrid/RRF and reranker scores are provenance,
not probabilities. `metadata._antfly` is reserved for ranking/graph annotations.
Fixed/per-call retrieval filters are intersected. A filter is not write authorization.
Graph expansion rejects arbitrary vector filters it cannot enforce. See the
[metadata filter recipe](recipes.md#filter-by-document-metadata) for Antfly query
syntax, qualified field names and exact term matching.

Store `mget` preserves input order, duplicate keys, and aligned `None` entries for
missing keys. Store prefixes and namespaces are encoded reversibly. JSON/document
codecs are versioned; bytes are bounded and lossless. No default pickle is used:

```python
from langchain_antfly import AntflyJSONStore

settings = AntflyJSONStore.create(client=client, table="owned_settings", namespace="demo")
settings.mset([("theme", {"name": "orchard", "enabled": True})])
assert settings.mget(["theme", "absent", "theme"])[1] is None
assert list(settings.yield_keys(prefix="the")) == ["theme"]
settings.mdelete(["theme"])
```

### Graph input models

`GraphDocument`, `Node` and `Relationship` are exported from `langchain_antfly`.
They are Pydantic values for ingestion, with a required source `Document`.
See [migration from Community](recipes.md#migrate-from-community-graph-inputs)
for version requirements, compatibility and imports.

### Canonical graph JSON

`antfly-graph-json-v1` accepts named operations containing bounded `match`,
`traverse`, `shortest_path` or `k_shortest_paths` queries. Namespace/table scope and
configured depth/result limits are enforced; incomplete or mismatched evidence
fails closed. `query_graph(mapping)` returns projected rows with `_antfly_graph`
provenance, not the SDK response envelope. MATCH requires a server that accepts
explicit table-qualified aliases; no unsafe unqualified fallback is attempted.
The result alias `_antfly_graph` is reserved for provenance and rejected in MATCH declarations, bindings, and aggregate names.
All graph node filters intersect the namespace with `graph_kind="node"`; internal
source-ownership manifests are excluded from node enumeration and counts.
Explicit source nodes created by `include_source=True` remain ordinary graph
nodes. New tables index both `graph_namespace` and `graph_kind` as keyword fields.
Attaching never changes table schema. The tested native server can apply the kind
predicate as a residual over the indexed namespace on older adapter-created tables;
servers requiring full predicate coverage may require an explicit schema migration.
Graph filter conjunctions use the public `conjuncts` form; graph WHERE conjunctions
use `and`.

`query(json_string, params={...})` binds exact `{"$param": "name"}` values, with no
string interpolation. Missing, extra, and unused parameters raise errors. For an
existing graph created by the graph recipe:

```python
import json

rows = graph.query(json.dumps({
    "neighbors": {"index": "graph", "traverse": {
        "start": {"keys": [{"$param": "start"}]},
        "max_depth": 1, "limit": 10, "include_documents": True,
    }},
}), params={"start": graph.node_key("Person", "mira")})
```

`get_schema` and `get_structured_schema` are properties; call `refresh_schema()`
to update their declared/inferred discovery information. Source identity uses
`source.metadata.id`; supply it explicitly across text changes. Shared ownership
survives another source's deletion. Entity identity includes type plus ID;
relationship contributions are merged under source ownership. Cypher is unsupported.

`create_graph_tools(graph)` supplies bounded `antfly_graph_query` and
`antfly_graph_traverse` tools with sync/async invocation and the supplied graph's
fixed scope. Tool arguments cannot switch table or namespace. The query tool
requires a JSON string in `query`; `query_graph()` accepts a mapping. See
[tool invocation](recipes.md#invoke-graph-tools-directly),
[parallel relationship evidence](recipes.md#parallel-relationships-and-traversal-evidence),
and [cross-store cleanup](recipes.md#own-graph-parent-and-media-cleanup-explicitly).

## langgraph-store-antfly

`AntflyStore(client=..., table=..., index=None, index_name="memory_vectors",
page_size=100)` attaches; `AntflyStore.create(client, table, ...)` provisions.
Use LangGraph `put/get/delete`, `search`, `list_namespaces`, ordered `batch`, and
async equivalents. Namespaces are tuples of components, so `("user", "1")` and
`("user", "10")` remain distinct. Tools must derive them from trusted invocation
context; they do not replace server/application authorization.

`index={"dims": ..., "embed": embedding, "fields": [...]}` enables semantic search.
Omit the index for embedding-free CRUD/enumeration; `index=False` on a put opts
that item out of semantic indexing. Creation timestamps survive overwrites;
updates monotonically advance modification timestamps. Ordered batches are not
advertised as atomic across all operations. `supports_ttl=False`; TTL requests
are rejected. No execution checkpointer is provided.

Async bridges have bounded work. Coroutine cancellation does not roll back an
already-submitted write. Inspect typed write outcomes and committed-key progress;
a lost or malformed response can mean unknown outcome. Do not blindly replay
ambiguous mutations. The [failure matrix](validation.md) describes the checks.
