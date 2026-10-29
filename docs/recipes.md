# Integration recipes

Use the same caller-owned `AntflyClient` in all adapters. The quickstart and
multimodal CLI provide executable setups; snippets below assume declared tables,
indexes, models, and permissions exist. They do not implicitly install models.

## Hybrid retrieval and standalone reranking

```python
from langchain_antfly import AntflyRetriever, AntflyReranker

reranker = AntflyReranker(client=client, model="your-installed-reranker", top_n=3)
retriever = AntflyRetriever(vectorstore=store, mode="hybrid", k=3,
                           fetch_k=10, reranker=reranker)
documents = retriever.invoke("Who handles shipment escalation?")
documents_async = await retriever.ainvoke("Who handles shipment escalation?")
reranked = reranker.compress_documents(candidate_documents, "shipment escalation")
```

Hybrid retrieval needs a compatible full-text index in addition to the vector
index. Provide `full_text_index` for a named full-text index. Fixed adapter filters
are intersected with per-call filters. Raw vector scores remain distances; hybrid
ranking and rerank provenance live in reserved metadata and are not normalized
probabilities. Standalone rerank sorts the server's input-order scores while
preserving original document identities and user metadata.

## Filter by document metadata

Vector and hybrid retrieval accept Antfly query objects. Metadata is stored under
`metadata` by default, so qualify the field with that prefix:

```python
from langchain_core.documents import Document

store.add_documents([
    Document(id="europa", page_content="A green locket was found on Europa.",
             metadata={"station": "Europa"}),
    Document(id="titan", page_content="A blue locket was found on Titan.",
             metadata={"station": "Titan"}),
])
station_filter = {"term": "Europa", "field": "metadata.station"}
found = store.similarity_search("lost locket", k=2, filter=station_filter)
assert {document.id for document in found} == {"europa"}
found_async = await store.asimilarity_search("lost locket", k=2, filter=station_filter)
assert {document.id for document in found_async} == {"europa"}
```

This example assumes an otherwise empty store. Term queries match exact values:
`"europa"` does not match `"Europa"`. Use your configured `metadata_field` prefix
if you changed it. Shorthand mappings such as `{"station": "Europa"}` and nested
forms such as `{"term": {"station": "Europa"}}` are not Antfly query syntax.
See [Antfly query syntax](https://github.com/antflydb/antfly/blob/main/specs/openapi/antfly/query.yaml) for
other operators. Constructor filters and per-call filters are intersected; they
do not grant write authorization or replace server access control.

## Embedding cache

```python
from langchain_antfly import AntflyByteStore
from langchain_classic.embeddings import CacheBackedEmbeddings

document_cache = AntflyByteStore.create(
    client=client, table="embedding_cache", namespace="provider-model-revision/documents"
)
query_cache = AntflyByteStore(
    client=client, table="embedding_cache", namespace="provider-model-revision/queries"
)
cached = CacheBackedEmbeddings.from_bytes_store(
    embedding, document_cache, namespace="provider-model-revision",
    query_embedding_cache=query_cache, key_encoder="sha256",
)
vectors = cached.embed_documents(["A stable sentence."])
# Repeating the same text reuses durable cached embeddings.
```

Keep document and query caches in separate namespaces: providers may embed the
same text differently for retrieval documents and retrieval queries. Sharing a
cache with `query_embedding_cache=True` would erase that distinction.
Change the namespace when model identity, revision, task instructions, dimensions,
or normalization changes. The adapter cannot infer semantic compatibility from
vector dimensions. ByteStore caches are separate from the media ByteStore and
from JSON/document stores. Stored values use versioned codecs, not Python pickle.

## Graph documents and canonical graph queries

The graph input types below are available in this source checkout and the next
release. Published `0.1.0rc1` still uses the Community imports described in the
[migration note](#migrate-from-community-graph-inputs).

Provision a graph table using the integration-owned input types:

```python
from langchain_antfly import AntflyGraphStore
from langchain_antfly import GraphDocument, Node, Relationship
from langchain_core.documents import Document

graph = AntflyGraphStore.create(client=client, table="report_graph", namespace="orchard")
owner = Node(id="mira", type="Person", properties={"name": "Mira Chen"})
report = Node(id="q3", type="Report", properties={"title": "Q3 review"})
graph.add_graph_documents([GraphDocument(
    nodes=[owner, report],
    relationships=[Relationship(source=owner, target=report, type="OWNS")],
    source=Document(id="orchard-source", metadata={"id": "orchard-source"},
                    page_content="Mira owns the Q3 review."),
)], include_source=True)
graph.refresh_schema()
print(graph.get_schema)
rows = graph.traverse([graph.node_key("Person", "mira")], max_depth=1, limit=10)
```

The graph store exposes `antfly-graph-json-v1`, not Cypher. `query()` accepts JSON
with declared parameter binding; `query_graph()` accepts canonical mappings.
Namespace filters, traversal depth, result limits, and incomplete-response handling
apply to all operations. Do not pass this store to a chain that unconditionally
generates Cypher. Repeated source ingestion updates that source's contribution;
`delete_source()` removes owned contributions while retaining shared provenance.

`create_graph_tools(graph)` from `langchain_antfly.tools` returns bounded read-only
query and traversal tools. Supply a graph store with the intended fixed namespace
and limits before exposing those tools to a model. Tool arguments cannot select a
new store/table/namespace. Persist the graph source's `metadata.id` explicitly when
you need stable update/removal identity across source text changes.

### Migrate from Community graph inputs

Graph storage, tools and input models now use the base `langchain-antfly` install.
Replace `from langchain_community.graphs.graph_document import ...` with:

```python
from langchain_antfly import GraphDocument, Node, Relationship
```

Fields retain their familiar names. `GraphDocument.source` is required because
source identity owns graph contributions. Nodes accept string/integer IDs;
`properties` defaults to an independent empty dictionary. These Pydantic models
support `model_dump()` and `model_validate()`.

`add_graph_documents()` also accepts existing objects exposing `nodes`,
`relationships` and a source `Document`, with compatible node/relationship fields.
Already-extracted Community or Neo4j graph documents need no conversion. Extraction
providers remain responsible for their own dependencies. `AntflyGraphStore` no
longer inherits Community's `GraphStore`; code relying on that exact base-class
identity must use the Antfly API directly. No stored-data migration is needed.
The `[graph]` extra remains accepted for existing installation commands and adds
no dependencies. Published `0.1.0rc1` is unchanged; use the source checkout until
the next release is published.

[Community's sunset guidance](https://github.com/langchain-ai/langchain-community/issues/674)
recommends provider-owned packages. LangChain Core does not currently export
these graph contracts; the maintained
[Neo4j graph types](https://github.com/langchain-ai/langchain-neo4j/tree/main/libs/neo4j/langchain_neo4j/graphs)
belong to a database-specific integration. Antfly owns its input models and graph
operations rather than requiring another database integration for base types.

### Invoke graph tools directly

Direct `query_graph()` calls take a Python mapping. The query **tool** takes a
`query` argument containing a JSON string; encode the mapping with `json.dumps`:

```python
import json
from langchain_antfly import create_graph_tools

tools = {tool.name: tool for tool in create_graph_tools(graph)}
key = graph.node_key("Person", "mira")
query = {"neighbors": {"index": "graph", "traverse": {
    "start": {"keys": [key]}, "max_depth": 1, "limit": 10,
    "include_documents": True,
}}}
rows = graph.query_graph(query)
rows_via_tool = tools["antfly_graph_query"].invoke({"query": json.dumps(query)})
rows_async = await tools["antfly_graph_query"].ainvoke({"query": json.dumps(query)})
traversed = tools["antfly_graph_traverse"].invoke(
    {"keys": [key], "max_depth": 1, "limit": 10}
)
traversed_async = await tools["antfly_graph_traverse"].ainvoke(
    {"keys": [key], "max_depth": 1, "limit": 10}
)
```

These async snippets run inside an async function or notebook. The store and its
client must remain open until awaited calls finish. Tool arguments cannot switch
the table or namespace supplied when constructing the graph.

### Parallel relationships and traversal evidence

Traversal discovers reachable nodes; it is not an enumeration of every edge or
every possible path. On the tested Antfly `0.3.0-rc.5` server, two relationship
types joining the same nodes produce one traversal result for that destination,
with one selected path. Do not infer that another relationship was deleted or
that the selected path represents all evidence.

To inspect one type, use `graph.traverse([key], edge_types=["OWNS"], max_depth=1)`.
To enumerate relationship matches, use canonical MATCH instead. For a graph with
`ASSERTS` and `REFUTES` relationships between the same nodes:

```python
matches = graph.query_graph({"claims": {
    "match": {"anchor": "left", "nodes": {"left": {}, "right": {}},
              "edges": [{"from": "left", "to": "right",
                         "types": ["ASSERTS", "REFUTES"]}]},
    "return": {"bindings": ["left", "right"], "limit": 20},
}})
```

MATCH can return repeated endpoint bindings for distinct relationships; preserve
row/provenance evidence rather than deduplicating rows by destination. Limits
still apply. Do not rely on which parallel edge traversal selects or on row order.

### Own graph, parent and media cleanup explicitly

`delete_source(source_id)` removes that source's graph contributions and preserves
other sources' contributions. It does not delete separate document/media-store
records. Keep an application-owned manifest of parent IDs and media keys for each
source. When each source owns distinct parent/media keys, remove them together:

```python
source_id = "orchard-q3"
owned_parent_ids = ["orchard-q3/report", "orchard-q3/chart"]
owned_media_keys = ["orchard-q3/chart.png"]

# graph, parents and media are scoped stores constructed by this application.
graph.delete_source(source_id)
parents.mdelete(owned_parent_ids)
media.mdelete(owned_media_keys)
assert parents.mget(owned_parent_ids) == [None, None]
assert media.mget(owned_media_keys) == [None]
```

The sequence is not a cross-store transaction. Serialize it with ingestion for
that source, retain the manifest until all steps succeed, and hide removed sources
from readers while cleanup is pending. Shared media/parents need reference
tracking: delete a shared key only after its last owner is removed. A failure
between steps can leave stale or missing references; readers must detect missing
parents/media before generation. Inspect uncertain write outcomes before retrying.

## Agent tools in a running graph

`examples.retrieval_agent.build_agent(retriever, existing_chat_model)` creates an
actual LangChain agent with a typed `search_reports` tool. Invoke or `ainvoke` the
agent with messages. The tool calls the retriever and labels evidence by document
ID. Use an existing tool-capable chat provider, not a newly implemented Antfly
chat adapter.

After the quickstart, run the executable recipe with an installed existing chat
provider and its documented credential environment:

```bash
uv run --with langchain-openai python -m examples.retrieval_agent \
  --model openai:your-tool-capable-model 'Who owns operations?'
```

Add `--async` for async graph invocation. This explicitly makes live provider
calls and can incur charges. Inspect tool messages when evaluating tool selection;
an answer without a retrieval-tool call does not establish the runtime gate.

## Agent memory through ToolRuntime.store

`examples.agent_memory` runs `create_agent(store=AntflyStore(...))`. Its tools
derive namespaces from a trusted `UserContext` supplied by the application; the
model cannot select a different user namespace in tool arguments. Install the
existing chat integration for the provider you choose, then run separate processes:

```bash
uv run --with langchain-openai python -m examples.agent_memory --model openai:your-model \
  --user alice 'Remember preference key style with value concise'
# Restart the application/server, keeping the same Antfly data directory.
uv run --with langchain-openai python -m examples.agent_memory --model openai:your-model \
  --user alice 'Recall preference key style'
uv run --with langchain-openai python -m examples.agent_memory --model openai:your-model \
  --user bob 'Recall preference key style'
```

Bob must not see Alice's preference. Repeat with `--async` to exercise async graph
invocation. Provider calls are live and may be billed. The provider must support
tools; refusal or failure to call a tool is a model/example failure, not proof of
memory persistence. Capture tool messages when validating the workflow.

This small recipe treats keys as literal user-specified names. The agent calls a
tool once and then reports the result; it must not search guessed alternative keys
or invent missing memory. A bounded recursion limit should surface model tool
loops. The validation run observed such a loop with a vague key request before
making this example contract explicit; raising the loop budget was not the fix.

For direct memory use:

```python
from langgraph_store_antfly import AntflyStore

memory = AntflyStore.create(client=client, table="memories")
memory.put(("users", "alice"), "style", {"value": "concise"})
assert memory.get(("users", "alice"), "style").value == {"value": "concise"}
assert memory.get(("users", "bob"), "style") is None
```

Optional semantic memory indexing uses LangGraph's `index={"dims": ..., "embed":
embedding, "fields": [...]}` configuration. Without an index, CRUD and namespace
enumeration make no embedding calls. TTL is not advertised. Direct store calls
alone do not establish the runtime/restart acceptance gate above.

## Executable hybrid, reranking and graph-context companion

After the multimodal example has persisted `orchard_demo`, select an installed
reranker explicitly through `ANTFLY_RERANK_MODEL` (discover available names in
`/ai/v1/models`). Keep its original `ANTFLY_EMBEDDING_MODEL` configuration:

```bash
uv run python -m examples.companion_retrieval --prefix orchard_demo \
  --graph-table orchard_companion_graph --output companion.json run
uv run python -m examples.companion_retrieval --prefix orchard_demo \
  --graph-table orchard_companion_graph cleanup
```

The recipe reads the same persisted children, parents and media, validates their
references, and creates only the dedicated graph table. It runs full-text search,
hybrid search, standalone reranking, hybrid reranking, and sync/async graph-context
retrieval with reranking. Graph nodes preserve the corpus parent/source IDs and
link the report's related evidence. An explicit seed mapper translates retrieved
child provenance into graph node identities; bounded traversal keeps the graph
store's namespace and table checks intact. No MATCH compatibility fallback or
paid caption/generation call is involved.

Graph expansion preserves retrieved seed documents and appends bounded unique
neighbors before final reranking. The owner's text remains the best result in
this `k=1` example. The receipt captures the actual seed-plus-neighbor candidate
lists entering reranking in sync and async runs. An example-local mapping clones
those captured documents and translates graph `properties.parent_id` into
`metadata.doc_id`, preserving caller documents and graph provenance. The exact
mapped sets then pass through the same strict `AntflyDocumentStore` and media
`AntflyByteStore` resolver in sync and async paths. The receipt requires the same
three persisted parents, two original image content parts and matching media
hashes; missing/conflicting graph parent references fail before generation.
No second search is used to reconstruct the candidate set. Inspect neighbor provenance in
`metadata._antfly.properties` and traversal evidence in `metadata._antfly.graph`.
This demonstrates composition rather than an answer-quality improvement claim.
The receipt records all results, models/capabilities, dependency versions, command
and duration. Cleanup removes only the companion graph; corpus tables remain.
