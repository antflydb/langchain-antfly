# Antfly for LangChain and LangGraph

Persistent retrieval and agent memory backed by [Antfly](https://antfly.io).
Use familiar LangChain interfaces for vector and hybrid search, text/image
embeddings, reranking, documents, media, graph context, and LangGraph memory.

| Package | Import | Purpose |
| --- | --- | --- |
| `langchain-antfly` | `langchain_antfly` | Retrieval, embeddings, reranking, document/media stores and graph tools |
| `langgraph-store-antfly` | `langgraph_store_antfly` | Persistent, namespaced long-term agent memory |

**Release candidate:** `0.1.0rc1`, available on PyPI for
[LangChain](https://pypi.org/project/langchain-antfly/0.1.0rc1/) and
[LangGraph memory](https://pypi.org/project/langgraph-store-antfly/0.1.0rc1/).

## Requirements

- Python 3.11 or newer.
- An Antfly server. Antfly `0.3.0-rc.5` is the tested server baseline, including
  table-scoped graph queries. Download a binary for your platform from the
  [official release](https://github.com/antflydb/antfly/releases/tag/v0.3.0-rc.5).
- An explicit embedding model: either installed in Antfly, or supplied through a
  standard LangChain `Embeddings` provider.

The Python packages connect to a server; they do not install or start one.
The dependency on `antfly-sdk>=0.2.1` is resolved from PyPI.

## First result

### Start a local server

Set `ANTFLY_BIN` to the downloaded Antfly executable. If you need a local text
embedding model, pull it explicitly before starting the server:

```bash
export ANTFLY_BIN=/absolute/path/to/antfly
"$ANTFLY_BIN" inference pull BAAI/bge-small-en-v1.5:onnx --tasks embed

mkdir -p .state
cat > .state/antfly.json <<'JSON'
{"public_inference":{"enabled":true}}
JSON
"$ANTFLY_BIN" standalone --config .state/antfly.json \
  --port 8080 --health-port 4200 --data-dir "$PWD/.state/server" \
  --process-memory-budget-mb 4096
```

In another terminal, confirm readiness and the exact served model name:

```bash
curl -fsS http://127.0.0.1:4200/readyz
curl -fsS http://127.0.0.1:8080/ai/v1/models
```

Model downloads and cold startup are separate from query time. The examples
never choose a paid provider or download models implicitly.

### Run the three-document example

```bash
git clone https://github.com/antflydb/langchain-antfly.git
cd langchain-antfly
uv sync --frozen --all-extras

export ANTFLY_URL=http://127.0.0.1:8080
export ANTFLY_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5:onnx
# export ANTFLY_TOKEN=...  # if your server requires authentication

uv run python -m examples.quickstart
```

Expected output:

```text
[owner] Mira Chen owns operations and handles shipment escalation.
```

Add `--async` for the async factory/retriever path. Stable IDs update the same
documents when rerun; `--cleanup` removes only the example's dedicated table.

### Install the distributions

Install the release candidates from PyPI:

```bash
pip install langchain-antfly==0.1.0rc1
pip install langgraph-store-antfly==0.1.0rc1
# Optional GraphDocument ingestion and graph tools:
pip install 'langchain-antfly[graph]==0.1.0rc1'
```

To build and install the candidate wheels from a checkout:

```bash
uv build --no-sources --package langchain-antfly
uv build --no-sources --package langgraph-store-antfly
uv venv .state/wheel-venv
uv pip install --python .state/wheel-venv/bin/python \
  dist/langchain_antfly-0.1.0rc1-py3-none-any.whl \
  dist/langgraph_store_antfly-0.1.0rc1-py3-none-any.whl
```

## Ordinary LangChain APIs

```python
from antfly import AntflyClient
from langchain_antfly import AntflyEmbeddings, AntflyVectorStore
from langchain_core.documents import Document

with AntflyClient(base_url="http://localhost:8080") as client:
    embedding = AntflyEmbeddings(
        client=client, model="BAAI/bge-small-en-v1.5:onnx"
    )
    store = AntflyVectorStore.from_documents(
        [Document(id="owner", page_content="Mira Chen owns operations.")],
        embedding,
        client=client,
        table="reports",
        distance_metric="cosine",
    )
    documents = store.as_retriever(search_kwargs={"k": 1}).invoke(
        "Who owns operations?"
    )
    print(documents[0].page_content)
```

Factories create missing tables or reuse compatible ones. Constructors attach
without creating resources. Existing tables are never silently dropped or
recreated, and supplied clients remain caller-owned.

External providers implement LangChain's standard `Embeddings` interface and need
no Antfly-specific model attributes. Returned dimensions and finite values are
validated. See the [API reference](docs/api-reference.md) for managed indexes,
scores, filters, write outcomes, and sync/async contracts.

## PDF and image RAG

The [multimodal walkthrough](examples/multimodal/README.md) extracts text and
images from a checked-in [synthetic PDF](examples/multimodal/fixtures/orchard-report.pdf),
stores searchable children alongside persistent parents and original image
bytes, and assembles grounded prompts for an existing vision provider.

```bash
uv run python -m examples.multimodal preflight
uv run python -m examples.multimodal ingest
uv run python -m examples.multimodal ask 'Who owns operations?'
uv run python -m examples.multimodal --output evidence.json verify
```

The fixture includes prose, a table, a chart and a system diagram. Its twenty
questions cover text, tables, visual details, combined evidence and abstention.
Live captions/answers require explicit provider configuration and opt-in.
The walkthrough also covers restart, repair, cleanup and native image embeddings.

## Documentation

- [Recipes](docs/recipes.md): metadata filters, hybrid retrieval, reranking, caching, graphs and memory.
- [Graph input migration](docs/recipes.md#migrate-from-community-graph-inputs):
  source-checkout APIs replacing Community in the next release.
- [API reference](docs/api-reference.md): interfaces and reliability contracts.
- [Backend migration](docs/backend-swap.md): replace in-memory stores with persistent Antfly stores.
- [Testing and compatibility](docs/validation.md) and [validation results](docs/validation-report.md).
- [Contributing](CONTRIBUTING.md), [release guide](docs/publishing.md), [security policy](SECURITY.md).
- [Changelog](CHANGELOG.md).

Graph operations use Antfly's JSON graph dialect. Agent memory is long-term
storage, not a LangGraph execution checkpointer. Use existing LangChain chat
providers for generation; no Antfly chat-model adapter is included.

Licensed under [Apache-2.0](LICENSE).
