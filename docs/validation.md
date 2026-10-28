# Testing and compatibility

## Supported versions

| Component | Requirement |
| --- | --- |
| Python | 3.11 or newer; hosted tests cover 3.11, 3.12 and 3.13 |
| Antfly Python client | `antfly-sdk>=0.2.1,<0.3` |
| LangChain core, retrieval | `langchain-core>=1.0,<2` |
| LangChain core, memory | `langchain-core>=1.4.4,<2` |
| LangGraph checkpoint library | `langgraph-checkpoint>=4.3,<5`; used for the public store interface |
| Antfly server | `0.3.0-rc.5` is the released compatibility baseline |

The lockfile records the reproducible development resolution. Package ranges
are supported by the tested resolutions and focused minimum-version checks,
not an exhaustive combination of every upstream release. Optional graph
integration uses `langchain-community`; examples using an agent or chat model
require the relevant LangChain/provider packages separately.

## Hermetic contracts

```bash
uv sync --frozen --all-extras
uv run ruff check .
uv run pytest -q
```

These tests require no server, model download or cloud credentials. Live cases
are skipped unless their prerequisites are explicitly configured.

The suite includes LangChain's upstream vector/embedding/store contracts,
reliability and cancellation regressions, graph scope tests, memory namespace
and concurrency checks, and the example evidence boundary.

## Live server profile

Use a dedicated Antfly instance with installed text-embedding and reranking
models. Test tables use unique names and are cleaned up by the tests.

```bash
uv sync --frozen --all-extras --group interoperability
export ANTFLY_TEST_URL=http://localhost:8080
export ANTFLY_TEST_GRAPH_URL=http://localhost:8080
export ANTFLY_TEST_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5:onnx
export ANTFLY_TEST_RERANK_MODEL=antflydb/mxbai-rerank-base-v1
export ANTFLY_TEST_EXTERNAL_PROVIDER=openai
# export ANTFLY_TEST_TOKEN=...  # if required by the server
uv run --group interoperability python scripts/validate.py \
  --output .state/live -- uv run --group interoperability pytest -q
```

The optional OpenAI integration here exercises an unchanged `OpenAIEmbeddings`
client against Antfly's local OpenAI-compatible embedding endpoint, not the
OpenAI cloud. The example configures a dummy local token and explicit transports.

The live profile checks immediate write/search visibility, managed and external
embeddings, third-party provider compatibility, multi-shard graph operations,
transaction outcomes, namespaced memory and persistent multimodal evidence.
Native image-model and live-generation recipes are separate example runs with
their own explicit prerequisites.

## Example checks

Follow the [multimodal walkthrough](../examples/multimodal/README.md) for
`preflight → ingest → ask → verify → cleanup`. Repeat queries from a fresh process
after restarting the server with the same data directory. Missing or corrupt
parents/media must prevent generation.

The [backend migration exercise](backend-swap.md) checks persistent attach
without repeating inference. The [agent example](../examples/agent_memory.py)
uses `create_agent(store=...)` and `ToolRuntime.store` with trusted user context.

## Distribution checks

```bash
uv build --no-sources --package langchain-antfly
uv build --no-sources --package langgraph-store-antfly
uvx 'twine>=7,<8' check --strict dist/*.whl dist/*.tar.gz
```

Install wheels in a clean environment, resolve the SDK from PyPI, run
`uv pip check` and repeat the contracts. The CI matrix exercises source and
wheel installations on Python 3.11–3.13 and a lower-bound resolution. A weekly
canary checks newer compatible upstream versions. Dedicated model jobs require
configured server/model variables. Scheduled live jobs run only when a dedicated
test-server URL is configured; manual live validation still requires all model
prerequisites. The hermetic weekly canary runs independently. Paid generation is
manual opt-in only.

The validation script records source hashes before/after execution, dependency
versions, raw output and JUnit counts. A failed or skipped live profile must not
be reported as a passing real-server run. See [recorded results](validation-report.md).
