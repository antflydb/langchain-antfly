# Persistent PDF and image RAG

This executable reference uses the architecture of the
[Machine Learning Mastery tutorial](https://machinelearningmastery.com/how-to-build-a-multi-modal-rag-pipeline-for-images-and-documents-in-python/)
with persistent Antfly stores and explicit evidence checks.

```text
text/image-summary child → AntflyDocumentStore parent → AntflyByteStore original media
```

Images are extracted from the checked-in three-page PDF with PyMuPDF. The default
summary proxies are declared fixture data, not freshly generated model evidence.
Real embedding and search calls are required for ingestion/retrieval. This example
does not promise layout-aware extraction for arbitrary PDFs or scan OCR.

## Prerequisites and preflight

From the repository root, install `uv sync --all-extras`. Start an Antfly server
with inference enabled and install a text embedding model explicitly. Set:

```bash
export ANTFLY_URL=http://localhost:8080
export ANTFLY_EMBEDDING_MODEL=your-installed-text-model
uv run python -m examples.multimodal preflight
```

Preflight reads server/model inventory and parses the fixture, but performs no
model inference or downloads. It reports media bytes and the selected model's
advertised capabilities. Credentials use `ANTFLY_TOKEN` when needed. If the server
advertises no inference models, enable inference/model discovery before continuing.
Preflight also checks existing vector-index mode, metric and advertised dimensions,
owned-record codecs, the per-image byte-store limit, and complete vision-provider
configuration. Remote vision endpoints require an explicit API key; preflight
checks its presence without authenticating or making generation calls. Download
sizes are reported as unavailable when the server does not advertise them. Large
download time, installation time, and warm query time must be measured separately.

## Ingest and retrieve

```bash
uv run python -m examples.multimodal ingest
uv run python -m examples.multimodal ask 'What color and shape is the audit marker in the shipment chart?'
uv run python -m examples.multimodal --output evidence.json verify
```

These commands use dedicated `orchard_demo_vectors`, `orchard_demo_parents`, and
`orchard_demo_media` tables. Override the shared `--prefix` before the subcommand
to isolate another run. Stable source hashes and IDs make repeated ingestion
idempotent. An unchanged complete ingestion reuses persisted data without model
calls; originals are stored before child references. Failures are surfaced rather
than hidden behind incomplete answers. Repeating a failed ingestion repairs the
same IDs. Completed captions and document embeddings are checkpointed in the parent
table before searchable publication, so repairing a later failed stage reuses
completed transforms. A pending manifest retains ownership across interrupted
source updates; replacements remove stale owned references only after publishing
and validating the new records. Retrying without `--regenerate-summaries` also
resumes an interrupted live-caption run, using its original provider configuration.

Keep model assets and task configuration fixed for a prefix. If model revision,
task instructions, dimensions or normalization change, choose a fresh prefix;
the model name alone cannot establish embedding compatibility. Regenerating
summaries reuses already completed embedding checkpoints when their input is
unchanged; it is not a model-cache invalidation command.

Existing complete ingestions from an older example can reuse captions retained in
children. If a legacy live caption is lost and no checkpoint exists, repair fails
with a request to regenerate explicitly rather than silently making a paid call.
A provider response lost before its checkpoint commits cannot establish whether
the provider completed that call; inspect the error before deciding to retry.

The answer path captures one child search result set, resolves each exact parent,
then resolves each media key and validates SHA-256 and MIME. Missing child IDs,
parents, media, or corrupt bytes prevent generation. Returned image hashes describe
the actual bytes placed into image content parts.

The healthy interoperability path constructs
`MultiVectorRetriever(vectorstore=vectors, docstore=parents, byte_store=None,
id_key="doc_id")` from `langchain_classic.retrievers.multi_vector`. The media store
is **not** passed as `byte_store`: that parameter replaces the parent docstore
with serialized byte-backed parents. Ordinary `MultiVectorRetriever` skips missing
parents, so its output alone is not the strict evidence validation used here.

## Live vision generation

Use an existing vision provider's OpenAI-compatible chat-completions endpoint:

```bash
export VISION_URL=https://your-provider.example/v1/chat/completions
export VISION_MODEL=your-vision-model
export VISION_API_KEY=your-key  # omit for an unauthenticated local endpoint
uv run python -m examples.multimodal ask 'What color and shape is the audit marker?' --live
uv run python -m examples.multimodal --output live-evidence.json verify --live
```

This uses the existing HTTP interface; it does not introduce an Antfly Chat Provider.
`--live` explicitly enables potentially paid generation. Original retrieved image
bytes and text are supplied together. Verification checks expected answer terms,
source citations, and retrieval evidence. The orange circular audit marker is
deliberately absent from textual proxies; its answer must come from the image.
The held-out detail also must remain absent when regenerating captions:

```bash
uv run python -m examples.multimodal ingest --regenerate-summaries
```

That command explicitly performs live vision caption calls; it rejects captions
that leak the held-out marker answer. Generative answer checks are a small fixture
regression suite, not a claim of broad model quality or a substitute for corpus
evaluation.

The frozen manifest contains twenty questions across text, table, visual,
cross-modal, and negative categories. Reports include per-category recall@5 and
answer accuracy. With only three parents, recall@5 includes every parent and is
an integrity smoke check, not a discriminating search benchmark. Visual-only
questions still require original image content and correct answers; no textual
proxy contains the marker's color and shape. Treat answer term matching as a
transparent fixture rubric and inspect the saved answers for contradictory claims.

For a Gemini provider, set `VISION_PROTOCOL=gemini`, `VISION_MODEL` to the chosen
model, `VISION_URL` to its full `:generateContent` REST endpoint, and
`VISION_API_KEY` through the environment. This example-local request conversion
preserves original image bytes and records provider usage. The default protocol
is `openai`. Quota, authentication, unsupported modality, and model errors stop
the run; there is no silent provider fallback.

## Restart and portability

Stop the application and restart the same server data directory. From a fresh
Python process, run `ask` and `verify` without re-ingestion. Original bytes are
resolved from Antfly, not extraction-directory paths. Keep the PDF for fixture
verification; it is not read by the query path. Persist evidence reports with the
server revision, model identity, lockfile revision, duration, and machine details
when recording a release run.

## Native image embeddings

The proxy example above proves summary-based retrieval and multimodal generation.
The separate native track embeds real image bytes, checks advertised image
capability, and verifies image-to-image and text-to-image search:

```bash
uv run python -m examples.native_multimodal --model your-installed-image-text-model ingest
# Restart the process/server before this command for persistence evidence.
uv run python -m examples.native_multimodal --model your-installed-image-text-model verify
uv run python -m examples.native_multimodal --model your-installed-image-text-model cleanup
```

The native index uses cosine distance explicitly. A text-only model fails
capability checks instead of silently embedding a filename or summary. Image
methods are explicit extensions; standard LangChain `embed_documents` remains a
text API. The native example stores media in its own byte table and verifies hashes.

## Compare native and proxy retrieval

Use the same frozen twenty questions for both retrieval modes. Set
`ANTFLY_IMAGE_MODEL` to an installed image/text embedding model, keep the proxy
ingestion above, and choose a new dedicated native table:

```bash
uv run python -m examples.evaluate_retrieval --prefix orchard_demo \
  --native-table orchard_comparison --output comparison.json ingest
uv run python -m examples.evaluate_retrieval --prefix orchard_demo \
  --native-table orchard_comparison --output comparison.json evaluate
uv run python -m examples.evaluate_retrieval --prefix orchard_demo \
  --native-table orchard_comparison --output comparison.json cleanup
```

This makes explicit embedding/search calls and no vision-generation calls. Native
text and original images share one multimodal index; proxy mode uses persisted
text/captions. Compare per-category recall@1/@2 as well as recall@5. Negative
questions have no required-evidence labels and are excluded from recall. The
three-parent fixture saturates recall@5, so it does not establish broad retrieval
quality. Answer quality and abstention remain separate live-generation checks.

## Supplementary frozen graph evaluation

The unchanged twenty questions above are supplemented by three frozen graph
labels in [graph-questions.json](fixtures/graph-questions.json): related evidence,
an exact native MATCH relationship count, and a shortest path. This evaluates the
persisted Orchard parents and proxies through the same graph construction as the
companion recipe. It makes no model or generation calls and uses a separate owned
graph table. A server supporting scoped, table-qualified MATCH aliases is required;
older servers fail closed.

```bash
uv run python -m examples.evaluate_graph run --prefix orchard_demo \
  --graph-table orchard_graph_evaluation --output graph-evaluation.json
uv run python -m examples.evaluate_graph cleanup --prefix orchard_demo \
  --graph-table orchard_graph_evaluation
```

The report records each frozen expectation and actual native evidence/count/path,
the question-manifest hash, original twenty-label hash, fixture hash, persisted
corpus hash, and ingestion-manifest hash. Failures exit nonzero. Cleanup checks an
ownership marker and preserves all input corpus tables. The opt-in live regression
uses `ANTFLY_TEST_GRAPH_URL` (and optional `ANTFLY_TEST_TOKEN`), independently of the
ordinary test server URL. These three synthetic graph cases establish operation
contracts rather than broad retrieval or generated-answer quality.

## Cleanup and fixture maintenance

```bash
uv run python -m examples.multimodal cleanup
```

If you ran the graph-context companion, first remove its owned graph using the
[companion cleanup command](../../docs/recipes.md#executable-hybrid-reranking-and-graph-context-companion).
The graph table is a separately owned recipe resource; removing the three corpus
tables alone does not remove graph copies of their evidence.

Cleanup deletes only the three dedicated tables for the chosen prefix, including
the owned vectors, parents, media, and manifest. It does not drop shared tables.
Run cleanup only for a prefix owned by this example. After cleanup, queries must
fail on the missing table rather than report an empty successful answer.

Regenerate the fixture with `uv run python -m examples.multimodal.generate_fixture`,
render all pages, inspect them, and rerun `tests/test_examples.py`. The answer
manifest and visual-only omission are deliberate contracts. Fixtures are synthetic
and redistributable under this repository's license.
