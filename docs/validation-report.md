# Validation results

The integration is checked with deterministic contracts, upstream LangChain
suites, real Antfly models/storage, clean wheel installations and executable
examples. This page describes the tested scope and limits.

## Python integration

The release-candidate implementation passed **377 checks with zero skips** in
the complete live profile using published `antfly-sdk==0.2.1`. This includes
seven release-verifier checks that reject mismatched, incomplete or ambiguous
registry artifacts. The hermetic and clean-wheel profiles passed **330 checks**,
with **47 explicitly configured live cases skipped**.

On 2026-10-28, the full 377-check profile passed again against the checksum-verified
published Antfly `0.3.0-rc.5` macOS ARM64 binary, with zero skips and unchanged
source during execution. It used local `BAAI/bge-small-en-v1.5:onnx` embeddings
and `antflydb/mxbai-rerank-base-v1` reranking. The server's release commit
`ff2c66bd7ba63b3ef68d872e30d9ba31e0cf7301` contains the required graph fixes.
The [compatibility record](evaluation/server-compatibility.json) includes artifact
digests, dependency versions and the released-server example checks, without
machine-specific paths or credentials.

Hosted Python 3.11, 3.12 and 3.13 contract/wheel jobs and the lower-bound job passed:
[CI run](https://github.com/antflydb/langchain-antfly/actions/runs/40595427715).
The [build-only workflow](https://github.com/antflydb/langchain-antfly/actions/runs/40595503504)
also produced checked release-candidate distributions. Live model checks are
separate from ordinary hosted contract jobs.

The tested application behaviors include:

- CRUD, standard factories/retrievers, raw distance and relevance scores, MMR,
  external/managed embeddings and sync/async operations.
- Full-text/hybrid search, standalone/combined reranking, and graph-context expansion.
- Persistent parent documents and original media with strict reference/hash checks.
- Scoped MATCH, traversal, paths and exact aggregates, including multi-shard cases.
- Ordered memory operations, namespace isolation, timestamps, semantic indexing,
  async graph runtime and actual server-restart persistence.
- Clear handling of missing records, partial writes, unknown commit outcomes and
  cancellation after work is submitted.

Use [Testing and compatibility](validation.md) to reproduce the profiles.

## Multimodal example

The checked-in fixture has three pages and three parent objects. All twenty
evidence questions passed after fresh ingestion and after server restart.
A separate historical vision-provider run on 2026-10-26 produced answers for all
twenty questions, including the image-only marker and four unanswerable questions.
The [recorded answers](evaluation/multimodal-answers.json) preserve the actual
answers, evidence identifiers and image hashes. This is historical generation
evidence, not a claim that later validation repeated paid inference.

A [same-question retrieval comparison](evaluation/retrieval-comparison.json)
on released server RC5 used proxy-text and native multimodal embeddings. Across sixteen answerable
questions, mean required-evidence recall@1/@2 was 81.25%/96.875% for proxy retrieval
and 62.5%/100% for native retrieval. Both reached 100% at five results on just
three parents: that threshold is saturated, not a broad retrieval benchmark.
Native visual recall@1 was 0%, so a perfect aggregate recall@5 must not be
interpreted as strong visual ranking.

Four negative questions have no evidence labels and are excluded from recall.
Answer term matching is a fixture regression rubric, not a general model-quality
evaluation. The example is a declared synthetic PDF parser, not arbitrary-PDF
layout extraction or scanned-document OCR.

## Limits

Results apply to the recorded implementations, dependency resolutions and model
configurations. They do not establish every platform, workload or upstream
version combination. Windows client behavior is not part of the hosted matrix.
External adoption or independent developer satisfaction is not inferred from
test counts. Graph queries use Antfly's JSON dialect, not Cypher compatibility;
agent memory does not provide execution checkpointing or per-item TTL.
