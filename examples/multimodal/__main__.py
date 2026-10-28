"""Run with ``python -m examples.multimodal --help`` from the repository root."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import httpx
from antfly import AntflyClient
from langchain_core.documents import Document

from .._client import client_from_env
from .corpus import FIXTURES, PDF, extract
from .evidence import aresolve, resolve, sha256

VISION_CALLS: list[dict[str, Any]] = []


def _vision(parts: list[dict[str, Any]], question: str) -> str:
    url = os.getenv("VISION_URL")
    model = os.getenv("VISION_MODEL")
    if not url or not model:
        raise ValueError("Live generation requires explicit VISION_URL (chat completions endpoint) and VISION_MODEL")
    headers = {}
    protocol = os.getenv("VISION_PROTOCOL", "openai")
    key = os.getenv("VISION_API_KEY")
    system = "Answer only from supplied evidence. Cite [source_id]. If unsupported, say insufficient evidence."
    body = {
        "model": model,
        "max_tokens": 600,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [*parts, {"type": "text", "text": question}]},
        ],
    }
    if protocol == "openai":
        if key:
            headers["Authorization"] = f"Bearer {key}"
    elif protocol == "gemini":
        if key:
            headers["x-goog-api-key"] = key
        content = []
        for part in parts:
            if part["type"] == "text":
                content.append({"text": part["text"]})
            else:
                data_url = part["image_url"]["url"]
                header, data = data_url.split(",", 1)
                content.append(
                    {"inlineData": {"mimeType": header.removeprefix("data:").removesuffix(";base64"), "data": data}}
                )
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [*content, {"text": question}]}],
            "generationConfig": {"maxOutputTokens": 900, "temperature": 0},
        }
    else:
        raise ValueError("VISION_PROTOCOL must be openai or gemini")
    response = httpx.post(url, headers=headers, timeout=120, json=body)
    if response.is_error:
        raise RuntimeError(f"Vision provider returned HTTP {response.status_code}: {response.text[:1000]}")
    payload = response.json()
    VISION_CALLS.append(
        {
            "model": payload.get("modelVersion", payload.get("model", model)),
            "usage": payload.get("usageMetadata", payload.get("usage", {})),
            "image_parts": sum(part["type"] == "image_url" for part in parts),
        }
    )
    if protocol == "gemini":
        answer = "".join(
            part.get("text", "") for part in payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        )
    else:
        answer = payload["choices"][0]["message"]["content"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Vision provider returned no usable text; check safety/finish reason and output token budget")
    return answer


def _classes() -> tuple[Any, Any, Any, Any]:
    from langchain_antfly import AntflyByteStore, AntflyDocumentStore, AntflyEmbeddings, AntflyVectorStore

    return AntflyByteStore, AntflyDocumentStore, AntflyEmbeddings, AntflyVectorStore


def attach(client: AntflyClient, prefix: str, model: str) -> tuple[Any, Any, Any]:
    byte_store, document_store, embeddings, vectors = _classes()
    return (
        vectors(client=client, table=f"{prefix}_vectors", embedding=embeddings(client=client, model=model)),
        document_store(client=client, table=f"{prefix}_parents"),
        byte_store(client=client, table=f"{prefix}_media"),
    )


def ingest(client: AntflyClient, prefix: str, model: str, *, regenerate: bool = False) -> dict[str, Any]:
    byte_store, document_store, embeddings, vectors = _classes()
    tables = {item.get("name", item.get("table_name")) for item in client.list_tables()}
    parents_store = (
        document_store(client=client, table=f"{prefix}_parents")
        if f"{prefix}_parents" in tables
        else document_store.create(client=client, table=f"{prefix}_parents")
    )
    media_store = (
        byte_store(client=client, table=f"{prefix}_media")
        if f"{prefix}_media" in tables
        else byte_store.create(client=client, table=f"{prefix}_media")
    )
    parents, children, media = extract()
    fingerprint = sha256(PDF.read_bytes() + model.encode() + (FIXTURES / "answers.json").read_bytes())
    previous = parents_store.mget(["ingestion-manifest"])[0]
    from langchain_core.embeddings import Embeddings

    # Each completed transform is persisted before publishing references. Retrying
    # a later failed stage reuses captions/vectors rather than calling providers.
    checkpoint_prefix = "transform-v1:"
    captions_live = regenerate or bool(
        previous and previous.metadata.get("fingerprint") == fingerprint and previous.metadata.get("captions_live")
    )
    caption_config = {
        "revision": "caption-v1-no-marker",
        "protocol": os.getenv("VISION_PROTOCOL", "openai"),
        "url": os.getenv("VISION_URL", ""),
        "model": os.getenv("VISION_MODEL", ""),
    }
    requested_caption_config = dict(caption_config)
    if not regenerate and previous and previous.metadata.get("fingerprint") == fingerprint:
        caption_config = previous.metadata.get("caption_config", caption_config)
    pending = parents_store.mget(["ingestion-pending"])[0]
    if not regenerate and pending and pending.metadata.get("fingerprint") == fingerprint:
        captions_live = pending.metadata["captions_live"]
        caption_config = pending.metadata["caption_config"]
    owned = {
        "children": {c.id for c in children},
        "parents": {p.id for p in parents},
        "media": set(media),
    }
    for state in (previous, pending):
        if state:
            for kind in owned:
                owned[kind].update(state.metadata.get(kind, []))
    parents_store.mset(
        [
            (
                "ingestion-pending",
                Document(
                    page_content="Resumable ingestion",
                    metadata={
                        "fingerprint": fingerprint,
                        "captions_live": captions_live,
                        "caption_config": caption_config,
                        **{kind: sorted(keys) for kind, keys in owned.items()},
                    },
                ),
            )
        ]
    )
    caption_calls = 0
    if captions_live:
        for parent, child in zip(parents, children, strict=True):
            if not parent.metadata.get("media_key"):
                continue
            key = checkpoint_prefix + sha256(
                json.dumps(caption_config, sort_keys=True).encode() + media[parent.metadata["media_key"]]
            )
            cached = parents_store.mget([key])[0]
            if (
                cached is None
                and not regenerate
                and previous
                and previous.metadata.get("captions_live")
                and previous.metadata.get("fingerprint") == fingerprint
                and not (pending and pending.metadata.get("fingerprint") == fingerprint)
            ):
                # Upgrade a legacy completed ingestion without repeating a paid caption.
                legacy = vectors(
                    client=client, table=f"{prefix}_vectors", embedding=embeddings(client=client, model=model)
                )
                saved = legacy.get_by_ids([child.id])
                if saved:
                    cached = Document(page_content=saved[0].page_content)
                    parents_store.mset([(key, cached)])
                else:
                    raise ValueError(
                        "Legacy live caption is missing and has no checkpoint; explicitly regenerate summaries to repair"
                    )
            if cached is None:
                if caption_config != requested_caption_config:
                    raise ValueError("Resume requires the original VISION_URL, VISION_MODEL and VISION_PROTOCOL")
                from .evidence import Evidence

                parts, _ = Evidence([parent], media, []).content_parts()
                caption = _vision(
                    parts,
                    "Write a searchable summary of this figure. Omit audit-marker color and shape, and exact numeric values.",
                )
                caption_calls += 1
                if any(term in caption.lower() for term in ("orange", "circle")):
                    raise ValueError(
                        "Caption leaked the held-out visual answer; regenerate with a suitable model/prompt"
                    )
                cached = Document(page_content=caption)
                parents_store.mset([(key, cached)])
            child.page_content = cached.page_content

    provider = embeddings(client=client, model=model)
    embedding_calls = 0

    class CheckpointEmbeddings(Embeddings):
        def embed_documents(self, texts):
            nonlocal embedding_calls
            result = []
            for text in texts:
                key = checkpoint_prefix + sha256(json.dumps(["embedding-v1", model, text]).encode())
                cached = parents_store.mget([key])[0]
                if cached is None:
                    vector = provider.embed_documents([text])[0]
                    embedding_calls += 1
                    cached = Document(page_content="Dense document embedding", metadata={"vector": vector})
                    parents_store.mset([(key, cached)])
                result.append(cached.metadata["vector"])
            return result

        def embed_query(self, text):
            return provider.embed_query(text)

    complete = (
        previous
        and previous.metadata.get("fingerprint") == fingerprint
        and not regenerate
        and owned["children"] == {c.id for c in children}
        and owned["parents"] == {p.id for p in parents}
        and owned["media"] == set(media)
    )
    if complete:
        from .evidence import IncompleteEvidenceError

        try:
            resolve(children, parents_store, media_store)
            existing = vectors(client=client, table=f"{prefix}_vectors", embedding=provider)
            actual_parents = parents_store.mget([p.id for p in parents])
            parents_match = all(
                actual is not None
                and actual.page_content == expected.page_content
                and actual.metadata == expected.metadata
                for actual, expected in zip(actual_parents, parents, strict=True)
            )
            actual = {d.id: d for d in existing.get_by_ids([c.id for c in children])}
            complete = (
                parents_match
                and len(actual) == len(children)
                and all(
                    c.id in actual
                    and actual[c.id].page_content == c.page_content
                    and {k: value for k, value in actual[c.id].metadata.items() if k != "_antfly"} == c.metadata
                    for c in children
                )
            )
            if complete:
                for child in children:
                    key = checkpoint_prefix + sha256(json.dumps(["embedding-v1", model, child.page_content]).encode())
                    checkpoint = parents_store.mget([key])[0]
                    if checkpoint is None:
                        continue  # Legacy complete ingestions did not retain transform checkpoints.
                    row = client.get(f"{prefix}_vectors", child.id, fields=["_embeddings"])
                    actual_vector = row.get("_embeddings", {}).get("vectors", [])
                    expected_vector = checkpoint.metadata["vector"]
                    if (
                        not isinstance(actual_vector, list)
                        or len(actual_vector) != len(expected_vector)
                        or not all(
                            isinstance(a, (int, float))
                            and not isinstance(a, bool)
                            and math.isfinite(a)
                            and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
                            for a, b in zip(actual_vector, expected_vector, strict=True)
                        )
                    ):
                        complete = False
                        break
        except Exception as error:
            from antfly import AntflyNotFoundError

            if not isinstance(error, (IncompleteEvidenceError, AntflyNotFoundError)):
                raise
            complete = False
        if complete:
            parents_store.mdelete(["ingestion-pending"])
            return {"status": "unchanged", "fingerprint": fingerprint, "model_calls": 0}
    # Originals precede searchable references; repeating failed stages is safe with stable IDs.
    media_store.mset(list(media.items()))
    parents_store.mset([(p.id, p) for p in parents])
    vector_store = vectors.from_documents(
        children,
        embedding=CheckpointEmbeddings(),
        client=client,
        table=f"{prefix}_vectors",
        distance_metric="cosine",
    )
    # Healthy standard interoperability; the answer path below additionally enforces strict references.
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

    MultiVectorRetriever(vectorstore=vector_store, docstore=parents_store, byte_store=None, id_key="doc_id")
    resolve(children, parents_store, media_store)
    stale = owned["children"] - {c.id for c in children}
    if stale:
        vector_store.delete(ids=sorted(stale))
    parents_store.mdelete(sorted(owned["parents"] - {p.id for p in parents}))
    media_store.mdelete(sorted(owned["media"] - set(media)))
    parents_store.mset(
        [
            (
                "ingestion-manifest",
                Document(
                    page_content="Complete ingestion",
                    metadata={
                        "fingerprint": fingerprint,
                        "model": model,
                        "children": [c.id for c in children],
                        "parents": [p.id for p in parents],
                        "media": list(media),
                        "captions_live": captions_live,
                        "caption_config": caption_config,
                    },
                ),
            )
        ]
    )
    parents_store.mdelete(["ingestion-pending"])
    return {
        "status": "ingested",
        "parents": len(parents),
        "children": len(children),
        "media": len(media),
        "fingerprint": fingerprint,
        "captions": "live" if captions_live else "checked-in fixture proxies",
        "model_calls": caption_calls + embedding_calls,
    }


def ask(
    client: AntflyClient, prefix: str, model: str, question: str, *, live: bool = False, k: int = 5
) -> dict[str, Any]:
    started = time.monotonic()
    vectors, parents, media = attach(client, prefix, model)
    children = vectors.similarity_search(question, k=k)
    evidence = resolve(children, parents, media)
    parts, trace = evidence.content_parts()
    result = {
        "question": question,
        "evidence_ids": [p.metadata["source_id"] for p in evidence.parents],
        "child_ids": [c.id for c in children],
        "images": trace,
        "generation": "not requested",
    }
    if live:
        result["answer"] = _vision(parts, question)
        result["generation"] = "live"
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    return result


async def aask(
    client: AntflyClient, prefix: str, model: str, question: str, *, live: bool = False, k: int = 5
) -> dict[str, Any]:
    started = time.monotonic()
    vectors, parents, media = attach(client, prefix, model)
    children = await vectors.asimilarity_search(question, k=k)
    evidence = await aresolve(children, parents, media)
    parts, trace = evidence.content_parts()
    result = {
        "question": question,
        "evidence_ids": [p.metadata["source_id"] for p in evidence.parents],
        "child_ids": [c.id for c in children],
        "images": trace,
        "generation": "not requested",
    }
    if live:
        result["answer"] = await asyncio.to_thread(_vision, parts, question)
        result["generation"] = "live"
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    return result


def preflight(client: AntflyClient, prefix: str, model: str) -> dict[str, Any]:
    """Inspect advertised capabilities/resources without inference or mutation."""
    from urllib.parse import urlparse

    parents, children, media = extract()
    available = client.models()
    info = available.get("embedders", {}).get(model)
    if not isinstance(info, dict):
        raise ValueError(f"Embedding model {model!r} is not installed; available: {list(available.get('embedders', {}))}")
    if "text" not in info.get("inputs", []):
        raise ValueError(f"Model {model!r} must advertise text embedding support")
    task = info.get("inference_capabilities", {}).get("task")
    if task is not None and task not in {"embed", "embedding"}:
        raise ValueError(f"Model {model!r} advertises incompatible inference task {task!r}")
    tables = {item.get("name", item.get("table_name")) for item in client.list_tables()}
    _, parent_store, media_store = attach(client, prefix, model)
    if f"{prefix}_vectors" in tables:
        _, _, embeddings, vector_class = _classes()
        existing = vector_class(
            client=client,
            table=f"{prefix}_vectors",
            embedding=embeddings(client=client, model=model),
            distance_metric="cosine",
        )
        existing.validate()
        width = info.get("dense_output")
        if existing.dimension is not None and isinstance(width, dict):
            if not width["minimum_dimensions"] <= existing.dimension <= width["native_dimensions"]:
                raise ValueError("Existing index dimension is outside the model's advertised dense range")
    for suffix, store in (("parents", parent_store), ("media", media_store)):
        if f"{prefix}_{suffix}" in tables:
            client.get_table(f"{prefix}_{suffix}")
            # Decode representative owned records so incompatible codecs fail early.
            keys = ["ingestion-manifest", *[p.id for p in parents]] if suffix == "parents" else list(media)
            store.mget(keys)
    for key, value in media.items():
        if len(value) > media_store.max_value_bytes:
            raise ValueError(f"Media {key!r} exceeds byte-store limit {media_store.max_value_bytes}")
    vision_url = os.getenv("VISION_URL", "")
    vision_model = os.getenv("VISION_MODEL", "")
    protocol = os.getenv("VISION_PROTOCOL", "openai")
    if bool(vision_url) != bool(vision_model):
        raise ValueError("Configure VISION_URL and VISION_MODEL together")
    if protocol not in {"openai", "gemini"}:
        raise ValueError("VISION_PROTOCOL must be openai or gemini")
    if vision_url:
        parsed = urlparse(vision_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("VISION_URL must be an HTTP(S) generation endpoint")
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if not local and not os.getenv("VISION_API_KEY"):
            raise ValueError(
                "Remote vision configuration requires VISION_API_KEY; credentials are not verified by preflight"
            )
        if vision_url.startswith(client.base_url.rstrip("/")):
            generator = available.get("generators", {}).get(vision_model, {})
            inputs = generator.get("inputs", generator.get("inference_capabilities", {}).get("input_modalities", []))
            if "image" not in inputs:
                raise ValueError(f"Local vision model {vision_model!r} does not advertise image support")
    return {
        "server_tables": len(tables),
        "fixture_sha256": sha256(PDF.read_bytes()),
        "parents": len(parents),
        "children": len(children),
        "media_bytes": sum(map(len, media.values())),
        "media_value_limit_bytes": media_store.max_value_bytes,
        "embedding_model": model,
        "downloads": "none; install missing models explicitly (server does not advertise download sizes)",
        "model_calls": 0,
        "embedding_capabilities": available["embedders"][model],
        "vision_configured": bool(vision_url and vision_model),
        "remote_credentials": "configured, not authenticated" if vision_url else "not configured",
        "schema": "existing owned resources checked; absent resources will be created during ingestion",
    }


def main() -> None:
    VISION_CALLS.clear()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix", default="orchard_demo", help="Dedicated table prefix; cleanup deletes only these tables"
    )
    parser.add_argument("--model", default=os.getenv("ANTFLY_EMBEDDING_MODEL"))
    parser.add_argument("--output", type=Path, help="Write the evidence/verification report as JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    ingestion = commands.add_parser("ingest")
    ingestion.add_argument("--regenerate-summaries", action="store_true", help="Explicitly make live vision calls")
    question = commands.add_parser("ask")
    question.add_argument("question")
    question.add_argument("--live", action="store_true", help="Explicitly call the configured vision provider")
    question.add_argument("--async", dest="asynchronous", action="store_true")
    verification = commands.add_parser("verify")
    verification.add_argument("--live", action="store_true")
    commands.add_parser("cleanup")
    args = parser.parse_args()
    if not args.prefix.replace("_", "").isalnum():
        parser.error("--prefix must contain only letters, digits and underscores")
    if args.command != "cleanup" and not args.model:
        parser.error("Set ANTFLY_EMBEDDING_MODEL or --model explicitly; no model is selected or downloaded implicitly")
    client = client_from_env()
    if args.command == "preflight":
        result = preflight(client, args.prefix, args.model)
    elif args.command == "ingest":
        result = ingest(client, args.prefix, args.model, regenerate=args.regenerate_summaries)
    elif args.command == "ask":
        result = (
            asyncio.run(aask(client, args.prefix, args.model, args.question, live=args.live))
            if args.asynchronous
            else ask(client, args.prefix, args.model, args.question, live=args.live)
        )
    elif args.command == "verify":
        manifest = json.loads((FIXTURES / "answers.json").read_text())
        results = [
            ask(client, args.prefix, args.model, item["question"], live=args.live) for item in manifest["questions"]
        ]
        failures = []
        category_counts = {}
        for expected, actual in zip(manifest["questions"], results, strict=True):
            required = expected["parents"]
            recalled = set(required).issubset(actual["evidence_ids"])
            if not recalled:
                failures.append(f"{expected['id']}: required evidence missing")
            if expected.get("requires_image") and not set(required).intersection(
                i["source_id"] for i in actual["images"]
            ):
                failures.append(f"{expected['id']}: raw image evidence absent")
            answer_passed = None
            if args.live:
                terms = expected["answer_terms"]
                answer_passed = all(term.lower() in actual["answer"].lower() for term in terms)
                if expected.get("answer_any"):
                    answer_passed &= any(term.lower() in actual["answer"].lower() for term in expected["answer_any"])
                if not answer_passed:
                    failures.append(f"{expected['id']}: expected answer terms absent")
                if any(f"[{parent}]" not in actual["answer"] for parent in required):
                    failures.append(f"{expected['id']}: source citation absent")
            counts = category_counts.setdefault(
                expected["category"], {"questions": 0, "retrieval_questions": 0, "recall_passed": 0, "answer_passed": 0}
            )
            counts["questions"] += 1
            counts["retrieval_questions"] += int(bool(required))
            counts["recall_passed"] += int(recalled and bool(required))
            counts["answer_passed"] += int(answer_passed or False)
            actual["evaluation_id"] = expected["id"]
        for counts in category_counts.values():
            counts["recall_at_5"] = (
                counts["recall_passed"] / counts["retrieval_questions"] if counts["retrieval_questions"] else None
            )
            counts["answer_accuracy"] = counts["answer_passed"] / counts["questions"] if args.live else None
        result = {
            "passed": not failures,
            "failures": failures,
            "live_generation": args.live,
            "results": results,
            "evaluation_revision": manifest["evaluation_revision"],
            "categories": category_counts,
            "fixture_sha256": sha256(PDF.read_bytes()),
            "embedding_model": args.model,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("langchain-core", "langchain-classic", "antfly-sdk", "pymupdf")
            },
            "retrieval_caveat": "Three-parent fixture: recall@5 covers all parents; not a retrieval benchmark",
        }
    else:
        tables = {item.get("name", item.get("table_name")) for item in client.list_tables()}
        removed = []
        for suffix in ("vectors", "parents", "media"):
            table = f"{args.prefix}_{suffix}"
            if table in tables:
                client.drop_table(table)
                removed.append(table)
        result = {"removed_tables": removed}
    if VISION_CALLS:
        result["vision_calls"] = VISION_CALLS
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    if args.command == "verify" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
