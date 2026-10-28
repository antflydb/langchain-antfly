import pytest
from langchain_antfly.embeddings import AntflyEmbeddings


class Client:
    def __init__(self):
        self.calls = []

    def embed(self, body):
        self.calls.append(body)
        return {"data": [{"index": i, "embedding": [1.0, float(i)]} for i in reversed(range(len(body["input"])))]}

    def models(self):
        return {"embedders": {"vision": {"inputs": ["text", "image"]}, "text": {"inputs": ["text"]}}}


def test_batches_task_types_order_and_no_hidden_truncation():
    client = Client()
    embedding = AntflyEmbeddings(client=client, model="text", batch_size=2)
    assert embedding.embed_documents(["a", "b", "c"]) == [[1, 0], [1, 1], [1, 0]]
    embedding.embed_query("question")
    assert [call["task_type"] for call in client.calls] == [
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
    ]
    assert all("dimensions" not in call for call in client.calls)
    assert embedding.embed_documents([]) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [{"index": 0, "embedding": [1, 2]}, {"index": 0, "embedding": [1, 2]}]},
        {"data": [{"index": 0, "embedding": [float("nan"), 2]}, {"index": 1, "embedding": [1, 2]}]},
        {"data": [{"index": 0, "embedding": {"indices": [1], "values": [1]}}, {"index": 1, "embedding": [1, 2]}]},
        {"data": [], "errors": [{"index": 0}]},
    ],
)
def test_malformed_or_partial_responses_fail(payload):
    client = Client()
    client.embed = lambda body: payload
    with pytest.raises(ValueError):
        AntflyEmbeddings(client=client, model="text").embed_documents(["a", "b"])


def test_explicit_images_check_capability_and_preserve_bytes():
    client = Client()
    with pytest.raises(ValueError, match="image"):
        AntflyEmbeddings(client=client, model="text").embed_images([b"original"])
    embedding = AntflyEmbeddings(client=client, model="vision")
    embedding.embed_images([b"original"])
    assert client.calls[-1]["input"][0]["image_url"]["url"] == "data:image/png;base64,b3JpZ2luYWw="


@pytest.mark.asyncio
async def test_async_embeddings():
    assert await AntflyEmbeddings(client=Client(), model="text").aembed_query("q") == [1, 0]


@pytest.mark.parametrize(
    "info,dimensions",
    [
        ({"inputs": ["image"]}, None),
        ({"inputs": ["text"], "inference_capabilities": {"task": "generate"}}, None),
        ({"dense_output": {"minimum_dimensions": 2, "native_dimensions": 4}}, 5),
    ],
)
def test_advertised_embedding_capabilities_fail_before_inference(info, dimensions):
    client = Client()
    client.models = lambda: {"embedders": {"text": info}}
    with pytest.raises(ValueError):
        AntflyEmbeddings(client=client, model="text", dimensions=dimensions).embed_documents(["secret"])
    assert client.calls == []
