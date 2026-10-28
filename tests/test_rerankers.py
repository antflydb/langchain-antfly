import pytest
from langchain_antfly.rerankers import AntflyReranker
from langchain_core.documents import Document


class Client:
    def rerank(self, body):
        return {
            "data": [{"index": i, "score": score} for i, score in enumerate([0.1, 0.8, 0.8])][: len(body["documents"])]
        }


def test_rerank_stable_ties_identity_and_runnable():
    documents = [Document(id=str(i), page_content=str(i), metadata={"source": i}) for i in range(3)]
    reranker = AntflyReranker(client=Client(), model="rank", top_n=2)
    result = reranker.as_runnable().invoke({"query": "q", "documents": documents})
    assert [doc.id for doc in result] == ["1", "2"]
    assert documents[1].metadata == {"source": 1}
    assert result[0].metadata["_antfly"]["rerank_score"] == 0.8
    assert reranker.rerank("q", documents, top_n=0) == []


@pytest.mark.asyncio
async def test_async_compression():
    result = await AntflyReranker(client=Client(), model="rank").acompress_documents([Document(page_content="a")], "q")
    assert result[0].page_content == "a"


def test_invalid_count_fails():
    client = Client()
    client.rerank = lambda body: {"data": []}
    with pytest.raises(ValueError, match="each document"):
        AntflyReranker(client=client, model="rank").rerank("q", [Document(page_content="a")])


def test_live_reranking():
    import os

    from antfly import AntflyClient

    url, model = os.environ.get("ANTFLY_TEST_URL"), os.environ.get("ANTFLY_TEST_RERANK_MODEL")
    if not url or not model:
        pytest.skip("Set ANTFLY_TEST_URL and ANTFLY_TEST_RERANK_MODEL for installed-model validation")
    documents = [
        Document(id="database", page_content="Antfly is a database."),
        Document(id="fruit", page_content="Bananas are yellow fruit."),
    ]
    ranked = AntflyReranker(
        client=AntflyClient(url, token=os.getenv("ANTFLY_TEST_TOKEN"), timeout=120), model=model
    ).rerank("database", documents)
    assert [doc.id for doc in ranked] == ["database", "fruit"]
