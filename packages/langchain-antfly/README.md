# Antfly for LangChain

Persistent vector and hybrid retrieval, text/image embeddings, reranking,
document and media storage, and graph context backed by an Antfly server.
Python 3.11 or newer. Apache-2.0 licensed.

**Release candidate:** `0.1.0rc1`. Requires `antfly-sdk>=0.2.1` and a compatible
Antfly server; `0.3.0-rc.5` is the tested baseline. See the
[server setup](https://github.com/antflydb/langchain-antfly#first-result).

## Install

```bash
pip install langchain-antfly==0.1.0rc1
# Optional GraphDocument ingestion and graph tools:
pip install 'langchain-antfly[graph]==0.1.0rc1'
```

The package connects to an existing Antfly server; it does not install or start
one. Configure a compatible server with inference enabled and choose an installed
embedding model using the [setup guide](https://github.com/antflydb/langchain-antfly#first-result).
Models are not downloaded automatically.

## Retrieve three documents

```python
import os

from antfly import AntflyClient
from langchain_antfly import AntflyEmbeddings, AntflyVectorStore
from langchain_core.documents import Document

with AntflyClient(
    base_url=os.environ["ANTFLY_URL"], token=os.getenv("ANTFLY_TOKEN")
) as client:
    embeddings = AntflyEmbeddings(
        client=client, model=os.environ["ANTFLY_EMBEDDING_MODEL"]
    )
    store = AntflyVectorStore.from_documents(
        [
            Document(id="owner", page_content="Mira Chen owns operations."),
            Document(id="target", page_content="The shipment target is 120 crates."),
            Document(id="storage", page_content="Apples are stored in the cool warehouse."),
        ],
        embeddings,
        client=client,
        table="langchain_example",
        distance_metric="cosine",
    )
    documents = store.as_retriever(search_kwargs={"k": 1}).invoke(
        "Who owns operations?"
    )
    print(documents[0].page_content)
```

The factory creates a missing table or reuses a compatible one. Stable document
IDs make repeated writes update the same records. To attach without writing,
construct `AntflyVectorStore(client=client, table="langchain_example", embedding=embeddings)`.
Delete this example's dedicated table with `client.drop_table("langchain_example")`
inside the client context when finished. Async factories, retrieval and mutations
are also available.

## More examples

- [Persistent PDF and image RAG](https://github.com/antflydb/langchain-antfly/tree/main/examples/multimodal):
  extract text and images, retrieve persistent parents and original media, and
  generate answers with an existing vision provider.
- [Hybrid search, reranking, graph context and caching](https://github.com/antflydb/langchain-antfly/blob/main/docs/recipes.md).
- [API reference and limitations](https://github.com/antflydb/langchain-antfly/blob/main/docs/api-reference.md).
- [Validation evidence](https://github.com/antflydb/langchain-antfly/blob/main/docs/validation-report.md).

Persistent LangGraph agent memory is available separately in
`langgraph-store-antfly`. This package does not include a chat-model adapter or
an execution checkpointer.
