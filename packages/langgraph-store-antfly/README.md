# Antfly memory for LangGraph

Persistent, namespaced long-term memory for LangGraph agents using Antfly.
Supports sync/async operations, namespace enumeration, filtering and optional
semantic search. Python 3.11 or newer. Apache-2.0 licensed.

**Release candidate:** `0.1.0rc2`. Uses the matching `langchain-antfly` candidate
and published `antfly-sdk>=0.2.1`. Antfly `0.3.0-rc.5` is the tested server
baseline. See the [server setup](https://github.com/antflydb/langchain-antfly#first-result).

## Install

```bash
pip install langgraph-store-antfly==0.1.0rc2
```

This installs the Python adapters and their dependencies. It does not install or
start the Antfly server. Plain key/value memory needs no embedding model.

## Remember a preference

```python
import os

from antfly import AntflyClient
from langgraph_store_antfly import AntflyStore

with AntflyClient(
    base_url=os.environ["ANTFLY_URL"], token=os.getenv("ANTFLY_TOKEN")
) as client:
    # Create once for this example; on later runs, attach instead.
    store = AntflyStore.create(client=client, table="agent_preferences")
    store.put(("users", "alice"), "drink", {"value": "tea"})

    # A new instance reads the same persisted records.
    attached = AntflyStore(client=client, table="agent_preferences")
    print(attached.get(("users", "alice"), "drink").value)
    assert attached.get(("users", "bob"), "drink") is None
```

For cleanup, drop the example's dedicated table with
`client.drop_table("agent_preferences")` inside the client context. Select
namespaces from trusted application identity; namespaces separate records but
do not replace server authorization. Client lifetimes belong to the caller.

## Use inside an agent

Pass the store to LangChain's `create_agent(store=...)` and access it in tools
through `ToolRuntime.store`. The
[two-user agent example](https://github.com/antflydb/langchain-antfly/blob/main/examples/agent_memory.py)
demonstrates that composition with trusted user context and sync/async execution.
Install the example's LangChain and chat-provider dependencies separately.

See the [memory recipes](https://github.com/antflydb/langchain-antfly/blob/main/docs/recipes.md)
for semantic indexing, search, and async operations, and the
[validation report](https://github.com/antflydb/langchain-antfly/blob/main/docs/validation-report.md)
for tested versions and restart/isolation evidence.

This is long-term memory storage. It does not implement a LangGraph execution
checkpointer or per-item TTL.
