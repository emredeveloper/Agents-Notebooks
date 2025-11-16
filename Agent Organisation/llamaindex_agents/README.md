LlamaIndex Agent Techniques (Simplified)
========================================

This folder collects tiny, deterministic examples that showcase other agentic
patterns without depending on a powerful LLM. Each script uses the same
`llama_index.core.workflow` primitives you saw in the AsyncThink example but
targets a different pattern.

Files
-----

1. `router_agent_workflow.py` – demonstrates a keyword router that chooses between
   latency and compliance specialists. Steps:
   - `router` inspects the query, emits `LatencyTask` and/or `ComplianceTask`.
   - Worker steps store their findings in the context.
   - `summary` returns a `StopEvent` describing which workers ran.

How to run
----------

```bash
cd "Agent Organisation/llamaindex_agents"
python router_agent_workflow.py
```

The console prints the `StopEvent` payload so you can see how routing affected
the output. Because everything is deterministic there is no need for an LLM,
making the example easy to follow.

Extending
---------

Add more workflows here to demonstrate other agent techniques (task graphs,
feedback loops, etc.). Keeping each file short and single-purpose makes it easy
to use them as learning references.
