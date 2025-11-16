Agent Organisation Workspace
============================

This folder now holds two sets of examples inspired by the AsyncThink organizer–worker paradigm:

- `basic_demos/`: the earlier plain-Python sequential, parallel, and asynchronous toy snippets plus the original summary README.
- `llamaindex_workflow/`: the new LlamaIndex-based implementation that uses the Workflow API (`Context.send_event`, typed `Event`s, and recursive organizers) to reproduce Fork/Join behavior described in [Microsoft Research's AsyncThink announcement](https://developers.llamaindex.ai/python/framework/).
- `llamaindex_agents/`: lightweight Router/worker-style demos that rely on the same workflow primitives to illustrate other agent techniques in a simple, runnable form.

See each subfolder README for usage details.
