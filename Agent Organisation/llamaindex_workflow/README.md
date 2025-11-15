LlamaIndex AsyncThink Workflow
==============================

We inspected the new [LlamaIndex framework guide](https://developers.llamaindex.ai/python/framework/) and mirrored its organizer–worker ideas with a concrete workflow:

* Use the `Workflow` + `Context.send_event` API to implement Fork and Join steps.
* Model instructions/results as typed `Event` subclasses (`WorkerAssignment`, `WorkerResult`, `OrganizerLog`, `SubOrganizerRequest`).
* Allow recursive organizers so that sub-teams can spawn their own workers, showcasing agentic organization and asynchronous thinking inside the same DAG.
* Aggregate everything with a Join step that emits a `StopEvent` after summarising the workers' notes via the configured LLM.

Directory contents
------------------

- `async_think_workflow.py`: defines `AsyncThinkWorkflow`, helper events, and a runnable demo that now saves its final payload to `artifacts/async_think_summary.json`.

Running the example
-------------------

1. Install dependencies (Ollama LLM adapter included):  
   `pip install llama-index llama-index-llms-ollama`
2. Start Ollama locally (`ollama serve`) and pull the Granite 4 3B chat model:  
   `ollama pull granite4:3b`. You can override the model via `OLLAMA_MODEL`, change host/port with `OLLAMA_BASE_URL`, adjust request timeouts using `OLLAMA_TIMEOUT` (seconds, default `120`), and enlarge workflow runtime with `WORKFLOW_TIMEOUT` (seconds, default `180`).
3. Execute `python async_think_workflow.py` inside this folder. Organizer/worker logs stream to stdout while the workflow runs, and any Ollama hiccups appear as fallback notices.
4. If no Ollama server is reachable the script falls back to the built-in `MockLLM` for demonstration purposes.

What the script does
--------------------

1. `organizer` step stores the user query, forks analysis/verification workers, and spawns a nested organizer for latency/scalability (mirrors the AsyncThink diagram).
2. `nested_organizer` sends extra worker assignments concurrently (heterogeneous pools).
3. `worker` step uses the shared LLM to reason about its assignment; each result is streamed as an `WorkerResult` event.
4. `gather` waits until every worker finishes, synthesises a final response with another LLM call, and emits a `StopEvent` payload containing the raw findings plus a summary.

The printed stdout shows organizer logs, worker emissions, and the joined AsyncThink-style summary.
