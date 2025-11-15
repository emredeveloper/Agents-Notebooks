"""
Agentic AsyncThink workflow built with LlamaIndex Workflows.

The design mirrors the organizer–worker protocol highlighted in
https://developers.llamaindex.ai/python/framework/: the organizer fans out work
with `Context.send_event`, nested organizers can spawn additional workers, and a
join step aggregates WorkerResult events before emitting a StopEvent.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import httpx
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

try:  # Optional dependency; prefer Ollama for local runs.
    from llama_index.llms.ollama import Ollama
except ImportError:  # pragma: no cover - optional extra
    Ollama = None  # type: ignore


class OrganizerLog(Event):
    """Lightweight streaming signal to mirror Fork/Join tracing."""

    message: str


class WorkerAssignment(Event):
    """Instruction packet that a worker step consumes."""

    worker: str
    focus: str
    instructions: str
    question: str


class WorkerResult(Event):
    """Worker output routed to the gather step and event stream."""

    worker: str
    focus: str
    response: str


class SubOrganizerRequest(Event):
    """Allows recursive organizer behavior."""

    question: str
    objective: str


class AsyncThinkWorkflow(Workflow):
    """Implements the organizer-worker AsyncThink pattern on top of LlamaIndex."""

    WORKER_PROMPT = PromptTemplate(
        (
            "You are {worker}, an expert focusing on {focus}.\n"
            "Main question: {question}\n"
            "Task: {instructions}\n"
            "Respond with short, actionable reasoning."
        )
    )
    SUMMARY_PROMPT = PromptTemplate(
        (
            "You are the chief organizer. Given the main question:\n"
            "{question}\n\n"
            "Fuse the worker findings below into a concise AsyncThink-style summary "
            "highlighting (1) reasoning advantages, (2) latency/accuracy insights, and "
            "(3) future scalability pathways.\n"
            "{notes}"
        )
    )

    def __init__(self, llm: LLM | None = None, **workflow_kwargs: Any) -> None:
        super().__init__(**workflow_kwargs)
        self.llm: LLM = llm or self._default_llm()

    @staticmethod
    def _default_llm() -> LLM:
        """
        Prefer a local Ollama model (default granite4:3b), fall back to MockLLM.

        This keeps the workflow runnable without remote credentials and makes it
        easy to override the local model via environment variables.
        """
        ollama_model = os.getenv("OLLAMA_MODEL", "granite4:3b")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        request_timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
        if Ollama is not None:
            return Ollama(
                model=ollama_model,
                base_url=ollama_base_url,
                request_timeout=request_timeout,
            )

        # Final fallback keeps the workflow demonstrable even without LLM access.
        return MockLLM(max_tokens=96)

    async def _increment_expected(self, ctx: Context, amount: int = 1) -> None:
        expected = await ctx.store.get("expected_workers", default=0)
        await ctx.store.set("expected_workers", expected + amount)

    async def _launch_worker(
        self, ctx: Context, assignment: WorkerAssignment, *, emit: bool = True
    ) -> None:
        await self._increment_expected(ctx)
        ctx.write_event_to_stream(
            OrganizerLog(message=f"Forking task for {assignment.worker} -> {assignment.focus}")
        )
        if emit:
            ctx.send_event(assignment)

    @step
    async def organizer(self, ctx: Context, ev: StartEvent) -> SubOrganizerRequest:
        """Plan the task tree and spawn workers."""
        question = ev.get("question") or "Explain AsyncThink improvements."
        await ctx.store.set("question", question)
        await ctx.store.set("results", [])
        await ctx.store.set("expected_workers", 0)

        ctx.write_event_to_stream(
            OrganizerLog(message=f"Organizer received: {question}")
        )

        assignments = [
            WorkerAssignment(
                worker="analysis-lead",
                focus="core reasoning improvements",
                instructions="Summarize organizer-worker gains over sequential/parallel thinking.",
                question=question,
            ),
            WorkerAssignment(
                worker="verification-lead",
                focus="accuracy and compliance",
                instructions="Check the latency and accuracy numbers quoted for AsyncThink and explain why they hold.",
                question=question,
            ),
        ]

        for assignment in assignments:
            await self._launch_worker(ctx, assignment)

        # Spawn a recursive organizer for latency strategies (return event to satisfy validation).
        ctx.write_event_to_stream(
            OrganizerLog(message="Organizer delegating latency plan to sub-organizer.")
        )
        return SubOrganizerRequest(
            question=question,
            objective="Design latency + scalability workstreams.",
        )

    @step
    async def nested_organizer(self, ctx: Context, ev: SubOrganizerRequest) -> WorkerAssignment | None:
        """Recursive organizer that forks two more specialists."""
        sub_assignments = [
            WorkerAssignment(
                worker="latency-planner",
                focus="latency optimization",
                instructions=(
                    "Describe how asynchronous forks/joins overlap computation to cut latency below parallel thinking."
                ),
                question=ev.question,
            ),
            WorkerAssignment(
                worker="scalability-scout",
                focus="future opportunities",
                instructions="List paths like heterogeneous agent pools and human-AI collaboration made possible by the framework.",
                question=ev.question,
            ),
        ]
        primary: WorkerAssignment | None = None
        for idx, assignment in enumerate(sub_assignments):
            emit_now = idx != 0
            await self._launch_worker(ctx, assignment, emit=emit_now)
            if not emit_now:
                primary = assignment
        return primary

    @step(num_workers=4)
    async def worker(self, ctx: Context, ev: WorkerAssignment) -> WorkerResult:
        """Each worker reasons independently using the shared LLM."""
        response = await self._predict_worker_response(ctx, ev)
        worker_result = WorkerResult(
            worker=ev.worker,
            focus=ev.focus,
            response=response,
        )
        ctx.write_event_to_stream(worker_result)
        return worker_result

    async def _predict_worker_response(self, ctx: Context, ev: WorkerAssignment) -> str:
        """Call the LLM with resilience against local inference hiccups."""
        try:
            return await self.llm.apredict(
                self.WORKER_PROMPT,
                worker=ev.worker,
                focus=ev.focus,
                question=ev.question,
                instructions=ev.instructions,
            )
        except (httpx.ReadTimeout, TimeoutError, Exception) as exc:
            ctx.write_event_to_stream(
                OrganizerLog(
                    message=(
                        f"LLM fallback for {ev.worker} because of a timeout/error: {exc}."
                        " Providing synthesized reasoning instead."
                    )
                )
            )
            return (
                f"{ev.worker} (fallback): Even without the model response we know that "
                f"{ev.focus} benefits from organizer-worker AsyncThink structure. "
                f"Task reminder: {ev.instructions}. Consider how forks overlap work "
                "and joins consolidate results while maintaining correctness checks."
            )

    @step
    async def gather(self, ctx: Context, ev: WorkerResult) -> StopEvent | None:
        """Join worker results and emit the final summary when all workers finish."""
        results: List[Dict[str, str]] = await ctx.store.get("results", default=[])
        results.append({"worker": ev.worker, "focus": ev.focus, "response": ev.response})
        await ctx.store.set("results", results)

        expected = await ctx.store.get("expected_workers", default=0)
        if expected and len(results) == expected:
            notes = "\n\n".join(
                f"{item['worker']} ({item['focus']}): {item['response']}" for item in results
            )
            question = await ctx.store.get("question", default="AsyncThink overview")
            summary = await self.llm.apredict(
                self.SUMMARY_PROMPT,
                question=question,
                notes=notes,
            )
            final_payload = {
                "question": question,
                "summary": summary,
                "worker_details": results,
            }
            ctx.write_event_to_stream(
                OrganizerLog(message="Join step produced final AsyncThink summary.")
            )
            return StopEvent(result=final_payload)
        return None


async def run_workflow(question: str) -> Dict[str, Any]:
    """Helper to execute the workflow and stream intermediate events."""
    workflow_timeout = float(os.getenv("WORKFLOW_TIMEOUT", "180"))
    workflow = AsyncThinkWorkflow(verbose=True, timeout=workflow_timeout)
    handler = workflow.run(question=question)

    async for event in handler.stream_events():
        if isinstance(event, OrganizerLog):
            print(f"[organizer] {event.message}")
        elif isinstance(event, WorkerResult):
            print(f"[worker:{event.worker}] {event.response[:80]}...")

    result: Dict[str, Any] = await handler
    return result


if __name__ == "__main__":
    prompt = "Explain why AsyncThink outperforms sequential and parallel thinking."
    final = asyncio.run(run_workflow(prompt))
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    output_path = artifacts / "async_think_summary.json"
    output_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"AsyncThink çıktısı {output_path} dosyasına kaydedildi.")
