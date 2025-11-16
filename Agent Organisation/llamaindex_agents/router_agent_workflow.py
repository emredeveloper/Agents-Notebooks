"""
Keyword-based router agent implemented with the LlamaIndex Workflow primitives.

It demonstrates:
* Routing: a `router` step inspects the query and chooses which specialized worker
  should own it by emitting the correct event.
* Tool-like workers: `LatencyInsights` and `ComplianceInsights` mimic independent
  experts that can run concurrently and store their findings in context state.
* Final synthesis: a `summary` step joins whatever workers ran and returns a
  minimal StopEvent payload.

No external LLM is required; the workers run deterministic helper functions so
the script stays easy to understand but still shows the event-driven agent style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


class LatencyTask(Event):
    question: str


class ComplianceTask(Event):
    question: str


class RouterDecision(Event):
    question: str
    route: str


@dataclass
class RouterPlan:
    """Simple plan state to keep track of which workers fired."""

    triggered_workers: List[str]


class RouterAgentWorkflow(Workflow):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @step
    async def router(self, ctx: Context, ev: StartEvent) -> RouterDecision:
        """Route the request by keyword heuristics."""
        query = ev.get("question", "")
        query_lower = query.lower()
        await ctx.store.set("plan", RouterPlan(triggered_workers=[]))

        if "latency" in query_lower or "hız" in query_lower:
            ctx.send_event(LatencyTask(question=query))
            route = "latency"
        if any(keyword in query_lower for keyword in ["compliance", "doğruluk", "accuracy"]):
            ctx.send_event(ComplianceTask(question=query))
            route = "compliance"
        else:
            route = "analysis"
            ctx.send_event(LatencyTask(question=query))

        return RouterDecision(question=query, route=route)

    @step
    async def latency_worker(self, ctx: Context, ev: LatencyTask) -> None:
        plan: RouterPlan = await ctx.store.get("plan")
        plan.triggered_workers.append("latency")
        await ctx.store.set(
            "latency_report",
            "AsyncThink azaltılmış bekleme süreleri sunar çünkü bağımsız alt görevler aynı anda işlenir.",
        )
        await ctx.store.set("plan", plan)

    @step
    async def compliance_worker(self, ctx: Context, ev: ComplianceTask) -> None:
        plan: RouterPlan = await ctx.store.get("plan")
        plan.triggered_workers.append("compliance")
        await ctx.store.set(
            "compliance_report",
            "Organizer-worker protokolü doğruluk kontrollerini ayrı iş parçalarında çalıştırarak hataları yakalar.",
        )
        await ctx.store.set("plan", plan)

    @step
    async def summary(self, ctx: Context, ev: RouterDecision) -> StopEvent:
        plan: RouterPlan = await ctx.store.get("plan")
        latency = await ctx.store.get("latency_report", default="Latent analiz çalıştırılmadı.")
        compliance = await ctx.store.get("compliance_report", default="Uygunluk analizi yapılmadı.")
        result: Dict[str, Any] = {
            "question": ev.question,
            "route": ev.route,
            "workers_triggered": plan.triggered_workers,
            "latency": latency,
            "compliance": compliance,
        }
        return StopEvent(result=result)


async def _demo() -> None:
    workflow = RouterAgentWorkflow(verbose=True)
    handler = workflow.run(question="AsyncThink latency ve accuracy açısından neden iyi?")
    final = await handler
    print(final)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_demo())
