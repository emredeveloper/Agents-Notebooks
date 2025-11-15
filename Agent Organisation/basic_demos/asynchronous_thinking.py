"""
Asynchronous organizer–worker example inspired by AsyncThink.

An organizer agent forks sub-queries to worker agents, awaits their partial results,
and may recursively spawn new organizers before joining everything into the final answer.
"""

import asyncio
from dataclasses import dataclass
from typing import List


@dataclass
class Task:
    name: str
    payload: str


async def worker(task: Task) -> str:
    await asyncio.sleep(0.2)
    return f"{task.name} solved {task.payload}"


async def organizer(task: Task) -> str:
    print(f"[organizer] Planning for {task.payload}")
    # Fork: split into sub-queries that can run concurrently.
    forks = [
        Task(name=f"{task.name}-analysis", payload="math reasoning"),
        Task(name=f"{task.name}-verification", payload="result checking"),
    ]
    # Join worker outputs.
    worker_outputs = await asyncio.gather(*(worker(t) for t in forks))

    # Optional recursive organizer to mimic nested agentic organization.
    sub_org_task = Task(name=f"{task.name}-suborg", payload="latency optimization plan")
    sub_org_output = await nested_organizer(sub_org_task)

    joined = worker_outputs + [sub_org_output]
    return "\n".join(joined)


async def nested_organizer(task: Task) -> str:
    print(f"[sub-organizer] Refining {task.payload}")
    forks = [
        Task(name=f"{task.name}-fork-1", payload="schedule math steps"),
        Task(name=f"{task.name}-fork-2", payload="overlap verification"),
    ]
    results = await asyncio.gather(*(worker(t) for t in forks))
    return f"{task.name} joined -> {' & '.join(results)}"


if __name__ == "__main__":
    prompt_task = Task(name="root", payload="Explain AsyncThink benefits")
    final_answer = asyncio.run(organizer(prompt_task))
    print("Final answer:\n", final_answer)
