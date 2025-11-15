"""
Parallel thinking example.

Multiple workers independently reason about the same question, then a reducer combines their outputs.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List


def worker(worker_id: int, question: str) -> str:
    return f"worker-{worker_id} sees {question} from a unique angle"


def reducer(outputs: List[str]) -> str:
    combined = " | ".join(outputs)
    return f"Combined insight: {combined}"


if __name__ == "__main__":
    prompt = "How does AsyncThink improve latency?"
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, i, prompt) for i in range(1, 4)]
        answers = [f.result() for f in futures]
    print("\n".join(answers))
    print(reducer(answers))
