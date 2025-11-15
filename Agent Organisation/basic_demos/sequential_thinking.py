"""
Sequential thinking example.

One worker receives the input question, runs through every step, and returns the result.
"""

import time


def sequential_worker(question: str) -> str:
    steps = [
        "Understand the context",
        "Plan a single chain of reasoning",
        "Execute the plan",
        "Produce the answer",
    ]
    for step in steps:
        print(f"[worker] {step}...")
        time.sleep(0.3)
    return f"Single-path answer for: {question}"


if __name__ == "__main__":
    prompt = "What is AsyncThink?"
    output = sequential_worker(prompt)
    print(output)
