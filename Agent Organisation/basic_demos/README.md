The Era of Agentic Organization
================================

Microsoft Research's AsyncThink describes how large language models can learn to organize their internal reasoning into concurrently executable structures via an organizer–worker protocol. The core ideas are:

* **Organizer–Worker protocol** – an organizer agent dynamically forks sub-queries to worker agents and joins their results, all through plain text generations.
* **Learning to organize** – models receive a two-stage training pipeline (format fine-tuning, then Group Relative Policy Optimization reinforcement learning) that rewards correctness, protocol compliance, and thinking concurrency.
* **Accuracy–latency frontier** – AsyncThink reaches the same 38.7 % accuracy as parallel thinking on AIME-24 but with 28 % lower latency (1,468 vs 2,048).
* **Generalization** – even when trained on countdown tasks only, the agents zero-shot transfer asynchronous thinking to new domains such as Sudoku, graph theory, and genetics.
* **Scalability** – future directions include massive heterogeneous worker pools, recursive organizers (workers that become organizers), and human–AI collaborative setups.

This folder contains simple Python snippets that mirror the diagram:

1. `sequential_thinking.py` – a single worker receives the prompt, reasons, and emits the answer.
2. `parallel_thinking.py` – the same prompt is sent to multiple workers concurrently and combined at the end.
3. `asynchronous_thinking.py` – an organizer forks/joins sub-queries to workers in an agentic organization style.

Run any script with `python <file>` to see the behavior.
