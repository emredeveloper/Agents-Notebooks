"""
ART RULER (LLM-as-Judge) demo using LM Studio with two local models.

Architecture:
  - Generator model: qwen3.5-2b  (small, fast)
  - Judge model    : gemma-3-12b (larger, evaluator)

Outputs (saved to data/ directory):
  - trajectories_log.jsonl : All attempts + scores + judge explanations
  - sft_dataset.jsonl      : High-scoring responses as SFT training data
  - run_summary.json       : Run statistics and summary

Usage:
  1. Load both models in LM Studio and start the server
  2. python art_trajectory_demo.py
  3. Use data/ files for fine-tuning with Unsloth, Axolotl, or LM Studio
"""

import json
import asyncio
import os
from datetime import datetime
from dataclasses import dataclass, field
from openai import AsyncOpenAI

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"

GENERATOR_MODEL = "qwen3.5-2b"
JUDGE_MODEL = "google/gemma-3-12b"

DATA_DIR = "data"
REWARD_THRESHOLD = 0.75


@dataclass
class Trajectory:
    messages: list[dict] = field(default_factory=list)
    reward: float = 0.0
    judge_explanation: str = ""
    category: str = ""
    timestamp: str = ""

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return self.messages.copy()

    @property
    def assistant_response(self) -> str:
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                return msg["content"]
        return ""


@dataclass
class Scenario:
    question: str
    rubric: str
    category: str


RULER_SYSTEM_PROMPT = """\
You are an AI evaluation judge. You will be given a question and multiple \
responses to that question. Score each response between 0.0 and 1.0.

Evaluation criteria:
{rubric}

General rules:
- Accuracy: Is the information correct?
- Completeness: Is the question fully answered?
- Clarity: Is the response easy to understand?
- Relativity: Compare responses against each other, give the best one a higher score.

Respond ONLY in the following JSON format, do not write anything else:
{{
  "scores": [
    {{"id": 1, "score": 0.0, "explanation": "..."}},
    {{"id": 2, "score": 0.0, "explanation": "..."}},
    ...
  ]
}}"""


# ─────────────────────────────────────────────────
#  DATA PERSISTENCE
# ─────────────────────────────────────────────────

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def save_trajectory_log(all_trajectories: list[Trajectory]):
    """Append all trajectories to the JSONL log file."""
    path = os.path.join(DATA_DIR, "trajectories_log.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for traj in all_trajectories:
            record = {
                "timestamp": traj.timestamp,
                "category": traj.category,
                "reward": traj.reward,
                "judge_explanation": traj.judge_explanation,
                "messages": traj.messages,
                "generator_model": GENERATOR_MODEL,
                "judge_model": JUDGE_MODEL,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n  [{len(all_trajectories)} trajectories] -> {path}")


def save_sft_dataset(all_trajectories: list[Trajectory], threshold: float = REWARD_THRESHOLD):
    """
    Build an SFT dataset from high-scoring trajectories.
    OpenAI fine-tuning format — compatible with Unsloth, Axolotl, LM Studio.
    """
    path = os.path.join(DATA_DIR, "sft_dataset.jsonl")
    good_ones = [t for t in all_trajectories if t.reward >= threshold]
    good_ones.sort(key=lambda t: t.reward, reverse=True)

    with open(path, "a", encoding="utf-8") as f:
        for traj in good_ones:
            record = {"messages": traj.messages}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(all_trajectories)
    selected = len(good_ones)
    rejected = total - selected

    print(f"  [{selected}/{total} selected, {rejected} rejected (threshold: {threshold})] -> {path}")
    return selected, rejected


def save_run_summary(
    all_trajectories: list[Trajectory],
    selected: int,
    rejected: int,
    scenarios: list[Scenario],
):
    """Save run summary as JSON."""
    path = os.path.join(DATA_DIR, "run_summary.json")

    rewards = [t.reward for t in all_trajectories]
    per_category = {}
    for traj in all_trajectories:
        cat = traj.category
        if cat not in per_category:
            per_category[cat] = []
        per_category[cat].append(traj.reward)

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "reward_threshold": REWARD_THRESHOLD,
        "total_trajectories": len(all_trajectories),
        "selected_for_sft": selected,
        "rejected": rejected,
        "overall_stats": {
            "mean_reward": round(sum(rewards) / len(rewards), 3) if rewards else 0,
            "max_reward": max(rewards) if rewards else 0,
            "min_reward": min(rewards) if rewards else 0,
        },
        "per_category": {
            cat: {
                "count": len(rews),
                "mean": round(sum(rews) / len(rews), 3),
                "max": max(rews),
                "min": min(rews),
            }
            for cat, rews in per_category.items()
        },
        "scenarios": [
            {"category": s.category, "question": s.question, "rubric": s.rubric}
            for s in scenarios
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  [summary] -> {path}")


# ─────────────────────────────────────────────────
#  RULER + ROLLOUT
# ─────────────────────────────────────────────────

async def ruler_judge(
    client: AsyncOpenAI,
    trajectories: list[Trajectory],
    scenario: Scenario,
) -> list[Trajectory]:
    responses_text = ""
    for i, traj in enumerate(trajectories, 1):
        responses_text += f"\n--- Response {i} ---\n{traj.assistant_response}\n"

    judge_messages = [
        {
            "role": "system",
            "content": RULER_SYSTEM_PROMPT.format(rubric=scenario.rubric),
        },
        {
            "role": "user",
            "content": f"QUESTION: {scenario.question}\n\n{responses_text}",
        },
    ]

    response = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=judge_messages,
        max_tokens=1024,
        temperature=0.1,
    )

    raw = response.choices[0].message.content or ""

    json_str = raw
    if "```" in json_str:
        start = json_str.find("{")
        end = json_str.rfind("}") + 1
        if start != -1 and end > start:
            json_str = json_str[start:end]

    try:
        result = json.loads(json_str)
        scores = result.get("scores", [])

        for entry in scores:
            idx = int(entry["id"]) - 1
            if 0 <= idx < len(trajectories):
                trajectories[idx].reward = float(entry["score"])
                trajectories[idx].judge_explanation = entry.get("explanation", "")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  [!] Judge JSON parse error: {e}")
        print(f"  [!] Raw output: {raw[:300]}")
        for traj in trajectories:
            traj.reward = 0.5

    return trajectories


async def rollout(client: AsyncOpenAI, scenario: Scenario) -> Trajectory:
    trajectory = Trajectory(
        category=scenario.category,
        timestamp=datetime.now().isoformat(),
    )

    trajectory.add_message(
        "system",
        "You are an expert assistant. Answer questions accurately and in detail.",
    )
    trajectory.add_message("user", scenario.question)

    response = await client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=trajectory.get_messages(),
        max_tokens=400,
        temperature=0.9,
    )

    assistant_content = response.choices[0].message.content or ""
    trajectory.add_message("assistant", assistant_content)
    return trajectory


async def gather_and_judge(
    client: AsyncOpenAI,
    scenarios: list[Scenario],
    attempts_per_scenario: int = 3,
) -> dict[str, list[Trajectory]]:
    groups: dict[str, list[Trajectory]] = {}

    for scenario in scenarios:
        print(f"\n[*] Scenario: {scenario.category}")
        print(f"    Generating {attempts_per_scenario} responses with {GENERATOR_MODEL}...")

        tasks = [rollout(client, scenario) for _ in range(attempts_per_scenario)]
        trajectories = list(await asyncio.gather(*tasks))

        print(f"    Judging with {JUDGE_MODEL}...")
        trajectories = await ruler_judge(client, trajectories, scenario)

        groups[scenario.category] = trajectories

    return groups


# ─────────────────────────────────────────────────
#  OUTPUT + MAIN LOOP
# ─────────────────────────────────────────────────

def print_results(groups: dict[str, list[Trajectory]], scenarios: list[Scenario]):
    all_rewards = []

    print("\n" + "=" * 70)
    print("  RULER RESULTS")
    print(f"  Generator: {GENERATOR_MODEL}  |  Judge: {JUDGE_MODEL}")
    print("=" * 70)

    for scenario in scenarios:
        trajectories = groups.get(scenario.category, [])
        if not trajectories:
            continue

        rewards = [t.reward for t in trajectories]
        all_rewards.extend(rewards)
        sorted_trajs = sorted(trajectories, key=lambda t: t.reward, reverse=True)

        print(f"\n{'─' * 70}")
        print(f"Question: {scenario.question}")
        print(f"Category: {scenario.category}")
        print(f"Scores: {[t.reward for t in sorted_trajs]}")
        print(f"Average: {sum(rewards) / len(rewards):.3f}")

        for rank, traj in enumerate(sorted_trajs, 1):
            preview = traj.assistant_response[:150]
            if len(traj.assistant_response) > 150:
                preview += "..."
            print(f"\n  #{rank} Score: {traj.reward:.2f}")
            print(f"  Response: {preview}")
            if traj.judge_explanation:
                print(f"  Judge note: {traj.judge_explanation}")

    if all_rewards:
        print(f"\n{'=' * 70}")
        print(f"OVERALL AVERAGE: {sum(all_rewards) / len(all_rewards):.3f}")
        print(f"HIGHEST SCORE: {max(all_rewards):.3f}")
        print(f"LOWEST SCORE: {min(all_rewards):.3f}")
        print(f"TOTAL ATTEMPTS: {len(all_rewards)}")
        print("=" * 70)


async def main():
    client = AsyncOpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key=LMSTUDIO_API_KEY,
    )

    scenarios = [
        Scenario(
            question="What is a decorator in Python and what is it used for? Explain with a code example.",
            rubric=(
                "- Should correctly explain the decorator concept\n"
                "- Should include a working code example\n"
                "- Should mention the @ syntax\n"
                "- Should state use cases"
            ),
            category="python_knowledge",
        ),
        Scenario(
            question="What are the differences between HTTP GET and POST requests?",
            rubric=(
                "- Should explain the purpose of both methods\n"
                "- Should describe the difference in data transmission\n"
                "- Should mention security differences\n"
                "- Should give practical usage examples"
            ),
            category="web_knowledge",
        ),
        Scenario(
            question="What is a branch in Git and why is it used?",
            rubric=(
                "- Should explain the branch concept\n"
                "- Should state why it is used\n"
                "- Should mention the merge process\n"
                "- Should explain advantages in team collaboration"
            ),
            category="git_knowledge",
        ),
    ]

    print("=" * 70)
    print("  ART RULER Demo — LLM-as-Judge Evaluation")
    print(f"  Generator model : {GENERATOR_MODEL}")
    print(f"  Judge model     : {JUDGE_MODEL}")
    print(f"  SFT threshold   : {REWARD_THRESHOLD}")
    print("=" * 70)

    groups = await gather_and_judge(client, scenarios, attempts_per_scenario=3)

    print_results(groups, scenarios)

    all_trajectories: list[Trajectory] = []
    for trajs in groups.values():
        all_trajectories.extend(trajs)

    ensure_data_dir()

    print(f"\n{'=' * 70}")
    print("  DATA PERSISTENCE")
    print(f"{'=' * 70}")

    save_trajectory_log(all_trajectories)
    selected, rejected = save_sft_dataset(all_trajectories)
    save_run_summary(all_trajectories, selected, rejected, scenarios)

    print(f"\n{'─' * 70}")
    print(f"  Total: {len(all_trajectories)} trajectories")
    print(f"  SFT dataset: {selected} high-quality samples (score >= {REWARD_THRESHOLD})")
    print(f"  Rejected: {rejected} low-quality samples")
    print(f"{'─' * 70}")
    print(f"\n  Next step: use data/sft_dataset.jsonl for fine-tuning")
    print(f"  with Unsloth, Axolotl, or LM Studio.")
    print(f"  Data accumulates with each run.\n")


if __name__ == "__main__":
    asyncio.run(main())
