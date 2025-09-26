"""Persona branching (parallel threads) demo.

Sends the same user input to different personas (system roles) in separate
threads (thread_id) and compares the outputs.

Features:
- Isolated memory per persona via InMemorySaver
- Shared initial prompt
- Side-by-side summary of answers
- Simple textual diff display

Usage (Windows cmd.exe):
  python langraph_branch_personas.py --prompt "Write a short motivational sentence" --temperature 0.7

Optional env vars:
  LG_BASE_URL  (default http://127.0.0.1:1234/v1)
  LG_API_KEY   (default lm-studio)
  LG_MODEL     (default google/gemma-3n-e4b)

Note: If the model is non-deterministic (temperature > 0), differences can
come not only from personas but also from sampling randomness.
"""
from __future__ import annotations

import os
import time
import logging
import argparse
import difflib
from typing import Annotated, TypedDict, List, Optional, Iterable

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI, APIConnectionError

# Rich (colored console) – graceful fallback if unavailable
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
except ImportError:  # pragma: no cover
    Console = None  # type: ignore

if Console:
    _console = Console()
else:  # type: ignore
    _console = None  # type: ignore

logging.basicConfig(level=os.environ.get("LG_LOG_LEVEL", "INFO"))
logger = logging.getLogger("langraph_branch_personas")

# Shared config
BASE_URL = os.environ.get("LG_BASE_URL", "http://127.0.0.1:1234/v1")
API_KEY = os.environ.get("LG_API_KEY", "lm-studio")
MODEL = os.environ.get("LG_MODEL", "google/gemma-3n-e4b")
RETRY_ATTEMPTS = int(os.environ.get("LG_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF = float(os.environ.get("LG_RETRY_BACKOFF", "0.6"))

# Persona definitions (id, system message)
PERSONAS = [
    {
        "id": "warm",
        "system": (
            "You are a warm and supportive assistant. Write in Turkish only. Use emojis sparingly. "
            "Keep the reply very short (1–2 sentences) and motivating. Do not use English words."
        ),
    },
    {
        "id": "formal",
        "system": (
            "You are formal and concise. Write in Turkish only. Produce a single clear motivational sentence. "
            "Use a plain, neutral tone. Do not use English."
        ),
    },
    {
        "id": "instructor",
        "system": (
            "You are a didactic instructor. Write in Turkish only. Do not list inner thoughts; produce only the final short motivational sentence. "
            "Do not add English explanations or translations."
        ),
    },
    {
        "id": "skeptical",
        "system": (
            "You are polite but slightly skeptical. Write in Turkish only. First give a one-sentence motivation, then optionally a very short second sentence questioning assumptions. Do not write in English."
        ),
    },
]


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    turn: int


client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def llm_node(state: AgentState, *, temperature: float, max_tokens: int) -> AgentState:
    """LLM node: send messages to the model and append the reply."""

    def _role_for(m: AnyMessage) -> str:
        if isinstance(m, HumanMessage):
            return "user"
        if isinstance(m, AIMessage):
            return "assistant"
        if isinstance(m, SystemMessage):
            return "system"
        t = getattr(m, "type", None)
        return t if t in ("system", "tool", "user", "assistant") else "user"

    payload = {
        "model": MODEL,
        "messages": [{"role": _role_for(m), "content": m.content} for m in state["messages"]],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(**payload)
            reply = response.choices[0].message.content
            return {
                "messages": list(state["messages"]) + [AIMessage(content=reply)],
                "turn": state.get("turn", 0) + 1,
            }
        except APIConnectionError as e:
            last_err = e
            logger.warning("Connection attempt %d/%d failed: %s", attempt, RETRY_ATTEMPTS, e)
            time.sleep(RETRY_BACKOFF * attempt)
    raise last_err


def build_graph(temperature: float, max_tokens: int):
    g = StateGraph(AgentState)
    # Wrap in a lambda to close over parameters
    g.add_node("llm", lambda s: llm_node(s, temperature=temperature, max_tokens=max_tokens))
    g.add_edge(START, "llm")
    g.add_edge("llm", END)  # single shot
    return g


def last_ai_content(msgs: List[AnyMessage]) -> str:
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m.content
    return "(No AI answer found)"


def make_diff(a: str, b: str, max_lines: int = 80) -> List[str]:
    diff_lines = list(
        difflib.unified_diff(
            a.splitlines(), b.splitlines(), lineterm="", fromfile="A", tofile="B"
        )
    )
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[: max_lines - 1] + ["... (truncated)"]
    return diff_lines or ["(No differences)"]


def render_summary_table(results: list, max_preview: int):
    if not _console:
        # Simple fallback
        print("--- Summary Table (Rich missing) ---")
        for r in results:
            preview = r["answer"].strip().replace("\n", " ")
            if len(preview) > max_preview:
                preview = preview[: max_preview - 3] + "..."
            warn = f" {r['warning']}" if r.get("warning") else ""
            print(f"[{r['id']}] -> {preview}{warn}")
        return

    table = Table(title="Persona Summaries", box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Persona", style="cyan", no_wrap=True)
    table.add_column("Preview", style="white")
    table.add_column("Warning", style="magenta", no_wrap=True)
    for r in results:
        preview = r["answer"].strip().replace("\n", " ")
        if len(preview) > max_preview:
            preview = preview[: max_preview - 3] + "..."
        warn = r.get("warning") or ""
        table.add_row(r["id"], preview, warn)
    _console.print(table)


def render_reference(base: dict):
    if not _console:
        print(f"=== Reference: {base['id']} ===\n{base['answer']}\n")
        return
    _console.print(Panel(base["answer"], title=f"Reference: {base['id']}", title_align="left", border_style="green"))


def _word_tokens(s: str) -> List[str]:
    # Simple whitespace split; for better results, split words/punct with regex.
    return s.split()


def word_level_diff(a: str, b: str) -> Iterable[tuple[str, str]]:
    """Produce word-level diff (op, token). op: ' ', '-', '+', '~'(change block)."""
    import difflib
    a_tokens = _word_tokens(a)
    b_tokens = _word_tokens(b)
    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for t in a_tokens[i1:i2]:
                yield (' ', t)
        elif tag == 'delete':
            for t in a_tokens[i1:i2]:
                yield ('-', t)
        elif tag == 'insert':
            for t in b_tokens[j1:j2]:
                yield ('+', t)
        elif tag == 'replace':
            # Önce silinen sonra eklenen
            for t in a_tokens[i1:i2]:
                yield ('-', t)
            for t in b_tokens[j1:j2]:
                yield ('+', t)


def render_side_by_side(base_text: str, other_text: str, left_title: str, right_title: str):
    if not _console:
        print(f"--- Side-by-side diff ({left_title} | {right_title}) (Rich missing) ---")
        base_lines = base_text.splitlines()
        other_lines = other_text.splitlines()
        width = max(len(l) for l in base_lines) if base_lines else 40
        for i in range(max(len(base_lines), len(other_lines))):
            l = base_lines[i] if i < len(base_lines) else ""
            r = other_lines[i] if i < len(other_lines) else ""
            print(f"{l:<{width}} | {r}")
        return
    from itertools import zip_longest
    table = Table(title=f"Side-by-side: {left_title} ↔ {right_title}", box=box.SIMPLE, show_lines=False)
    table.add_column(left_title, style="white", ratio=1)
    table.add_column(right_title, style="white", ratio=1)
    for a_line, b_line in zip_longest(base_text.splitlines(), other_text.splitlines(), fillvalue=""):
        table.add_row(a_line, b_line)
    _console.print(table)


def render_word_diff(base: dict, other: dict):
    if not _console:
        print(f"--- Word Diff: {base['id']} vs {other['id']} ---")
        for op, tok in word_level_diff(base['answer'], other['answer']):
            print(f"{op}{tok}", end=' ')
        print('\n')
        return
    parts_a = base['answer']
    parts_b = other['answer']
    # Tek panelde ikinci cevabın farkları
    text = Text()
    for op, tok in word_level_diff(parts_a, parts_b):
        if op == ' ':
            text.append(tok + ' ')
        elif op == '-':
            text.append(tok + ' ', style="red")
        elif op == '+':
            text.append(tok + ' ', style="green")
    _console.print(Panel(text, title=f"Word Differences: {base['id']} vs {other['id']}", border_style="purple"))


def render_unified_diff(base: dict, other: dict):
    lines = make_diff(base["answer"], other["answer"])
    if not _console:
        print(f"--- Unified Diff: {base['id']} vs {other['id']} ---")
        print("\n".join(lines))
        print()
        return
    text = Text()
    for ln in lines:
        if ln.startswith("+++") or ln.startswith("---") or ln.startswith("@@"):
            style = "bold yellow"
        elif ln.startswith("+"):
            style = "green"
        elif ln.startswith("-"):
            style = "red"
        else:
            style = "white"
        text.append(ln + "\n", style=style)
    _console.print(Panel(text, title=f"Unified: {base['id']} vs {other['id']}", border_style="blue"))


def render_diff(base: dict, other: dict, mode: str):
    # mode: unified | side | words | all
    if mode in ("unified", "all"):
        render_unified_diff(base, other)
    if mode in ("side", "all"):
        render_side_by_side(base['answer'], other['answer'], base['id'], other['id'])
    if mode in ("words", "all"):
        render_word_diff(base, other)


def run_branching(
    prompt: str,
    temperature: float,
    max_tokens: int,
    personas: list[dict],
    show_diff: bool,
    max_preview: int,
    strict_turkish: bool,
    diff_mode: str,
):
    # Shared graph + checkpoint (each persona has a unique thread_id)
    graph_builder = build_graph(temperature=temperature, max_tokens=max_tokens)
    checkpoint = InMemorySaver()
    graph = graph_builder.compile(checkpointer=checkpoint)

    results = []
    for persona in personas:
        thread_id = f"persona-{persona['id']}"
        config = {"configurable": {"thread_id": thread_id}}
        initial = {
            "messages": [
                SystemMessage(content=persona["system"]),
                HumanMessage(content=prompt),
            ],
            "turn": 0,
        }
        logger.info("Running persona '%s' (thread_id=%s)", persona["id"], thread_id)
        final_state = graph.invoke(initial, config)
        answer = last_ai_content(final_state["messages"])
        if strict_turkish:
            # Simple English detection: does it contain common English words?
            eng_tokens = ["the", "and", "you", "your", "Okay", "Success", "learning", "step", "Let's"]
            lowered = answer.lower()
            eng_hits = [w for w in eng_tokens if w.lower() in lowered]
            warning = None
            if eng_hits:
                warning = f"(WARNING: English tokens detected: {', '.join(eng_hits)})"
                # Optional: apply a simple filter; for now, only warn.
            results.append({
                "id": persona["id"],
                "system": persona["system"],
                "answer": answer,
                "warning": warning,
            })
        else:
            results.append({
                "id": persona["id"],
                "system": persona["system"],
                "answer": answer,
            })

    # Display outputs
    print("\n=== Persona Answers (Prompt): ===")
    print(prompt)
    # Summary table (colored)
    if _console:
        _console.rule("Summary")
    else:
        print("\n--- Summary Table ---")
    render_summary_table(results, max_preview)

    if show_diff and results:
        base = results[0]
        if _console:
            _console.rule("Reference")
        render_reference(base)
        for r in results[1:]:
            render_diff(base, r, diff_mode)

    return results


def parse_args():
    ap = argparse.ArgumentParser(description="Persona branching comparison demo")
    ap.add_argument("--prompt", required=False, default="Write a short motivational sentence.", help="User input")
    ap.add_argument("--temperature", type=float, default=0.7, help="Model temperature")
    ap.add_argument("--max-tokens", type=int, default=256, help="Max response tokens")
    ap.add_argument("--list-personas", action="store_true", help="List personas and exit")
    ap.add_argument("--no-diff", action="store_true", help="Do not show diff output")
    ap.add_argument("--max-preview-chars", type=int, default=120, help="Summary preview length")
    ap.add_argument("--strict-turkish", action="store_true", help="Detect English leakage and warn")
    ap.add_argument(
        "--diff-mode",
        choices=["unified", "side", "words", "all"],
        default="unified",
        help="Diff display mode (unified, side, words, or all)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.list_personas:
        print("Persona listesi:")
        for p in PERSONAS:
            print(f"- {p['id']}: {p['system']}")
        return

    run_branching(
        prompt=args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        personas=PERSONAS,
        show_diff=not args.no_diff,
        max_preview=args.max_preview_chars,
        strict_turkish=args.strict_turkish,
        diff_mode=args.diff_mode,
    )


if __name__ == "__main__":
    main()
