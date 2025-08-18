"""LangGraph streaming example with per-thread in-memory checkpoints.

Demonstrates how to use `InMemorySaver` so different `thread_id` values
maintain separate conversation histories. Thread 1 will remember the
user's name; thread 2 starts fresh.

Run (Windows cmd.exe):

  python langraph_stream_memory.py

Environment variables (optional):
  LG_BASE_URL  OpenAI-compatible base URL (default http://127.0.0.1:1234/v1)
  LG_API_KEY   API key (default lm-studio)
  LG_MODEL     Model name (default google/gemma-3n-e4b)
"""
from __future__ import annotations

import os
import time
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI, APIConnectionError

logging.basicConfig(level=os.environ.get("LG_LOG_LEVEL", "INFO"))
logger = logging.getLogger("langraph_stream_memory")

# Config
BASE_URL = os.environ.get("LG_BASE_URL", "http://127.0.0.1:1234/v1")
API_KEY = os.environ.get("LG_API_KEY", "lm-studio")
MODEL = os.environ.get("LG_MODEL", "google/gemma-3n-e4b")
RETRY_ATTEMPTS = int(os.environ.get("LG_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF = float(os.environ.get("LG_RETRY_BACKOFF", "0.6"))


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turn: int


client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def llm_node(state: AgentState) -> AgentState:
    """Call the local/OpenAI-compatible chat completion API and append reply."""

    def _role_for(m: AnyMessage) -> str:
        if isinstance(m, HumanMessage):
            return "user"
        if isinstance(m, AIMessage):
            return "assistant"
        t = getattr(m, "type", None)
        return t if t in ("system", "tool", "user", "assistant") else "user"

    payload = {
        "model": MODEL,
        "messages": [{"role": _role_for(m), "content": m.content} for m in state["messages"]],
        "temperature": 0.7,
        "max_tokens": 256,
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


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("llm", llm_node)
    g.add_edge(START, "llm")
    g.add_edge("llm", END)  # single-shot for simplicity in streaming demo
    return g


def run_thread_examples():
    graph_builder = build_graph()
    memory = InMemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    # Thread 1: introduce name
    config1 = {"configurable": {"thread_id": "1"}}
    user_input = "Hi there! My name is Will."  # we expect model to maybe acknowledge
    print("\n--- Thread 1: first message ---")
    events = graph.stream({"messages": [HumanMessage(content=user_input)], "turn": 0}, config1, stream_mode="values")
    for event in events:
        event["messages"][-1].pretty_print()

    # Ask model to remember name in same thread
    print("\n--- Thread 1: recall test ---")
    user_input = "Remember my name?"
    events = graph.stream({"messages": [HumanMessage(content=user_input)], "turn": 0}, config1, stream_mode="values")
    for event in events:
        event["messages"][-1].pretty_print()

    # Different thread id -> no prior history
    print("\n--- Thread 2: fresh context ---")
    config2 = {"configurable": {"thread_id": "2"}}
    events = graph.stream({"messages": [HumanMessage(content=user_input)], "turn": 0}, config2, stream_mode="values")
    for event in events:
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    run_thread_examples()
