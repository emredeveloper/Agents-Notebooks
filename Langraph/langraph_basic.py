"""
LangGraph agent that uses a locally hosted LLM via an OpenAI-compatible endpoint.
The agent keeps looping until the LLM's reply contains the word "done".
"""
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from openai import OpenAI, APIConnectionError
import os
import time
import logging
import argparse


# ------------------------- Configuration ------------------

# Environment-configurable values with sensible defaults.
BASE_URL = os.environ.get("LG_BASE_URL", "http://127.0.0.1:1234/v1")
API_KEY = os.environ.get("LG_API_KEY", "lm-studio")
MODEL = os.environ.get("LG_MODEL", "google/gemma-3n-e4b")
MAX_TURNS = int(os.environ.get("LG_MAX_TURNS", "50"))

# Small retry settings for transient network failures.
RETRY_ATTEMPTS = int(os.environ.get("LG_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF = float(os.environ.get("LG_RETRY_BACKOFF", "0.6"))

logging.basicConfig(level=os.environ.get("LG_LOG_LEVEL", "INFO"))
logger = logging.getLogger("langraph_app")


# ------------------------- State -------------------------

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # conversation history
    turn: int                                             # number of turns elapsed


# ------------------------- LLM client --------------------

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def llm_node(state: AgentState) -> AgentState:
    """LangGraph node: call the LLM and append the reply to state."""
    def _role_for(m: AnyMessage) -> str:
        # Map internal message objects to OpenAI-compatible roles.
        if isinstance(m, HumanMessage):
            return "user"
        if isinstance(m, AIMessage):
            return "assistant"
        # Some messages might already carry an appropriate type/name
        t = getattr(m, "type", None)
        if t in ("system", "tool", "user", "assistant"):
            return t
        # Default to 'user' for unknown message types
        return "user"

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
            # Append assistant reply to the existing history
            new_msgs = list(state["messages"]) + [AIMessage(content=reply)]
            return {"messages": new_msgs, "turn": state["turn"] + 1}
        except APIConnectionError as e:
            last_err = e
            logger.warning("Connection attempt %d/%d failed: %s", attempt, RETRY_ATTEMPTS, e)
            time.sleep(RETRY_BACKOFF * attempt)
    # If we reach here, all retries failed
    logger.error("Failed to contact LLM after %d attempts: %s", RETRY_ATTEMPTS, last_err)
    raise last_err

# ------------------------- Routing -----------------------

def should_continue(state: AgentState) -> str:
    """Stop if the reply contains 'done'; otherwise continue."""
    last = state["messages"][-1]
    # Stop if we've reached the configured maximum number of turns.
    if state.get("turn", 0) >= MAX_TURNS:
        logger.info("Max turns (%d) reached, stopping.", MAX_TURNS)
        return "end"
    return "end" if "done" in last.content.lower() else "continue"

# ------------------------- Graph -------------------------

workflow = StateGraph(AgentState)

workflow.add_node("llm", llm_node)
workflow.add_edge(START, "llm")
workflow.add_conditional_edges(
    "llm",
    should_continue,
    {"continue": "llm", "end": END}
)

app = workflow.compile()

# ------------------------- Run ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LangGraph demo agent")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS, help="Maximum conversational turns before giving up")
    args = parser.parse_args()

    # Honor CLI max-turns for this run
    MAX_TURNS = args.max_turns

    initial = {
        "messages": [HumanMessage(content="Count from 1 to 5")],
        "turn": 0,
    }

    logger.info("Starting run (model=%s, base_url=%s)", MODEL, BASE_URL)

    # Increase recursion limit for complex graphs to avoid GraphRecursionError
    final = app.invoke(initial, {"recursion_limit": 100})

    # Print conversation
    print("\nConversation:")
    for msg in final["messages"]:
        role = "👤" if isinstance(msg, HumanMessage) else "🤖"
        print(f"{role} {msg.content}")