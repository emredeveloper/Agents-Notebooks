"""Minimal AgentScope examples.

This script shows two tiny examples based on the Agentscope quickstart docs:
1) Using a provided ReAct agent with an in-memory memory and a mock model.
2) Creating a custom AgentBase subclass and running it.

These examples are intentionally minimal and do not call any external APIs.
They are synchronous wrappers around the async API for convenience.
"""
import asyncio
import os

from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeChatFormatter
from agentscope.agent import ReActAgent, AgentBase
from agentscope.model import OllamaChatModel


async def react_agent_example() -> None:
    """Create and run a minimal ReActAgent-like example.

    NOTE: This example uses a fake/simple model by mocking the model call with a
    small coroutine that returns a Msg-like object. In a real setup you would
    pass a real model implementation (e.g. DashScopeChatModel or another
    agentscope.model implementation) and provide API keys as needed.
    """

    # Try to use Ollama locally (default host http://localhost:11434). If
    # the environment variable OLLAMA_HOST is set, use that host. If Ollama is
    # not available, fall back to a simple fake model so the example still runs.
    host = "http://localhost:11434"

    try:
        ollama_model = OllamaChatModel(model_name="llama3.2:3b", host=host, stream=False)
        model = ollama_model
    except Exception as e:
        # If Ollama initialization fails, print the error so the user can
        # diagnose why the real model wasn't used. Then fall back to a fake model.
        print(f"Warning: failed to initialize OllamaChatModel ({e}). Falling back to FakeModel.")

        # Fallback fake model
        class FakeModel:
            # Agentscope model wrappers expect attributes like `stream`.
            stream = False

            async def __call__(self, messages, **kwargs):
                # Return an object that has a `content` attribute which is a list
                # of text blocks (matching AgentScope's ChatResponse.content).
                class R:
                    def __init__(self, content):
                        self.content = content

                # messages might be a list of dicts — create a simple echo
                last = None
                if isinstance(messages, list) and messages:
                    # messages usually are dicts with 'content' field
                    last = messages[-1].get("content") if isinstance(messages[-1], dict) else str(messages[-1])
                else:
                    last = str(messages)

                text = "Echoing (fake model): " + str(last)
                # content as a list of text blocks
                return R(content=[{"type": "text", "text": text}])

        model = FakeModel()

    toolkit = None  # no tools used in this minimal example

    agent = ReActAgent(
        name="Jarvis",
        sys_prompt="You are Jarvis, a helpful assistant.",
        model=model,
        formatter=DashScopeChatFormatter(),
        memory=InMemoryMemory(),
        toolkit=toolkit,
    )

    user_msg = Msg(name="user", content="Hello, run a quick test.", role="user")
    await agent(user_msg)


class MyAgent(AgentBase):
    """A tiny custom agent that replies with a canned response."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Friday"
        self.sys_prompt = "You're Friday, a friendly assistant."
        # Use the formatter and memory helpers from Agentscope
        self.formatter = DashScopeChatFormatter()
        self.memory = InMemoryMemory()

    async def reply(self, msg: Msg | list[Msg] | None) -> Msg:
        # store incoming message
        await self.memory.add(msg)

        # prepare a simple prompt by concatenating contents from memory
        mem = await self.memory.get_memory()
        prompt_text = " ".join(m.content for m in mem if getattr(m, 'content', None))

        # create a reply message
        response_text = "Hi — I got: " + (prompt_text or "(no message)")
        out = Msg(name=self.name, content=response_text, role="assistant")

        # record and print
        await self.memory.add(out)
        await self.print(out)
        return out


async def custom_agent_example() -> None:
    agent = MyAgent()
    msg = Msg(name="user", content="Who are you?", role="user")
    await agent(msg)


def main():
    """Run both examples sequentially."""
    print("Running ReAct example (fake model)...")
    asyncio.run(react_agent_example())
    print("\nRunning custom AgentBase example...")
    asyncio.run(custom_agent_example())


if __name__ == "__main__":
    main()
