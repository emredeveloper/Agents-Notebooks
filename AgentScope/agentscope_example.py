"""Minimal AgentScope examples using a local Ollama model."""
import asyncio
import os

from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
from agentscope.formatter import OllamaChatFormatter
from agentscope.agent import ReActAgent, AgentBase
from agentscope.model import OllamaChatModel


async def react_agent_example() -> None:
    """Create and run a ReAct agent against the local Ollama server."""
    model = OllamaChatModel(
        model_name=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        stream=False,
    )

    toolkit = None  # no tools used in this minimal example

    agent = ReActAgent(
        name="Jarvis",
        sys_prompt="You are Jarvis, a helpful assistant.",
        model=model,
        formatter=OllamaChatFormatter(),
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
    print("Running ReAct example (local Ollama)...")
    asyncio.run(react_agent_example())
    print("\nRunning custom AgentBase example...")
    asyncio.run(custom_agent_example())


if __name__ == "__main__":
    main()
