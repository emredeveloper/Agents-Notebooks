import os

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.hackernews import HackerNewsTools

def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY before running")
    agent = Agent(
        model=OpenRouter(id="x-ai/grok-4-fast:free", api_key=api_key),
        tools=[HackerNewsTools()],
        markdown=True,
    )
    agent.print_response("Summarize the top 5 stories on hackernews", stream=True)


if __name__ == "__main__":
    main()
