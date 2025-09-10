import asyncio
import os
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool

# Configure LM Studio via OpenAI-compatible API defaults
# You can override these via environment variables before running the app.
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"))
# Some clients use OPENAI_API_BASE instead of OPENAI_BASE_URL
os.environ.setdefault("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL"))
os.environ.setdefault("OPENAI_API_KEY", os.getenv("LMSTUDIO_API_KEY", "lm-studio"))
MODEL_NAME = os.getenv("LMSTUDIO_MODEL", "google/gemma-3n-e4b")
VERBOSE = os.getenv("BEEAI_VERBOSE", "0") in ("1", "true", "True")

console = Console()


async def main() -> None:
    knowledge_agent = RequirementAgent(
        llm=ChatModel.from_name(f"openai:{MODEL_NAME}"),
        tools=[ThinkTool(), WikipediaTool()],
        role="Knowledge Specialist",
        instructions="Provide answers to general questions about the world.",
    )

    weather_agent = RequirementAgent(
        llm=ChatModel.from_name(f"openai:{MODEL_NAME}"),
        tools=[OpenMeteoTool()],
        role="Weather Specialist",
        instructions="Provide weather forecast for a given destination.",
    )

    main_agent = RequirementAgent(
        name="MainAgent",
        llm=ChatModel.from_name(f"openai:{MODEL_NAME}"),
        tools=[
            ThinkTool(),
            HandoffTool(
                knowledge_agent,
                name="KnowledgeLookup",
                description="Consult the Knowledge Agent for general questions.",
            ),
            HandoffTool(
                weather_agent,
                name="WeatherLookup",
                description="Consult the Weather Agent for forecasts.",
            ),
    ],
    # Log tool calls only when verbose is enabled
    middlewares=[GlobalTrajectoryMiddleware(included=[Tool])] if VERBOSE else [],
    )

    question = "If I travel to Rome next weekend, what should I expect in terms of weather, and also tell me one famous historical landmark there?"
    console.print(Rule("BeeAI LM Studio Demo"))
    console.print(Panel.fit(question, title="User", border_style="cyan"))

    try:
        response = await main_agent.run(question, expected_output="Helpful and clear response.")
        console.print(Panel(response.last_message.text, title="Agent", border_style="green"))
    except FrameworkError as err:
        console.print(Panel(err.explain(), title="Error", border_style="red"))


if __name__ == "__main__":
    asyncio.run(main())