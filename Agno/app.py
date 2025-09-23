from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=OpenRouter(id="x-ai/grok-4-fast:free",api_key = "sk-or-v1-...."),
    tools=[HackerNewsTools()],
    markdown=True,
)
agent.print_response("Summarize the top 5 stories on hackernews", stream=True)