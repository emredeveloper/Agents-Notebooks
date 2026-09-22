"""Agno replacement for the former Phidata CSV analysis demo."""

import csv
import io
import os

import requests
from agno.agent import Agent
from agno.models.google import Gemini


DATA_URL = "https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv"


def average_movie_rating() -> float:
    """Return the mean Rating from the fixed IMDB demo CSV."""
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    rows = csv.DictReader(io.StringIO(response.text))
    ratings = [float(row["Rating"]) for row in rows if row.get("Rating")]
    if not ratings:
        raise ValueError("CSV contains no movie ratings")
    return sum(ratings) / len(ratings)


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY before running")
    agent = Agent(
        model=Gemini(id="gemini-flash-latest", api_key=api_key),
        tools=[average_movie_rating],
        instructions="Use the rating tool for numeric answers; do not invent data.",
        markdown=True,
    )
    agent.print_response("What is the average rating of movies?")


if __name__ == "__main__":
    main()
