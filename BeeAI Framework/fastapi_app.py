import asyncio
import json
import os
import re
from typing import Dict, Iterable, List, Optional

import requests
from fastapi import FastAPI, Body, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool


# LM Studio / OpenAI-compatible config (align with app.py)
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"))
os.environ.setdefault("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL"))
os.environ.setdefault("OPENAI_API_KEY", os.getenv("LMSTUDIO_API_KEY", "lm-studio"))
DEFAULT_MODEL = os.getenv("LMSTUDIO_MODEL", "google/gemma-3n-e4b")
VERBOSE = os.getenv("BEEAI_VERBOSE", "0") in ("1", "true", "True")

# Wikipedia request headers to avoid throttling
WIKI_HEADERS = {
    "User-Agent": "Agents-Notebooks/1.0 (+https://github.com/emredeveloper)",
    "Accept": "application/json",
}


def build_agent(model_name: str, verbose: bool = False) -> RequirementAgent:
    knowledge_agent = RequirementAgent(
        llm=ChatModel.from_name(f"openai:{model_name}"),
        tools=[ThinkTool(), WikipediaTool()],
        role="Knowledge Specialist",
        instructions="Provide answers to general questions about the world.",
    )

    weather_agent = RequirementAgent(
        llm=ChatModel.from_name(f"openai:{model_name}"),
        tools=[OpenMeteoTool()],
        role="Weather Specialist",
        instructions="Provide weather forecast for a given destination.",
    )

    main_agent = RequirementAgent(
        name="MainAgent",
        llm=ChatModel.from_name(f"openai:{model_name}"),
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
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])] if verbose else [],
    )
    return main_agent


AGENT = build_agent(DEFAULT_MODEL, VERBOSE)


def search_wikipedia_pages(query: str, limit: int = 6) -> List[Dict]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
    }
    try:
        r = requests.get(url, params=params, headers=WIKI_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("query", {}).get("search", [])
    except Exception:
        return []


def get_wikipedia_thumbnails(page_ids: List[int], thumb_size: int = 800) -> Dict[int, Dict]:
    if not page_ids:
        return {}
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "pageids": "|".join(str(pid) for pid in page_ids),
        "prop": "pageimages|info",
        "inprop": "url",
        "pithumbsize": thumb_size,
        "format": "json",
    }
    try:
        r = requests.get(url, params=params, headers=WIKI_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("query", {}).get("pages", {})
    except Exception:
        return {}


def search_wikipedia_with_generator(query: str, limit: int = 6, thumb_size: int = 800) -> List[Dict]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "prop": "pageimages|info",
        "inprop": "url",
        "pithumbsize": thumb_size,
        "format": "json",
    }
    try:
        r = requests.get(url, params=params, headers=WIKI_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        items: List[Dict] = []
        for _, obj in pages.items():
            thumb = obj.get("thumbnail", {})
            src = thumb.get("source")
            if src:
                items.append({
                    "url": src,
                    "title": obj.get("title", ""),
                    "page_url": obj.get("fullurl") or f"https://en.wikipedia.org/wiki/{obj.get('title','').replace(' ', '_')}",
                    "width": thumb.get("width"),
                    "height": thumb.get("height"),
                })
        return items[:limit]
    except Exception:
        return []


def search_wikipedia_rest_title(query: str, limit: int = 6) -> List[Dict]:
    """Fallback using Wikimedia REST API: /w/rest.php/v1/search/title"""
    url = "https://en.wikipedia.org/w/rest.php/v1/search/title"
    params = {"q": query, "limit": limit}
    try:
        r = requests.get(url, params=params, headers=WIKI_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        pages = data.get("pages", [])
        items: List[Dict] = []
        for obj in pages:
            thumb = obj.get("thumbnail") or {}
            src = thumb.get("url")
            if src:
                key = obj.get("key") or obj.get("title", "")
                page_url = f"https://en.wikipedia.org/wiki/{str(key).replace(' ', '_')}"
                items.append({
                    "url": src,
                    "title": obj.get("title", ""),
                    "page_url": page_url,
                    "width": thumb.get("width"),
                    "height": thumb.get("height"),
                })
        return items[:limit]
    except Exception:
        return []


def gather_images(topic: str, limit: int = 6) -> List[Dict]:
    """Collect images for a topic using multiple strategies (blocking)."""
    items: List[Dict] = []
    pages = search_wikipedia_pages(topic, limit=limit)
    if pages:
        page_ids = [p.get("pageid") for p in pages if p.get("pageid")]
        pages_map = get_wikipedia_thumbnails(page_ids)
        for p in pages:
            pid = p.get("pageid")
            title = p.get("title")
            page_obj = pages_map.get(str(pid)) or pages_map.get(pid)
            if isinstance(page_obj, dict):
                thumb = page_obj.get("thumbnail", {})
                src = thumb.get("source")
                if src:
                    items.append({
                        "url": src,
                        "title": title,
                        "page_url": page_obj.get("fullurl") or f"https://en.wikipedia.org/wiki/{str(title).replace(' ', '_')}",
                        "width": thumb.get("width"),
                        "height": thumb.get("height"),
                    })
            if len(items) >= limit:
                return items

    if len(items) < limit:
        more = search_wikipedia_with_generator(topic, limit=limit - len(items))
        items.extend(more)

    if len(items) < limit:
        more2 = search_wikipedia_rest_title(topic, limit=limit - len(items))
        items.extend(more2)

    return items[:limit]


_COMMON_STOP = {
    "if",
    "i",
    "you",
    "your",
    "there",
    "one",
    "next",
    "weekend",
    "weather",
    "what",
    "should",
    "terms",
    "also",
    "tell",
    "me",
    "famous",
    "historical",
    "landmark",
    "in",
    "of",
    "the",
    "and",
    "a",
    "an",
}

# Extra stops to avoid unrelated topics like months/weekdays
_MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
_WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
_GENERIC_STOP = _COMMON_STOP | _MONTHS | _WEEKDAYS | {"today", "tomorrow", "tonight", "morning", "afternoon", "evening"}


def guess_image_topic(prompt: str, answer: Optional[str] = None) -> str:
    m = re.search(r"\b(?:in|at|for)\s+([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)*)", prompt)
    if m:
        return m.group(1)

    def _caps(text: str) -> List[str]:
        parts = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
        out: List[str] = []
        for p in parts:
            tokens = [t for t in p.split() if t.lower() not in _GENERIC_STOP]
            if tokens:
                out.append(" ".join(tokens))
        return out

    cands: List[str] = _caps(prompt)
    if answer:
        cands += _caps(answer)
    seen = set()
    for c in cands:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            return c
    return prompt.strip()[:60]


def _extract_landmark(answer: Optional[str]) -> Optional[str]:
    if not answer:
        return None
    # Try to capture landmark mentions like "landmark is the Colosseum" or similar
    patterns = [
        r"(?:landmark|monument|attraction|site)\s+(?:is|:)?\s*(?:the\s+)?([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,3})",
        r"is\s+(?:a|an)\s+(?:famous\s+)?(?:landmark|monument)\s+(?:called\s+)?([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,3})",
    ]
    for pat in patterns:
        m = re.search(pat, answer)
        if m:
            return m.group(1)
    return None


def extract_topics(prompt: str, answer: Optional[str], max_topics: int = 3) -> List[str]:
    """Extract multiple likely topics (e.g., landmarks or places) from prompt/answer.
    Heuristic: Title Case spans (1-3 words), filter common stop words, dedupe, limit.
    """
    topics: List[str] = []

    def _caps_multi(text: str) -> List[str]:
        # Capture 1 to 3 Title Case words sequences
        parts = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text or "")
        out: List[str] = []
        for p in parts:
            tokens = [t for t in p.split() if t.lower() not in _GENERIC_STOP]
            if tokens:
                out.append(" ".join(tokens))
        return out

    topics: List[str] = []
    landmark = _extract_landmark(answer)
    place = guess_image_topic(prompt, answer)

    # High-priority combos
    if landmark and place:
        topics.append(f"{landmark} {place}")
    if landmark and landmark.lower() not in (_GENERIC_STOP):
        topics.append(landmark)
    if place and place.lower() not in (_GENERIC_STOP):
        topics.append(place)

    # Add other caps spans as backup
    cands: List[str] = []
    if answer:
        cands += _caps_multi(answer)
    cands += _caps_multi(prompt)
    seen = {t.lower() for t in topics}
    for c in cands:
        lc = c.lower()
        if lc in seen:
            continue
        # Filter out single generic tokens like months/days
        toks = c.split()
        if len(toks) == 1 and lc in (_MONTHS | _WEEKDAYS):
            continue
        seen.add(lc)
        topics.append(c)
        if len(topics) >= max_topics:
            break

    if not topics:
        topics = [place or (prompt.strip()[:60] or "")]  # last resort
    return topics


app = FastAPI(title="BeeAI Agent (LM Studio) API")

# Serve static frontend (HTML/CSS/JS)
HERE = os.path.dirname(__file__)
STATIC_DIR = os.path.join(HERE, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    # Serve static index directly (200 OK) and avoid redirects for cleaner logs
    static_index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(static_index):
        return FileResponse(static_index, media_type="text/html", headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        })
    # Fallback to old inline page if static not present
    path = os.path.join(HERE, "index.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return HTMLResponse("<h1>BeeAI Agent API</h1>")


@app.post("/api/ask")
async def ask(payload: Dict = Body(...)):
    prompt = (payload or {}).get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)

    result = await AGENT.run(prompt, expected_output="Helpful and clear response.")
    return {"answer": result.last_message.text}


@app.get("/api/images/stream")
async def image_stream(
    prompt: str = Query("", description="User prompt"),
    answer: Optional[str] = Query(None, description="Agent answer for refinement"),
    limit: int = Query(4, ge=0, le=4),
    delay: float = Query(0.4, ge=0.0, le=5.0),
    topic: Optional[str] = Query(None, description="Override topic for image search"),
    expand: bool = Query(True, description="Extract multiple topics from answer"),
):
    topic_val = (topic or "").strip()
    topics: List[str]
    if topic_val:
        topics = [topic_val]
    else:
        topics = extract_topics(prompt, answer) if expand else [guess_image_topic(prompt, answer)]

    async def gen():
        try:
            total_sent = 0
            remaining = max(min(limit, 4), 0)
            if remaining == 0:
                yield f"data: {json.dumps({"info": "no-limit"})}\n\n"
                return
            if not topics:
                topics_local = [guess_image_topic(prompt, answer)]
            else:
                topics_local = topics
            # Simple round-robin across topics
            ti = 0
            per_min = 1
            global_seen: set[str] = set()
            while remaining > 0 and ti < len(topics_local):
                t = topics_local[ti]
                # Fetch a small batch for each topic to interleave results
                batch = min(max(remaining // (len(topics_local) - ti) or per_min, per_min), remaining)
                imgs = await run_in_threadpool(gather_images, t, batch)
                for item in imgs:
                    url = item.get("url")
                    if not url or url in global_seen:
                        continue
                    global_seen.add(url)
                    payload = json.dumps({"index": total_sent, "topic": t, **item}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    total_sent += 1
                    remaining -= 1
                    if remaining <= 0:
                        break
                    await asyncio.sleep(delay)
                ti += 1
            if total_sent == 0:
                payload = json.dumps({"info": "no-images", "topic": topics_local[0] if topics_local else ""}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)})
            yield f"data: {err}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
