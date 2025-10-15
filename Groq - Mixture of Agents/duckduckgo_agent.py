import os
from typing import List

from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain_community.tools import (
    DuckDuckGoSearchRun as CommunityDuckDuckGoSearchRun,
    DuckDuckGoSearchResults as CommunityDuckDuckGoSearchResults,
)


def create_duckduckgo_search_agent(model_name: str = "llama-3.1-8b-instant") -> AgentExecutor:
    """Sadece DuckDuckGo arama araçlarını kullanan basit bir ReAct ajanı oluşturur."""
    search_run = CommunityDuckDuckGoSearchRun()
    search_results = CommunityDuckDuckGoSearchResults()

    tools: List[Tool] = [
        Tool(
            name="duckduckgo_search",
            func=search_run.run,
            description="DuckDuckGo ile web araması yapar ve özet döndürür",
        ),
        Tool(
            name="duckduckgo_results_json",
            func=search_results.run,
            description="DuckDuckGo ilk sayfa sonuçlarını JSON benzeri formatta döndürür",
        ),
    ]

    llm = ChatGroq(
        model=model_name,
        temperature=0.2,
        max_tokens=700,
        timeout=60,
    )

    prompt = ChatPromptTemplate.from_template(
        """
Sen bir web araştırma ajanısın. Sadece DuckDuckGo araçlarını kullan.

Mevcut araçlar:
{tools}

Kullanabileceğin araçlar: {tool_names}

Format (uzun ve yapılandırmaya uygun içerik üret):
Question: {input}
Thought: Ne yapmam gerekiyor?
Action: [araç_adı]
Action Input: [araç_girişi]
Observation: [araç_çıkışı]
Final Answer: [120-180 kelimelik zengin özet]

Kurallar:
- Önce "duckduckgo_results_json" ile ilk sayfa sonuçlarını al, ardından özet üret.
- En fazla 3 kaynağı özetin sonuna ekle (Başlık: URL).
- Araç adı ve girişini tam yaz.

{agent_scratchpad}
"""
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
        max_execution_time=60,
    )
    return executor


def run_query(query: str) -> None:
    agent = create_duckduckgo_search_agent()
    result = agent.invoke({"input": query})

    output = result.get("output") if isinstance(result, dict) else None

    import json as _json
    from datetime import datetime as _dt

    def _collect_sources(q: str, n: int = 5):
        results_tool = CommunityDuckDuckGoSearchResults()
        raw = results_tool.run(q)
        links = []
        if isinstance(raw, list):
            for item in raw[:n]:
                url = (item.get("link") or item.get("href") or item.get("url") or "").strip()
                title = (item.get("title") or item.get("source") or url).strip()
                if url:
                    links.append({"title": title, "url": url})
        elif isinstance(raw, dict) and "results" in raw:
            for item in raw["results"][:n]:
                url = (item.get("link") or item.get("href") or item.get("url") or "").strip()
                title = (item.get("title") or item.get("source") or url).strip()
                if url:
                    links.append({"title": title, "url": url})
        return links

    summary_text = None
    if output and "Agent stopped due to iteration limit or time limit" not in str(output):
        summary_text = str(output)
    else:
        # Fallback: araçlardan özet çek
        try:
            summary_tool = CommunityDuckDuckGoSearchRun()
            summary_text = summary_tool.run(query)
        except Exception:
            summary_text = ""

    sources = []
    try:
        sources = _collect_sources(query, n=5)
    except Exception:
        sources = []

    structured = {
        "query": query,
        "summary": (summary_text or "").strip(),
        "sources": sources,
        "tool": "duckduckgo",
        "generated_at": _dt.utcnow().isoformat() + "Z",
    }

    print(_json.dumps(structured, ensure_ascii=False, indent=2))

    # Kaynak linkleri (ilk 3)
    try:
        results_tool = CommunityDuckDuckGoSearchResults()
        raw = results_tool.run(query)
        links = []
        if isinstance(raw, list):
            for item in raw[:3]:
                url = (item.get("link") or item.get("href") or item.get("url") or "").strip()
                title = (item.get("title") or item.get("source") or url).strip()
                if url:
                    links.append(f"- {title}: {url}")
        elif isinstance(raw, dict) and "results" in raw:
            for item in raw["results"][:3]:
                url = (item.get("link") or item.get("href") or item.get("url") or "").strip()
                title = (item.get("title") or item.get("source") or url).strip()
                if url:
                    links.append(f"- {title}: {url}")
        if links:
            print("\nKaynaklar:\n" + "\n".join(links))
    except Exception:
        pass


if __name__ == "__main__":
    import sys

    GROQ_API_KEY  = "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"
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    query = " ".join(sys.argv[1:]).strip() or "Python ML kütüphaneleri nelerdir?"
    run_query(query)


