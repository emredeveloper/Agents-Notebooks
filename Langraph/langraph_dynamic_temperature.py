"""Dynamic temperature selection demo.

Goal: Classify the user's prompt type (factual, creative, reasoning, code, translation, etc.)
with simple heuristics and automatically set the model's temperature.

Example strategy:
- Short factual/info query               -> low temperature (0.1 - 0.2)
- Deep reasoning / multi-step            -> medium (0.5)
- Creative writing / story / slogan      -> high (0.8 - 0.95)
- Translation request                    -> very low (0.0 - 0.15)
- Code explanation / sample code         -> low-medium (0.2 - 0.35)

This script can run the same prompt in two modes:
1. Dynamically selected temperature
2. Fixed temperature provided by the user (with --compare)

Usage:
  python langraph_dynamic_temperature.py --prompt "Write a short motivational sentence" --compare

Optional env vars:
  LG_BASE_URL  (default http://127.0.0.1:1234/v1)
  LG_API_KEY   (default lm-studio)
  LG_MODEL     (default google/gemma-3n-e4b)
"""
from __future__ import annotations

import os
import re
import time
import logging
import argparse
from dataclasses import dataclass
from typing import Annotated, TypedDict, List

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from openai import OpenAI, APIConnectionError

logging.basicConfig(level=os.environ.get("LG_LOG_LEVEL", "INFO"))
logger = logging.getLogger("langraph_dynamic_temperature")

# Shared config
BASE_URL = os.environ.get("LG_BASE_URL", "http://127.0.0.1:1234/v1")
API_KEY = os.environ.get("LG_API_KEY", "lm-studio")
MODEL = os.environ.get("LG_MODEL", "google/gemma-3n-e4b")
RETRY_ATTEMPTS = int(os.environ.get("LG_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF = float(os.environ.get("LG_RETRY_BACKOFF", "0.6"))

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    turn: int


@dataclass
class ClassificationResult:
    category: str
    temperature: float
    rationale: str


FAKTUEL_KELIMELER = [
    "nedir", "kim", "ne zaman", "hangi", "tanım", "kaç", "listele", "özeti", "özetle",
]
YARATICI_KELIMELER = [
    "hikaye", "hikâye", "öykü", "şiir", "roman", "yaratıcı", "senaryo", "metafor", "slogan", "epik",
]
AKIL_KELIMELER = [
    "adım adım", "mantık", "kanıtla", "çıkarım", "derinle", "analiz et", "sebep", "gerekçe",
]
KOD_KELIMELER = [
    "python", "kod", "snippet", "örnek kod", "function", "class", "hata", "stacktrace", "algoritma",
]
CEVIRI_KALIPLARI = [
    re.compile(r"\b(çevir|çeviri|translate)\b", re.IGNORECASE),
    re.compile(r"\b(English|Türkçe'ye|Türkçeye)\b", re.IGNORECASE),
]


def classify_prompt(prompt: str) -> ClassificationResult:
    p_lower = prompt.lower()
    tokens = re.findall(r"\w+", p_lower)
    length = len(tokens)

    def contains_any(words):
        return any(w in p_lower for w in words)

    # Çeviri önce
    for pat in CEVIRI_KALIPLARI:
        if pat.search(prompt):
            return ClassificationResult(
                category="ceviri",
                temperature=0.05,
                rationale="Çeviri ifadesi tespit edildi, tutarlılık için düşük sıcaklık."
            )

    # Yaratıcı
    if contains_any(YARATICI_KELIMELER):
        base = 0.85
        # Daha uzun yaratıcı istek -> biraz yükselt
        if length > 40:
            base = min(0.95, base + 0.05)
        return ClassificationResult(
            category="yaratıcı",
            temperature=base,
            rationale="Yaratıcı üretim kelimeleri bulundu (hikaye/şiir vb.)."
        )

    # Akıl yürütme
    if contains_any(AKIL_KELIMELER) or ("?" in prompt and length > 25):
        return ClassificationResult(
            category="akil_yurutme",
            temperature=0.5,
            rationale="Akıl yürütme / çok adımlı ipuçları veya uzun soru."
        )

    # Kod / teknik
    if contains_any(KOD_KELIMELER) or any(ch in prompt for ch in ["{", "}", "()", "[]"]):
        return ClassificationResult(
            category="kod",
            temperature=0.25,
            rationale="Kod / teknik anahtar kelimeler bulundu."
        )

    # Faktüel kısa sorgu
    if contains_any(FAKTUEL_KELIMELER) or (length < 8 and prompt.endswith("?")):
        return ClassificationResult(
            category="faktuel",
            temperature=0.15,
            rationale="Kısa faktüel sorgu tespit edildi."
        )

    # Varsayılan genel
    # Orta kısalıkta içerik: hafif yaratıcılık için 0.35-0.45
    temp = 0.4
    rationale = "Varsayılan kategori (belirgin yaratıcı/kod/çeviri sinyali yok)."
    # Çok kısa ise düşür
    if length <= 3:
        temp = 0.25
        rationale += " Çok kısa olduğu için sıcaklık düşürüldü."
    # Çok uzun serbest metin -> daha yaratıcı olabilir
    elif length > 80:
        temp = 0.55
        rationale += " Uzun serbest metin, çeşitlilik için yükseltildi."
    return ClassificationResult(category="genel", temperature=temp, rationale=rationale)


def llm_node(state: AgentState, *, temperature: float, max_tokens: int) -> AgentState:
    def _role_for(m: AnyMessage) -> str:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        if isinstance(m, HumanMessage):
            return "user"
        if isinstance(m, AIMessage):
            return "assistant"
        if isinstance(m, SystemMessage):
            return "system"
        t = getattr(m, "type", None)
        return t if t in ("system", "tool", "user", "assistant") else "user"

    payload = {
        "model": MODEL,
        "messages": [{"role": _role_for(m), "content": m.content} for m in state["messages"]],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(**payload)
            reply = response.choices[0].message.content
            return {"messages": list(state["messages"]) + [AIMessage(content=reply)], "turn": state.get("turn", 0) + 1}
        except APIConnectionError as e:
            last_err = e
            logger.warning("Bağlantı denemesi %d/%d başarısız: %s", attempt, RETRY_ATTEMPTS, e)
            time.sleep(RETRY_BACKOFF * attempt)
    raise last_err


def build_graph(dynamic_temp: float, max_tokens: int):
    g = StateGraph(AgentState)
    g.add_node("llm", lambda s: llm_node(s, temperature=dynamic_temp, max_tokens=max_tokens))
    g.add_edge(START, "llm")
    g.add_edge("llm", END)
    return g


def invoke_once(prompt: str, system: str, temperature: float, max_tokens: int):
    graph_builder = build_graph(temperature, max_tokens)
    app = graph_builder.compile()
    initial = {
        "messages": [SystemMessage(content=system), HumanMessage(content=prompt)],
        "turn": 0,
    }
    final = app.invoke(initial)
    answer = final["messages"][-1].content if final["messages"] else "(cevap yok)"
    return answer


def parse_args():
    ap = argparse.ArgumentParser(description="Dinamik temperature demo")
    ap.add_argument("--prompt", required=True, help="Kullanıcı girişi")
    ap.add_argument("--system", default="Kısa ve net cevap ver.", help="Sistem mesajı")
    ap.add_argument("--max-tokens", type=int, default=256, help="Maks yanıt token")
    ap.add_argument("--fixed-temperature", type=float, default=0.7, help="Karşılaştırma için sabit sıcaklık")
    ap.add_argument("--compare", action="store_true", help="Dinamik ve sabiti yan yana göster")
    ap.add_argument("--show-rationale", action="store_true", help="Seçim gerekçesini yazdır")
    return ap.parse_args()


def main():
    args = parse_args()
    cls = classify_prompt(args.prompt)
    if args.show_rationale:
        print(f"Sınıflandırma: {cls.category}  -> sıcaklık={cls.temperature:.2f}\nGerekçe: {cls.rationale}\n")
    else:
        print(f"[Dinamik] kategori={cls.category} sıcaklık={cls.temperature:.2f}")

    dyn_answer = invoke_once(args.prompt, args.system, cls.temperature, args.max_tokens)
    print("=== Dinamik Yanıt ===")
    print(dyn_answer)

    if args.compare:
        fixed_answer = invoke_once(args.prompt, args.system, args.fixed_temperature, args.max_tokens)
        print("\n=== Sabit Yanıt (temperature={:.2f}) ===".format(args.fixed_temperature))
        print(fixed_answer)

        # Basit kıyas: uzunluk ve eşleşen kelime oranı
        dyn_tokens = dyn_answer.split()
        fix_tokens = fixed_answer.split()
        overlap = len(set(dyn_tokens) & set(fix_tokens))
        ratio = overlap / max(1, len(set(dyn_tokens) | set(fix_tokens)))
        print("\n--- Kısa İstatistikler ---")
        print(f"Dinamik uzunluk: {len(dyn_tokens)} kelime")
        print(f"Sabit   uzunluk: {len(fix_tokens)} kelime")
        print(f"Kelime örtüşme oranı: {ratio:.2f}")


if __name__ == "__main__":
    main()
