"""Persona branching (paralel thread) demo.

Aynı kullanıcı girdisini farklı persona (sistem rolü) mesajlarıyla
ayrı thread'lerde (thread_id) koşturur ve cevapları karşılaştırır.

Özellikler:
- InMemorySaver ile her persona için izole hafıza
- Ortak başlangıç prompt'u
- Cevapların yan yana özetlenmesi
- Basit metin farkı (diff) gösterimi

Kullanım (Windows cmd.exe):
  python langraph_branch_personas.py --prompt "Kısa bir motivasyon cümlesi yaz" --temperature 0.7

İsteğe bağlı env değişkenleri:
  LG_BASE_URL  (varsayılan http://127.0.0.1:1234/v1)
  LG_API_KEY   (varsayılan lm-studio)
  LG_MODEL     (varsayılan google/gemma-3n-e4b)

Not: Model deterministik değilse (temperature > 0), farklılıklar sadece
persona'dan değil örnekleme rastgeleliğinden de kaynaklanabilir.
"""
from __future__ import annotations

import os
import time
import logging
import argparse
import difflib
from typing import Annotated, TypedDict, List, Optional, Iterable

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI, APIConnectionError

# Rich (renkli konsol) – yoksa graceful fallback
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
except ImportError:  # pragma: no cover
    Console = None  # type: ignore

if Console:
    _console = Console()
else:  # type: ignore
    _console = None  # type: ignore

logging.basicConfig(level=os.environ.get("LG_LOG_LEVEL", "INFO"))
logger = logging.getLogger("langraph_branch_personas")

# Ortak konfig
BASE_URL = os.environ.get("LG_BASE_URL", "http://127.0.0.1:1234/v1")
API_KEY = os.environ.get("LG_API_KEY", "lm-studio")
MODEL = os.environ.get("LG_MODEL", "google/gemma-3n-e4b")
RETRY_ATTEMPTS = int(os.environ.get("LG_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF = float(os.environ.get("LG_RETRY_BACKOFF", "0.6"))

# Persona tanımları (id, system mesajı) – TÜMÜ TÜRKÇE ve İngilizce üretmemesi istenir.
PERSONAS = [
    {
        "id": "sicak",
        "system": (
            "Sen sıcak, destekleyici bir asistansın. Yalnızca TÜRKÇE yaz. Emojiyi ölçülü kullan. "
            "Cevap çok kısa (en fazla 1-2 cümle) ve motive edici olsun. İngilizce kelime kullanma."
        ),
    },
    {
        "id": "resmi",
        "system": (
            "Sen resmi ve öz bir asistansın. Yalnızca TÜRKÇE yaz. Tek net motivasyon cümlesi üret. "
            "Sade ve duygusuz bir üslup kullan. İngilizce kullanma."
        ),
    },
    {
        "id": "egitmen",
        "system": (
            "Sen didaktik bir eğitmensin. Yalnızca TÜRKÇE yaz. İç düşüncelerini listeleme, sadece nihai kısa motivasyon cümlesi ver. "
            "İngilizce açıklama ya da çeviri ekleme."
        ),
    },
    {
        "id": "supheci",
        "system": (
            "Sen nazik ama hafif şüpheci bir asistansın. Yalnızca TÜRKÇE yaz. Önce tek cümlelik motivasyon ver, sonra (isteğe bağlı) çok kısa bir ikinci cümlede varsayımları sorgula. İngilizce yazma."
        ),
    },
]


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    turn: int


client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def llm_node(state: AgentState, *, temperature: float, max_tokens: int) -> AgentState:
    """LLM düğümü: mesajları modele iletir, cevap ekler."""

    def _role_for(m: AnyMessage) -> str:
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
            return {
                "messages": list(state["messages"]) + [AIMessage(content=reply)],
                "turn": state.get("turn", 0) + 1,
            }
        except APIConnectionError as e:
            last_err = e
            logger.warning("Bağlantı denemesi %d/%d başarısız: %s", attempt, RETRY_ATTEMPTS, e)
            time.sleep(RETRY_BACKOFF * attempt)
    raise last_err


def build_graph(temperature: float, max_tokens: int):
    g = StateGraph(AgentState)
    # closure param kullanımını sağlamak için lambda ile sarıyoruz
    g.add_node("llm", lambda s: llm_node(s, temperature=temperature, max_tokens=max_tokens))
    g.add_edge(START, "llm")
    g.add_edge("llm", END)  # tek atış
    return g


def last_ai_content(msgs: List[AnyMessage]) -> str:
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m.content
    return "(AI cevabı bulunamadı)"


def make_diff(a: str, b: str, max_lines: int = 80) -> List[str]:
    diff_lines = list(
        difflib.unified_diff(
            a.splitlines(), b.splitlines(), lineterm="", fromfile="A", tofile="B"
        )
    )
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[: max_lines - 1] + ["... (kısaltıldı)"]
    return diff_lines or ["(Fark yok)"]


def render_summary_table(results: list, max_preview: int):
    if not _console:
        # Basit fallback
        print("--- Özet Tablo (Rich yok) ---")
        for r in results:
            preview = r["answer"].strip().replace("\n", " ")
            if len(preview) > max_preview:
                preview = preview[: max_preview - 3] + "..."
            warn = f" {r['warning']}" if r.get("warning") else ""
            print(f"[{r['id']}] -> {preview}{warn}")
        return

    table = Table(title="Persona Özetleri", box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Persona", style="cyan", no_wrap=True)
    table.add_column("Önizleme", style="white")
    table.add_column("Uyarı", style="magenta", no_wrap=True)
    for r in results:
        preview = r["answer"].strip().replace("\n", " ")
        if len(preview) > max_preview:
            preview = preview[: max_preview - 3] + "..."
        warn = r.get("warning") or ""
        table.add_row(r["id"], preview, warn)
    _console.print(table)


def render_reference(base: dict):
    if not _console:
        print(f"=== Referans: {base['id']} ===\n{base['answer']}\n")
        return
    _console.print(Panel(base["answer"], title=f"Referans: {base['id']}", title_align="left", border_style="green"))


def _word_tokens(s: str) -> List[str]:
    # Basit boşluk ayırma; daha iyi sonuç için regex ile kelime + noktalama ayrılabilir.
    return s.split()


def word_level_diff(a: str, b: str) -> Iterable[tuple[str, str]]:
    """Kelime bazlı diff üret (op, token). op: ' ', '-', '+', '~'(değişim bloğu)."""
    import difflib
    a_tokens = _word_tokens(a)
    b_tokens = _word_tokens(b)
    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for t in a_tokens[i1:i2]:
                yield (' ', t)
        elif tag == 'delete':
            for t in a_tokens[i1:i2]:
                yield ('-', t)
        elif tag == 'insert':
            for t in b_tokens[j1:j2]:
                yield ('+', t)
        elif tag == 'replace':
            # Önce silinen sonra eklenen
            for t in a_tokens[i1:i2]:
                yield ('-', t)
            for t in b_tokens[j1:j2]:
                yield ('+', t)


def render_side_by_side(base_text: str, other_text: str, left_title: str, right_title: str):
    if not _console:
        print(f"--- Side-by-side diff ({left_title} | {right_title}) (Rich yok) ---")
        base_lines = base_text.splitlines()
        other_lines = other_text.splitlines()
        width = max(len(l) for l in base_lines) if base_lines else 40
        for i in range(max(len(base_lines), len(other_lines))):
            l = base_lines[i] if i < len(base_lines) else ""
            r = other_lines[i] if i < len(other_lines) else ""
            print(f"{l:<{width}} | {r}")
        return
    from itertools import zip_longest
    table = Table(title=f"Yan Yana: {left_title} ↔ {right_title}", box=box.SIMPLE, show_lines=False)
    table.add_column(left_title, style="white", ratio=1)
    table.add_column(right_title, style="white", ratio=1)
    for a_line, b_line in zip_longest(base_text.splitlines(), other_text.splitlines(), fillvalue=""):
        table.add_row(a_line, b_line)
    _console.print(table)


def render_word_diff(base: dict, other: dict):
    if not _console:
        print(f"--- Word Diff: {base['id']} vs {other['id']} ---")
        for op, tok in word_level_diff(base['answer'], other['answer']):
            print(f"{op}{tok}", end=' ')
        print('\n')
        return
    parts_a = base['answer']
    parts_b = other['answer']
    # Tek panelde ikinci cevabın farkları
    text = Text()
    for op, tok in word_level_diff(parts_a, parts_b):
        if op == ' ':
            text.append(tok + ' ')
        elif op == '-':
            text.append(tok + ' ', style="red")
        elif op == '+':
            text.append(tok + ' ', style="green")
    _console.print(Panel(text, title=f"Kelime Farkları: {base['id']} vs {other['id']}", border_style="purple"))


def render_unified_diff(base: dict, other: dict):
    lines = make_diff(base["answer"], other["answer"])
    if not _console:
        print(f"--- Unified Diff: {base['id']} vs {other['id']} ---")
        print("\n".join(lines))
        print()
        return
    text = Text()
    for ln in lines:
        if ln.startswith("+++") or ln.startswith("---") or ln.startswith("@@"):
            style = "bold yellow"
        elif ln.startswith("+"):
            style = "green"
        elif ln.startswith("-"):
            style = "red"
        else:
            style = "white"
        text.append(ln + "\n", style=style)
    _console.print(Panel(text, title=f"Unified: {base['id']} vs {other['id']}", border_style="blue"))


def render_diff(base: dict, other: dict, mode: str):
    # mode: unified | side | words | all
    if mode in ("unified", "all"):
        render_unified_diff(base, other)
    if mode in ("side", "all"):
        render_side_by_side(base['answer'], other['answer'], base['id'], other['id'])
    if mode in ("words", "all"):
        render_word_diff(base, other)


def run_branching(
    prompt: str,
    temperature: float,
    max_tokens: int,
    personas: list[dict],
    show_diff: bool,
    max_preview: int,
    strict_turkish: bool,
    diff_mode: str,
):
    # Ortak graph + checkpoint (her persona thread_id farklı olduğundan ayrışacak)
    graph_builder = build_graph(temperature=temperature, max_tokens=max_tokens)
    checkpoint = InMemorySaver()
    graph = graph_builder.compile(checkpointer=checkpoint)

    results = []
    for persona in personas:
        thread_id = f"persona-{persona['id']}"
        config = {"configurable": {"thread_id": thread_id}}
        initial = {
            "messages": [
                SystemMessage(content=persona["system"]),
                HumanMessage(content=prompt),
            ],
            "turn": 0,
        }
        logger.info("Persona '%s' çalışıyor (thread_id=%s)", persona["id"], thread_id)
        final_state = graph.invoke(initial, config)
        answer = last_ai_content(final_state["messages"])
        if strict_turkish:
            # Basit İngilizce tespiti: tipik İngilizce kelimeler içeriyor mu?
            eng_tokens = ["the", "and", "you", "your", "Okay", "Success", "learning", "step", "Let's"]
            lowered = answer.lower()
            eng_hits = [w for w in eng_tokens if w.lower() in lowered]
            warning = None
            if eng_hits:
                warning = f"(UYARI: İngilizce öğeler bulundu: {', '.join(eng_hits)})"
                # İstersen burada basit filtre uygulayabilirsin; şimdilik sadece uyarı.
            results.append({
                "id": persona["id"],
                "system": persona["system"],
                "answer": answer,
                "warning": warning,
            })
        else:
            results.append({
                "id": persona["id"],
                "system": persona["system"],
                "answer": answer,
            })

    # Çıktıları göster
    print("\n=== Persona Cevapları (Prompt): ===")
    print(prompt)
    # Özet tablo (renkli)
    if _console:
        _console.rule("Özet")
    else:
        print("\n--- Özet Tablo ---")
    render_summary_table(results, max_preview)

    if show_diff and results:
        base = results[0]
        if _console:
            _console.rule("Referans")
        render_reference(base)
        for r in results[1:]:
            render_diff(base, r, diff_mode)

    return results


def parse_args():
    ap = argparse.ArgumentParser(description="Persona branching karşılaştırma demosu")
    ap.add_argument("--prompt", required=False, default="Kısa bir motivasyon cümlesi yaz.", help="Kullanıcı girişi")
    ap.add_argument("--temperature", type=float, default=0.7, help="Model sıcaklığı")
    ap.add_argument("--max-tokens", type=int, default=256, help="Maksimum yanıt token")
    ap.add_argument("--list-personas", action="store_true", help="Sadece persona listesini yaz ve çık")
    ap.add_argument("--no-diff", action="store_true", help="Diff çıktısını gösterme")
    ap.add_argument("--max-preview-chars", type=int, default=120, help="Özet tablo satır önizleme uzunluğu")
    ap.add_argument("--strict-turkish", action="store_true", help="Basit İngilizce tespiti yap ve uyarı ver")
    ap.add_argument(
        "--diff-mode",
        choices=["unified", "side", "words", "all"],
        default="unified",
        help="Diff gösterim modu (unified=klasik, side=yan yana, words=kelime, all=hepsi)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.list_personas:
        print("Persona listesi:")
        for p in PERSONAS:
            print(f"- {p['id']}: {p['system']}")
        return

    run_branching(
        prompt=args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        personas=PERSONAS,
        show_diff=not args.no_diff,
        max_preview=args.max_preview_chars,
        strict_turkish=args.strict_turkish,
        diff_mode=args.diff_mode,
    )


if __name__ == "__main__":
    main()
