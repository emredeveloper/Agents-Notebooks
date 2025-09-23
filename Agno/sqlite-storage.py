from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.db.sqlite import SqliteDb
from rich.pretty import pprint
import os
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

api_key = os.environ.get("OPENROUTER_API_KEY", "")
model_id = "x-ai/grok-4-fast:free"
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY bulunamadı. Lütfen .env veya ortam değişkeni olarak ekleyin.")

# Sqlite dosyası için dizini güvenceye al
db_dir = Path("tmp")
db_dir.mkdir(parents=True, exist_ok=True)
db_file = str(db_dir / "data.db")


def build_agent(session_id: str, history_runs: int = 3) -> Agent:
    return Agent(
        model=OpenRouter(id=model_id, api_key=api_key),
        session_id=session_id,
        db=SqliteDb(db_file=db_file),
        add_history_to_context=True,
        num_history_runs=history_runs,
    )


def summarize_messages_brief(messages: list, last_n: int = 5) -> list:
    tail = messages[-last_n:]
    summary = []
    for m in tail:
        role = m.get("role") or m.get("type") or "unknown"
        content = m.get("content") or ""
        content_str = content if isinstance(content, str) else str(content)
        content_short = (content_str[:140] + "...") if len(content_str) > 140 else content_str
        summary.append({"role": role, "content": content_short})
    return summary


def main() -> None:
    session_id = os.environ.get("AGNO_SESSION_ID", "demo_session_001")

    # 1) İlk yürütme döngüsü (aynı session_id)
    agent1 = build_agent(session_id=session_id, history_runs=3)
    agent1.print_response("Merhaba, bu bir kalıcı oturum denemesidir.")
    agent1.print_response("Fransa'nın başkenti neresidir?")
    agent1.print_response("Son sorum neydi?")

    # 2) Yeniden başlatma: Ajanı kapatıp aynı session_id ile tekrar oluştur
    agent2 = build_agent(session_id=session_id, history_runs=5)
    agent2.print_response("Yeniden başlattım. Son sorum neydi?")
    agent2.print_response("Oturum geçmişinden son iki etkileşimi kısaca özetler misin?")

    # 3) Oturum içgörüleri: mesaj sayısı ve son mesajların özeti
    messages = agent2.get_messages_for_session()
    print("\n[Oturum Özeti]")
    print(f"session_id: {session_id}")
    print(f"db_file: {db_file}")
    print(f"mesaj_sayisi: {len(messages)}")
    print("son_mesajlar (kısa):")
    pprint(summarize_messages_brief(messages, last_n=5))


if __name__ == "__main__":
    main()