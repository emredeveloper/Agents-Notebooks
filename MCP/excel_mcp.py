"""
Excel MCP Server + Ollama Integration
======================================
Bu script:
1. excel-mcp-server'ı stdio transport ile başlatır
2. Ollama'ya MCP tool'larını tanıtır
3. Ollama'nın tool calling özelliğiyle Excel işlemleri yaptırır
4. Çıktıları terminale yazdırır

Gereksinimler:
  pip install mcp ollama
  uvx excel-mcp-server  (veya: pip install excel-mcp-server)
  Ollama çalışıyor olmalı: ollama serve
  Model: ollama pull llama3.2  (veya qwen2.5, mistral vb.)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# ── Renkli terminal çıktısı için ANSI kodları ──────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

def banner(text: str, color: str = CYAN) -> None:
    line = "─" * 60
    print(f"\n{color}{BOLD}{line}{RESET}")
    print(f"{color}{BOLD}  {text}{RESET}")
    print(f"{color}{BOLD}{line}{RESET}\n")

def info(msg: str)    -> None: print(f"{CYAN}[INFO]{RESET}  {msg}")
def success(msg: str) -> None: print(f"{GREEN}[OK]{RESET}    {msg}")
def warn(msg: str)    -> None: print(f"{YELLOW}[WARN]{RESET}  {msg}")
def error(msg: str)   -> None: print(f"{RED}[ERROR]{RESET} {msg}")

# ── Çalışma dizini ─────────────────────────────────────────────────────────
WORKDIR = Path(__file__).parent / "excel_output"
WORKDIR.mkdir(exist_ok=True)
EXCEL_FILE = str(WORKDIR / "demo_report.xlsx")

# ── Ollama modeli ──────────────────────────────────────────────────────────
OLLAMA_MODEL = "qwen3.5:9b"
OLLAMA_HOST  = "http://localhost:11434"   # Ollama API adresi

# ═══════════════════════════════════════════════════════════════════════════
# 1. MCP Client — excel-mcp-server'a bağlan, tool listesini al
# ═══════════════════════════════════════════════════════════════════════════

async def get_mcp_tools():
    """
    MCP stdio transport ile excel-mcp-server'a bağlanır ve
    tool listesini döndürür.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        error("'mcp' paketi bulunamadı. Lütfen: pip install mcp")
        sys.exit(1)

    server_params = StdioServerParameters(
        command="uvx",
        args=["excel-mcp-server", "stdio"],
        env={**os.environ, "EXCEL_FILES_PATH": str(WORKDIR)},
    )

    banner("MCP Server'a Bağlanılıyor", CYAN)
    info(f"Komut: uvx excel-mcp-server stdio")
    info(f"Excel çalışma dizini: {WORKDIR}")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            tools = tools_response.tools
            success(f"{len(tools)} tool yüklendi.")
            for t in tools:
                print(f"  {DIM}• {t.name}{RESET}")
            return [t.model_dump() for t in tools]


# ═══════════════════════════════════════════════════════════════════════════
# 2. MCP Tool Çağrısı — Ollama'nın istediği tool'u çalıştır
# ═══════════════════════════════════════════════════════════════════════════

async def call_mcp_tool(tool_name: str, tool_args: dict) -> str:
    """
    Verilen tool_name ile tool_args'ı MCP server'a gönderir,
    sonucu string olarak döndürür.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        return "mcp paketi yüklü değil."

    server_params = StdioServerParameters(
        command="uvx",
        args=["excel-mcp-server", "stdio"],
        env={**os.environ, "EXCEL_FILES_PATH": str(WORKDIR)},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_args)

    # Sonucu düz metin olarak birleştir
    parts = []
    for content in result.content:
        if hasattr(content, "text"):
            parts.append(content.text)
        else:
            parts.append(str(content))
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MCP tool şemasını Ollama formatına dönüştür
# ═══════════════════════════════════════════════════════════════════════════

def mcp_tools_to_ollama_format(mcp_tools: list) -> list:
    """
    MCP tool tanımlarını Ollama'nın beklediği OpenAI-uyumlu
    function-calling formatına çevirir.
    """
    ollama_tools = []
    for t in mcp_tools:
        schema = t.get("inputSchema") or t.get("input_schema") or {}
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": schema,
            },
        })
    return ollama_tools


# ═══════════════════════════════════════════════════════════════════════════
# 4. Agentic döngü — Ollama + MCP tool calling
# ═══════════════════════════════════════════════════════════════════════════

async def run_ollama_with_mcp(prompt: str, mcp_tools: list) -> str:
    """
    Verilen prompt'u Ollama'ya gönderir; model tool çağrısı yaparsa
    MCP server üzerinden çalıştırır ve sonucu tekrar modele verir.
    Bu döngü model düz yanıt verene kadar devam eder.
    """
    try:
        import ollama as ollama_lib
    except ImportError:
        error("'ollama' paketi bulunamadı. Lütfen: pip install ollama")
        sys.exit(1)

    # Explicit host ile client oluştur (default host sorun çıkarabilir)
    client = ollama_lib.Client(host=OLLAMA_HOST)

    ollama_tools = mcp_tools_to_ollama_format(mcp_tools)
    messages = [{"role": "user", "content": prompt}]

    banner("Ollama ile Konuşma Başlıyor", YELLOW)
    info(f"Model  : {OLLAMA_MODEL}")
    info(f"Host   : {OLLAMA_HOST}")
    info(f"Prompt : {prompt}\n")

    max_rounds = 10   # sonsuz döngüye karşı güvenlik
    for round_no in range(1, max_rounds + 1):
        info(f"Tur {round_no}: Ollama'ya istek gönderiliyor…")

        response = client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=ollama_tools,
        )

        # response: ChatResponse nesnesi — hem dict hem attr erişimi destekler
        msg = response.message if hasattr(response, "message") else response["message"]

        # messages listesi dict beklediği için dönüştür
        msg_dict = {
            "role": msg.role if hasattr(msg, "role") else msg["role"],
            "content": msg.content if hasattr(msg, "content") else msg.get("content", ""),
        }
        # tool_calls varsa ekle
        raw_tool_calls = (
            msg.tool_calls if hasattr(msg, "tool_calls") else msg.get("tool_calls")
        ) or []
        if raw_tool_calls:
            msg_dict["tool_calls"] = raw_tool_calls
        messages.append(msg_dict)

        if not raw_tool_calls:
            # Model düz metin yanıt verdi — bitiş
            final = msg_dict.get("content", "")
            success("Model yanıtı alındı.")
            return final

        # Tool çağrılarını işle
        for tc in raw_tool_calls:
            # tc: ToolCall nesnesi veya dict
            if hasattr(tc, "function"):
                fn_obj = tc.function
                name = fn_obj.name if hasattr(fn_obj, "name") else fn_obj["name"]
                args = fn_obj.arguments if hasattr(fn_obj, "arguments") else fn_obj.get("arguments", {})
            else:
                fn_obj = tc.get("function", {})
                name = fn_obj.get("name", "")
                args = fn_obj.get("arguments", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            print(f"\n{YELLOW}[TOOL CALL]{RESET} {BOLD}{name}{RESET}")
            print(f"  Args: {json.dumps(args, ensure_ascii=False, indent=2)}")

            result_text = await call_mcp_tool(name, args)

            print(f"{GREEN}[TOOL RESULT]{RESET}")
            # Uzun sonuçları kırp
            preview = result_text[:600] + ("…" if len(result_text) > 600 else "")
            print(f"  {preview}\n")

            # Tool sonucunu sohbet geçmişine ekle
            messages.append({
                "role": "tool",
                "content": result_text,
            })

    warn("Maksimum tur sayısına ulaşıldı.")
    return messages[-1].get("content", "")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main — Demo senaryosu
# ═══════════════════════════════════════════════════════════════════════════

def check_ollama() -> bool:
    """
    Ollama HTTP API'sine ping atar; cevap gelirse True döner.
    """
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=4)
        req.read()
        return True
    except Exception:
        return False


async def main():
    banner("Excel MCP Server  ×  Ollama Demo", GREEN)

    # ── Adım 0: Ollama bağlantı kontrolü ───────────────────────────────────
    banner("Ollama Bağlantı Kontrolü", YELLOW)
    if check_ollama():
        success(f"Ollama API erişilebilir → {OLLAMA_HOST}")
    else:
        error("Ollama API'ye bağlanılamadı!")
        print(f"""
{YELLOW}Çözüm:{RESET}
  1) Yeni bir CMD penceresi aç ve şunu çalıştır:
     {BOLD}ollama serve{RESET}
  2) Bu terminal açık kalsın, sonra bu scripti tekrar başlat.

  Veya Ollama masaüstü uygulamasını başlat (sistem tepsisinde 🦙 ikonu).
""")
        sys.exit(1)

    # ── Adım 1: Tool listesini al ──────────────────────────────────────────
    mcp_tools = await get_mcp_tools()

    # ── Adım 2: Prompt hazırla ─────────────────────────────────────────────
    prompt = f"""
You are an Excel automation assistant.
The Excel file path to use is: {EXCEL_FILE}

Please do the following steps IN ORDER:
1. Create a new workbook at the path above.
2. Create a worksheet named "Sales".
3. Write the following sales data to the "Sales" sheet starting at A1:
   - Headers: Product, Q1, Q2, Q3, Q4, Total
   - Row 1:   Laptop,    45000, 52000, 48000, 61000, (sum formula)
   - Row 2:   Phone,     32000, 38000, 41000, 45000, (sum formula)
   - Row 3:   Tablet,    18000, 21000, 19000, 24000, (sum formula)
   - Row 4:   Monitor,   12000, 15000, 13000, 17000, (sum formula)
4. Read the data back from the "Sales" sheet and show me the result.

Use the available tools to complete each step.
""".strip()

    # ── Adım 3: Ollama ajanını çalıştır ───────────────────────────────────
    try:
        final_answer = await run_ollama_with_mcp(prompt, mcp_tools)
    except Exception as exc:
        error(f"Ollama çalıştırılırken hata: {exc}")
        raise

    # ── Adım 4: Sonucu göster ──────────────────────────────────────────────
    banner("Model'in Nihai Yanıtı", GREEN)
    print(final_answer)

    # ── Adım 5: Dosya bilgisi ──────────────────────────────────────────────
    if Path(EXCEL_FILE).exists():
        size_kb = Path(EXCEL_FILE).stat().st_size / 1024
        banner(f"Excel Dosyası Oluşturuldu ✔", GREEN)
        success(f"Konum : {EXCEL_FILE}")
        success(f"Boyut : {size_kb:.1f} KB")
    else:
        warn(f"Excel dosyası beklenen konumda bulunamadı: {EXCEL_FILE}")

    banner("Demo Tamamlandı", CYAN)


if __name__ == "__main__":
    # Windows'ta asyncio event loop politikası
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
