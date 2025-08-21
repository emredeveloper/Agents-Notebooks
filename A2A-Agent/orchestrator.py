import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

try:
    from a2a_demo.common import (
        AGENT_CARD_WELL_KNOWN_PATH,
        build_public_agent_card,
        jsonrpc_error,
        jsonrpc_success,
    )
except ImportError:  # allow running from inside a2a_demo directory
    from common import (
        AGENT_CARD_WELL_KNOWN_PATH,
        build_public_agent_card,
        jsonrpc_error,
        jsonrpc_success,
    )


PORT = int(os.getenv("ORCH_PORT", 8100))
BASE_URL = os.getenv("ORCH_BASE_URL", f"http://localhost:{PORT}")
MATH_URL = os.getenv("MATH_AGENT_URL", "http://localhost:8001")
WRITER_URL = os.getenv("WRITER_AGENT_URL", "http://localhost:8002")

app = FastAPI(title="Orchestrator")


@app.get(AGENT_CARD_WELL_KNOWN_PATH)
async def agent_card():
    card = build_public_agent_card(
        name="Orchestrator",
        description="Kullanıcı girdisini matematik ve yazım ajanlarına yönlendirir.",
        base_url=BASE_URL,
        streaming=False,
    )
    return JSONResponse(card)


async def send_message(agent_base: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(f"{agent_base}/", json=payload)
            # WriterAgent hata durumunda da 200 dönebilir; 4xx/5xx ise yakala ve JSON-RPC hata üret.
            if resp.status_code >= 400:
                return {
                    "jsonrpc": "2.0",
                    "id": payload.get("id", "0"),
                    "error": {
                        "code": -32000,
                        "message": f"Upstream {agent_base} HTTP {resp.status_code}",
                        "data": resp.text,
                    },
                }
            return resp.json()
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": payload.get("id", "0"),
                "error": {"code": -32001, "message": f"Upstream error: {str(e)}"},
            }


@app.post("/")
async def rpc_root(request: Request):
    body = await request.json()
    method = body.get("method")
    id_val = str(body.get("id", "0"))

    if method != "agent.sendMessage":
        return JSONResponse(jsonrpc_error(id_val, -32601, "Method not found"), status_code=404)

    params = body.get("params", {})
    message = params.get("message", {})
    parts = message.get("parts", [])

    user_text = "\n".join([p.get("text", "") for p in parts if p.get("kind") == "text"]) 

    # Basit yönlendirme: sayı/işlem geçiyorsa önce MathAgent, ardından WriterAgent ile güzel bir paragraf
    wants_math = any(s in user_text for s in ["+", "-", "*", "/", "topla", "cikar", "carp", "bol"]) 

    history: list = []

    if wants_math:
        math_payload = {
            "jsonrpc": "2.0",
            "id": "math-1",
            "method": "agent.sendMessage",
            "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": user_text}]}}
        }
        math_resp = await send_message(MATH_URL, math_payload)
        history.append({"agent": "math", "response": math_resp})
        try:
            math_text = math_resp.get("result", {}).get("message", {}).get("parts", [{}])[0].get("text", "")
        except Exception:
            math_text = str(math_resp)
        user_text = f"Hesap sonucu: {math_text}\n\nİstenen anlatım: {user_text}"

    writer_payload = {
        "jsonrpc": "2.0",
        "id": "writer-1",
        "method": "agent.sendMessage",
        "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": user_text}]}}
    }
    writer_resp = await send_message(WRITER_URL, writer_payload)
    history.append({"agent": "writer", "response": writer_resp})

    # Nihai sonuç: Writer yanıtı veya hata mesajı
    if "result" in writer_resp:
        final_message = writer_resp.get("result", {}).get("message")
    else:
        # Hata durumunda anlamlı bir metin dön.
        err_text = writer_resp.get("error", {}).get("message", "Bilinmeyen hata")
        final_message = {
            "role": "assistant",
            "parts": [
                {
                    "kind": "text",
                    "text": f"WriterAgent yanıtı alınamadı: {err_text}. Lütfen WriterAgent ve LM Studio'yu kontrol edin.",
                }
            ],
        }
    result = {
        "contextId": message.get("contextId"),
        "message": final_message,
        "metadata": {"trace": history},
    }
    return JSONResponse(jsonrpc_success(id_val, result))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)


