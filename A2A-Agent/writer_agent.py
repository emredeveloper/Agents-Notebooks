import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
try:
    from a2a_demo.common import (
        AGENT_CARD_WELL_KNOWN_PATH,
        build_public_agent_card,
        jsonrpc_error,
        jsonrpc_success,
        call_lmstudio,
    )
except ImportError:  # allow running from inside a2a_demo directory
    from common import (
        AGENT_CARD_WELL_KNOWN_PATH,
        build_public_agent_card,
        jsonrpc_error,
        jsonrpc_success,
        call_lmstudio,
    )


PORT = int(os.getenv("WRITER_AGENT_PORT", 8002))
BASE_URL = os.getenv("WRITER_AGENT_BASE_URL", f"http://localhost:{PORT}")
MODEL = os.getenv("LMSTUDIO_MODEL", "qwen3-4b-2507")

app = FastAPI(title="WriterAgent")


@app.get(AGENT_CARD_WELL_KNOWN_PATH)
async def agent_card():
    card = build_public_agent_card(
        name="WriterAgent",
        description="Kısa metinler üreten LLM tabanlı ajan (LM Studio)",
        base_url=BASE_URL,
        streaming=False,
    )
    return JSONResponse(card)


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
    prompt = ""
    for p in parts:
        if p.get("kind") == "text":
            prompt += p.get("text", "") + "\n"

    try:
        lm_response = await call_lmstudio(
            prompt=f"Kısa, öz ve teknik bir Türkçe yanıt yaz.\n\nKonunun özeti: {prompt.strip()}",
            model=MODEL,
            temperature=0.3,
        )
    except Exception as e:
        # JSON-RPC hata cevabı (200 ile dönüyoruz ki orkestratör yakalayıp işleyebilsin)
        return JSONResponse(
            jsonrpc_success(
                id_val,
                {
                    "contextId": message.get("contextId"),
                    "message": {
                        "role": "assistant",
                        "parts": [
                            {
                                "kind": "text",
                                "text": f"WriterAgent LLM hatası: {str(e)}. Lütfen LM Studio sunucusunu ve model adını kontrol edin.",
                            }
                        ],
                    },
                    "error": True,
                },
            )
        )

    result = {
        "contextId": message.get("contextId"),
        "message": {
            "role": "assistant",
            "parts": [{"kind": "text", "text": lm_response}],
        },
    }
    return JSONResponse(jsonrpc_success(id_val, result))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)


