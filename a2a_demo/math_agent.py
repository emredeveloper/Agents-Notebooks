import os
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


PORT = int(os.getenv("MATH_AGENT_PORT", 8001))
BASE_URL = os.getenv("MATH_AGENT_BASE_URL", f"http://localhost:{PORT}")

app = FastAPI(title="MathAgent")


@app.get(AGENT_CARD_WELL_KNOWN_PATH)
async def agent_card():
    card = build_public_agent_card(
        name="MathAgent",
        description="Basit matematik soruları yanıtlayan ajan",
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
    text = ""
    for p in parts:
        if p.get("kind") == "text":
            text += p.get("text", "") + "\n"

    answer = "Sonuç hesaplanamadı"
    try:
        # çok basit: '10 USD kaç TRY' gibi stringlerden rakamları çıkarıp toplama/çarpma vb. örneği
        # demo amaçlı sadece sayıları toplayalım
        import re

        nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)]
        if len(nums) >= 2 and ("topla" in text.lower() or "+" in text):
            answer = f"Toplam: {sum(nums)}"
        elif len(nums) >= 2 and ("carp" in text.lower() or "*" in text):
            prod = 1.0
            for n in nums:
                prod *= n
            answer = f"Çarpım: {prod}"
        elif len(nums) == 2 and ("bol" in text.lower() or "/" in text):
            if nums[1] == 0:
                answer = "Bölme: tanımsız (0'a bölme)"
            else:
                answer = f"Bölme: {nums[0] / nums[1]}"
        elif len(nums) == 2 and ("cikar" in text.lower() or "-" in text):
            answer = f"Fark: {nums[0] - nums[1]}"
        elif len(nums) > 0:
            answer = f"Sayılar: {nums}"
        else:
            answer = "Lütfen basit bir işlem belirtin (topla, cikar, carp, bol)."
    except Exception as e:
        answer = f"Hata: {e}"

    result = {
        "contextId": message.get("contextId"),
        "message": {
            "role": "assistant",
            "parts": [{"kind": "text", "text": answer}],
        },
    }
    return JSONResponse(jsonrpc_success(id_val, result))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)


