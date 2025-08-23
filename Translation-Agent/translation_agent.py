import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import sys

# Import common utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'A2A-Agent'))
from common import (
    AGENT_CARD_WELL_KNOWN_PATH,
    build_public_agent_card,
    jsonrpc_error,
    jsonrpc_success,
    call_lmstudio,
)

PORT = int(os.getenv("TRANSLATION_AGENT_PORT", 8004))
BASE_URL = os.getenv("TRANSLATION_AGENT_BASE_URL", f"http://localhost:{PORT}")
MODEL = os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-4b-2507")

app = FastAPI(title="TranslationAgent")


@app.get(AGENT_CARD_WELL_KNOWN_PATH)
async def agent_card():
    card = build_public_agent_card(
        name="TranslationAgent",
        description="Türkçe-İngilizce ve diğer diller arası çeviri yapan ajan (LM Studio)",
        base_url=BASE_URL,
        streaming=False,
    )
    return JSONResponse(card)


def detect_language_intent(text: str) -> tuple[str, str, str]:
    """
    Metinden çeviri niyetini tespit eder.
    Returns: (source_lang, target_lang, clean_text)
    """
    text_lower = text.lower().strip()
    
    # Türkçe → İngilizce
    if any(phrase in text_lower for phrase in [
        "ingilizceye çevir", "inglizce çevir", "translate to english", 
        "ing çevir", "ingilizce yap", "inglizceye"
    ]):
        # Çeviri komutunu temizle
        for phrase in ["ingilizceye çevir", "inglizce çevir", "ing çevir", "ingilizce yap", "inglizceye"]:
            text = text.replace(phrase, "").strip()
        return "Turkish", "English", text
    
    # İngilizce → Türkçe  
    elif any(phrase in text_lower for phrase in [
        "türkçeye çevir", "turkceye cevir", "translate to turkish",
        "tr çevir", "türkçe yap", "turkce yap"
    ]):
        for phrase in ["türkçeye çevir", "turkceye cevir", "tr çevir", "türkçe yap", "turkce yap"]:
            text = text.replace(phrase, "").strip()
        return "English", "Turkish", text
    
    # Otomatik dil tespiti - Türkçe karakterler varsa Türkçe → İngilizce
    elif any(char in text for char in "çğıöşüÇĞIÖŞÜ"):
        return "Turkish", "English", text
    
    # Diğer durumlarda İngilizce → Türkçe
    else:
        return "English", "Turkish", text


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
        # Dil tespiti ve çeviri niyetini belirle
        source_lang, target_lang, clean_text = detect_language_intent(prompt.strip())
        
        if not clean_text:
            translation_result = "Lütfen çevrilecek bir metin girin."
        else:
            # LM Studio ile çeviri yap
            translation_prompt = f"""Sen profesyonel bir çevirmensin. Aşağıdaki metni {source_lang} dilinden {target_lang} diline çevir.

Çeviri kuralları:
- Doğal ve akıcı çeviri yap
- Bağlamı ve anlamı koru
- Kültürel nüansları dikkate al
- Sadece çeviriyi döndür, açıklama yapma

Çevrilecek metin:
{clean_text}

Çeviri:"""

            lm_response = await call_lmstudio(
                prompt=translation_prompt,
                model=MODEL,
                temperature=0.1,  # Çeviri için düşük temperature
            )
            
            translation_result = f"🌐 {source_lang} → {target_lang}\n\n{lm_response.strip()}"
            
    except Exception as e:
        # JSON-RPC hata cevabı
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
                                "text": f"TranslationAgent LLM hatası: {str(e)}. Lütfen LM Studio sunucusunu ve model adını kontrol edin.",
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
            "parts": [{"kind": "text", "text": translation_result}],
        },
    }
    return JSONResponse(jsonrpc_success(id_val, result))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)