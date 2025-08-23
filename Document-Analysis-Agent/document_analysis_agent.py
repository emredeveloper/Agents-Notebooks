import os
import re
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

PORT = int(os.getenv("DOCUMENT_AGENT_PORT", 8005))
BASE_URL = os.getenv("DOCUMENT_AGENT_BASE_URL", f"http://localhost:{PORT}")
MODEL = os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-4b-2507")

app = FastAPI(title="DocumentAnalysisAgent")


@app.get(AGENT_CARD_WELL_KNOWN_PATH)
async def agent_card():
    card = build_public_agent_card(
        name="DocumentAnalysisAgent",
        description="Metin ve döküman analizi, özetleme ve içerik çıkarımı yapan ajan (LM Studio)",
        base_url=BASE_URL,
        streaming=False,
    )
    return JSONResponse(card)


def analyze_request_type(text: str) -> tuple[str, str]:
    """
    Kullanıcının ne tür bir analiz istediğini belirler.
    Returns: (analysis_type, clean_text)
    """
    text_lower = text.lower().strip()
    
    # Özetleme isteği
    if any(phrase in text_lower for phrase in [
        "özetle", "özetini çıkar", "özet yap", "summarize", 
        "kısaca anlat", "özetleme", "summary"
    ]):
        return "summarize", text
    
    # Ana fikirler çıkarma
    elif any(phrase in text_lower for phrase in [
        "ana fikir", "ana nokta", "key point", "main idea",
        "önemli nokta", "temel mesaj"
    ]):
        return "key_points", text
    
    # Duygu analizi
    elif any(phrase in text_lower for phrase in [
        "duygu analiz", "sentiment", "ton analiz", "duygusal",
        "pozitif mi", "negatif mi"
    ]):
        return "sentiment", text
    
    # Kategori belirleme
    elif any(phrase in text_lower for phrase in [
        "kategori", "category", "tür", "type", "sınıflandır",
        "hangi alan", "hangi konu"
    ]):
        return "categorize", text
    
    # Varsayılan: genel analiz
    else:
        return "general", text


def extract_content_from_text(text: str) -> str:
    """
    Metinden asıl içeriği çıkarır (URL'ler, dosya yolları vb. temizler).
    """
    # URL'leri kaldır
    text = re.sub(r'https?://\S+', '[URL]', text)
    
    # Email adreslerini kaldır
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Dosya yollarını kaldır
    text = re.sub(r'[A-Za-z]:\\[^\s]+', '[DOSYA_YOLU]', text)
    
    # Çoklu boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


async def perform_analysis(analysis_type: str, content: str) -> str:
    """
    Belirtilen analiz türünü gerçekleştirir.
    """
    content = extract_content_from_text(content)
    
    if analysis_type == "summarize":
        prompt = f"""Sen profesyonel bir metin analizcisisin. Aşağıdaki metni Türkçe olarak özetle.

Özetleme kuralları:
- Ana fikirları koru
- Önemli detayları dahil et
- Kısa ve öz yaz (maksimum 3-4 cümle)
- Orijinal anlamı değiştirme

Metin:
{content}

Özet:"""

    elif analysis_type == "key_points":
        prompt = f"""Sen profesyonel bir metin analizcisisin. Aşağıdaki metnin ana fikirlerini çıkar.

Ana fikir çıkarma kuralları:
- En önemli 3-5 noktayı belirle
- Her noktayı madde işareti ile yaz
- Kısa ve net ifadeler kullan
- Türkçe yanıt ver

Metin:
{content}

Ana Fikirler:"""

    elif analysis_type == "sentiment":
        prompt = f"""Sen profesyonel bir duygu analizcisisin. Aşağıdaki metnin duygusal tonunu analiz et.

Analiz kriterleri:
- Pozitif, negatif veya nötr olarak sınıflandır
- Duygu yoğunluğunu belirt (zayıf, orta, güçlü)
- Gerekçesini açıkla
- Türkçe yanıt ver

Metin:
{content}

Duygu Analizi:"""

    elif analysis_type == "categorize":
        prompt = f"""Sen profesyonel bir metin sınıflandırıcısısın. Aşağıdaki metni kategorize et.

Sınıflandırma kriterleri:
- Metnin ana konusunu belirle
- Uygun kategori öner (teknoloji, sağlık, eğitim, vs.)
- Gerekçesini açıkla
- Türkçe yanıt ver

Metin:
{content}

Kategori Analizi:"""

    else:  # general
        prompt = f"""Sen profesyonel bir metin analizcisisin. Aşağıdaki metni genel olarak analiz et.

Analiz kapsamı:
- Metnin ana konusu
- Önemli noktalar
- Ton ve stil
- Kısa bir değerlendirme
- Türkçe yanıt ver

Metin:
{content}

Genel Analiz:"""

    return await call_lmstudio(
        prompt=prompt,
        model=MODEL,
        temperature=0.3,
    )


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
        if len(prompt.strip()) < 10:
            analysis_result = "📄 Lütfen analiz edilecek yeterli uzunlukta bir metin girin (en az 10 karakter)."
        else:
            # İstek türünü belirle
            analysis_type, clean_text = analyze_request_type(prompt)
            
            # Analizi gerçekleştir
            lm_response = await perform_analysis(analysis_type, clean_text)
            
            # Analiz türüne göre emoji ve başlık ekle
            type_icons = {
                "summarize": "📋 Özet",
                "key_points": "🎯 Ana Fikirler", 
                "sentiment": "💭 Duygu Analizi",
                "categorize": "📂 Kategori Analizi",
                "general": "📄 Genel Analiz"
            }
            
            icon = type_icons.get(analysis_type, "📄 Analiz")
            analysis_result = f"{icon}\n\n{lm_response.strip()}"
            
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
                                "text": f"DocumentAnalysisAgent LLM hatası: {str(e)}. Lütfen LM Studio sunucusunu ve model adını kontrol edin.",
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
            "parts": [{"kind": "text", "text": analysis_result}],
        },
    }
    return JSONResponse(jsonrpc_success(id_val, result))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)