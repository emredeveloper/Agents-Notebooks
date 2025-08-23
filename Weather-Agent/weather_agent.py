import os
import json
import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
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

PORT = int(os.getenv("WEATHER_AGENT_PORT", 8006))
BASE_URL = os.getenv("WEATHER_AGENT_BASE_URL", f"http://localhost:{PORT}")
MODEL = os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-4b-2507")

# OpenWeatherMap API (ücretsiz, kayıt gerekiyor)
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
WEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5"

app = FastAPI(title="WeatherAgent")


@app.get(AGENT_CARD_WELL_KNOWN_PATH)
async def agent_card():
    card = build_public_agent_card(
        name="WeatherAgent",
        description="Hava durumu bilgisi ve tahminleri sağlayan ajan (OpenWeatherMap + LM Studio)",
        base_url=BASE_URL,
        streaming=False,
    )
    return JSONResponse(card)


def extract_city_from_text(text: str) -> str:
    """
    Metinden şehir ismini çıkarır.
    """
    text_lower = text.lower().strip()
    
    # Türkiye'nin büyük şehirleri
    turkish_cities = [
        "istanbul", "ankara", "izmir", "bursa", "antalya", "adana", "konya", 
        "şanlıurfa", "gaziantep", "kayseri", "mersin", "eskişehir", "diyarbakır",
        "samsun", "denizli", "malatya", "kahramanmaraş", "erzurum", "van",
        "batman", "elazığ", "erzincan", "tokat", "ordu", "karabük", "amasya",
        "trabzon", "manisa", "afyon", "uşak", "balıkesir", "tekirdağ", 
        "sakarya", "kocaeli", "rize", "giresun", "sinop", "çorum", "yozgat"
    ]
    
    # Dünya şehirleri
    world_cities = [
        "london", "paris", "berlin", "rome", "madrid", "vienna", "amsterdam",
        "brussels", "zurich", "stockholm", "oslo", "helsinki", "copenhagen",
        "new york", "los angeles", "chicago", "miami", "boston", "washington",
        "tokyo", "beijing", "shanghai", "seoul", "mumbai", "delhi", "bangkok",
        "singapore", "sydney", "melbourne", "toronto", "vancouver", "moscow"
    ]
    
    all_cities = turkish_cities + world_cities
    
    # Hava durumu komutlarını temizle
    clean_text = text_lower
    for phrase in ["hava durumu", "weather", "hava", "sıcaklık", "weather forecast", "tahmin"]:
        clean_text = clean_text.replace(phrase, "").strip()
    
    # Şehir ismi ara
    for city in all_cities:
        if city in clean_text:
            return city.title()
    
    # Kelime kelime kontrol et (şehir ismi bulma)
    words = clean_text.split()
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if len(clean_word) > 2:  # En az 3 karakter
            return clean_word.title()
    
    # Varsayılan şehir
    return "Istanbul"


async def get_weather_data(city: str) -> dict:
    """
    OpenWeatherMap API'den hava durumu verisini alır.
    """
    if not WEATHER_API_KEY:
        # API key yoksa mock data döndür
        return {
            "weather": [{"main": "Clear", "description": "açık hava"}],
            "main": {"temp": 22, "feels_like": 24, "humidity": 60, "pressure": 1013},
            "wind": {"speed": 3.5, "deg": 120},
            "visibility": 10000,
            "name": city,
            "mock": True
        }
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Güncel hava durumu
            url = f"{WEATHER_BASE_URL}/weather"
            params = {
                "q": city,
                "appid": WEATHER_API_KEY,
                "units": "metric",  # Celsius
                "lang": "tr"
            }
            
            response = await client.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                # Hata durumunda mock data
                return {
                    "weather": [{"main": "Unknown", "description": "bilinmeyen"}],
                    "main": {"temp": 20, "feels_like": 20, "humidity": 50, "pressure": 1013},
                    "wind": {"speed": 2.0, "deg": 0},
                    "visibility": 10000,
                    "name": city,
                    "error": f"API Error: {response.status_code}"
                }
    except Exception as e:
        # Bağlantı hatası durumunda mock data
        return {
            "weather": [{"main": "Unknown", "description": "bilinmeyen"}],
            "main": {"temp": 20, "feels_like": 20, "humidity": 50, "pressure": 1013},
            "wind": {"speed": 2.0, "deg": 0},
            "visibility": 10000,
            "name": city,
            "error": f"Connection Error: {str(e)}"
        }


async def format_weather_response(weather_data: dict, city: str) -> str:
    """
    Hava durumu verisini LM Studio ile doğal dilde formatlama.
    """
    # Temel bilgileri çıkar
    temp = weather_data["main"]["temp"]
    feels_like = weather_data["main"]["feels_like"]
    humidity = weather_data["main"]["humidity"]
    pressure = weather_data["main"]["pressure"]
    description = weather_data["weather"][0]["description"]
    wind_speed = weather_data["wind"]["speed"]
    
    # Mock data uyarısı
    is_mock = weather_data.get("mock", False)
    has_error = weather_data.get("error", None)
    
    # LM Studio ile doğal yanıt oluştur
    weather_prompt = f"""Sen hava durumu uzmanısın. Aşağıdaki hava durumu verilerini Türkçe olarak doğal ve anlaşılır şekilde kullanıcıya sun.

Şehir: {city}
Sıcaklık: {temp}°C
Hissedilen: {feels_like}°C  
Nem: {humidity}%
Basınç: {pressure} hPa
Durum: {description}
Rüzgar: {wind_speed} m/s

Kurallar:
- Samimi ve arkadaşça ton kullan
- Giyinme önerisi ver
- Günün nasıl geçeceği hakkında yorum yap
- Emojiler kullan
- 2-3 cümle ile özetle

Hava Durumu Raporu:"""

    try:
        lm_response = await call_lmstudio(
            prompt=weather_prompt,
            model=MODEL,
            temperature=0.7,  # Yaratıcılık için biraz yüksek
        )
        
        result = f"🌤️ {city} Hava Durumu\n\n{lm_response.strip()}"
        
        # Uyarılar ekle
        if is_mock:
            result += "\n\n⚠️ Bu örnek veridir. Gerçek hava durumu için OpenWeatherMap API key'i ayarlayın."
        elif has_error:
            result += f"\n\n⚠️ API Hatası: {has_error}"
            
        return result
        
    except Exception as e:
        # LM Studio hatası durumunda basit format
        return f"""🌤️ {city} Hava Durumu

🌡️ Sıcaklık: {temp}°C (Hissedilen: {feels_like}°C)
💧 Nem: {humidity}%
🌪️ Rüzgar: {wind_speed} m/s
📊 Basınç: {pressure} hPa
☁️ Durum: {description}

⚠️ LM Studio formatı başarısız: {str(e)}"""


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
        # Şehir ismini çıkar
        city = extract_city_from_text(prompt.strip())
        
        # Hava durumu verisini al
        weather_data = await get_weather_data(city)
        
        # Doğal dilde formatla
        weather_result = await format_weather_response(weather_data, city)
            
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
                                "text": f"WeatherAgent hatası: {str(e)}. Lütfen şehir adını kontrol edin veya sistem yöneticisine başvurun.",
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
            "parts": [{"kind": "text", "text": weather_result}],
        },
    }
    return JSONResponse(jsonrpc_success(id_val, result))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)