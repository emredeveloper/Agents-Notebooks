# 🌤️ Weather Agent

Bu agent, OpenWeatherMap API ve LM Studio entegrasyonu ile hava durumu bilgisi ve tahminleri sağlar. Doğal dil işleme ile kullanıcı dostu yanıtlar üretir.

## 🎯 Özellikler

- **🌍 Geniş Şehir Desteği**: Türkiye ve dünya şehirleri
- **🤖 Doğal Dil Yanıtları**: LM Studio ile arkadaşça tonlama
- **👕 Giyinme Önerileri**: Hava durumuna göre tavsiyeler
- **📊 Detaylı Bilgi**: Sıcaklık, nem, rüzgar, basınç
- **⚡ A2A Protokol Uyumlu**: JSON-RPC ve agent card desteği
- **🔄 Akıllı Şehir Tespiti**: Metinden otomatik şehir çıkarımı
- **🛡️ Hata Toleransı**: API olmasa da çalışır (mock data)

## 🚀 Kurulum

### Gereksinimler
```bash
pip install fastapi uvicorn httpx
```

### OpenWeatherMap API Key (Opsiyonel)
1. [OpenWeatherMap](https://openweathermap.org/api) hesabı oluşturun
2. Ücretsiz API key alın
3. Environment variable olarak ayarlayın

### LM Studio Kurulumu
1. LM Studio'yu indirin ve başlatın
2. Bir dil modeli yükleyin (örn. Qwen, Gemma)
3. "OpenAI compatible server" özelliğini aktif edin

## 📋 Kullanım

### Agent'ı Başlatma
```bash
cd Weather-Agent
python weather_agent.py
```

Default port: `8006`

### Ortam Değişkenleri
```bash
export WEATHER_AGENT_PORT=8006
export OPENWEATHER_API_KEY=your_api_key_here
export LMSTUDIO_BASE_URL=http://localhost:1234
export LMSTUDIO_MODEL=qwen/qwen3-4b-2507
```

### Test Etme
```bash
# Agent card bilgisini görüntüle
curl http://localhost:8006/.well-known/agent-card.json | jq

# Hava durumu testi
curl -X POST http://localhost:8006/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "agent.sendMessage",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "İstanbul hava durumu"}]
      }
    }
  }' | jq
```

## 🌤️ Hava Durumu Komutları

### 🏠 Türkiye Şehirleri
```
✅ "İstanbul hava durumu"
✅ "Ankara'da hava nasıl?"
✅ "İzmir sıcaklık"
✅ "Antalya hava"
✅ "Bursa weather"
```

### 🌍 Dünya Şehirleri
```
✅ "London hava durumu"
✅ "Paris weather"
✅ "New York sıcaklık"
✅ "Tokyo hava"
✅ "Berlin weather forecast"
```

### 🔍 Akıllı Tespit
Agent şu şekillerde şehir tespiti yapar:
- **Doğrudan isim**: "İstanbul"
- **Soru formatı**: "Ankara'da hava nasıl?"
- **Karışık metin**: "Bugün İzmir'e gidiyorum, hava durumu?"
- **İngilizce**: "London weather please"

## 📊 Yanıt Formatı

### Örnek Çıktı
```
🌤️ İstanbul Hava Durumu

🌤️ İstanbul'da bugün güzel bir gün sizi bekliyor! 22°C ile oldukça 
rahat bir sıcaklık var. Hafif rüzgar ve %60 nem ile ferah hissedeceksiniz. 
Hafif bir ceket alın, akşama doğru serinleyebilir! ☀️

⚠️ Bu örnek veridir. Gerçek hava durumu için OpenWeatherMap API key'i ayarlayın.
```

### Detaylı Bilgiler
- **🌡️ Sıcaklık**: Güncel ve hissedilen
- **💧 Nem**: Yüzde olarak
- **🌪️ Rüzgar**: m/s cinsinden hız
- **📊 Basınç**: hPa cinsinden
- **☁️ Durum**: Açıklayıcı metin
- **👕 Öneriler**: Giyinme tavsiyeleri

## 🔧 Orchestrator Entegrasyonu

Bu agent'ı orchestrator.py'ye eklemek için:

```python
# orchestrator.py içinde
WEATHER_URL = os.getenv("WEATHER_AGENT_URL", "http://localhost:8006")

# Hava durumu isteği algılama
wants_weather = any(s in user_text.lower() for s in [
    "hava", "weather", "sıcaklık", "temperature", "tahmin", "forecast"
])

if wants_weather:
    weather_payload = {
        "jsonrpc": "2.0",
        "id": "weather-1",
        "method": "agent.sendMessage",
        "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": user_text}]}}
    }
    weather_resp = await send_message(WEATHER_URL, weather_payload)
    # ... handle response
```

## 🎨 UI Streamlit Entegrasyonu

UI'ye eklemek için agent listesine ekleyin:

```python
# ui_streamlit.py içinde
agents = {
    "Weather": {"url": "http://localhost:8006", "port": 8006},
    # ... diğer agents
}
```

## 🌆 Desteklenen Şehirler

### 🇹🇷 Türkiye
- İstanbul, Ankara, İzmir, Bursa, Antalya
- Adana, Konya, Şanlıurfa, Gaziantep, Kayseri
- Mersin, Eskişehir, Diyarbakır, Samsun, Denizli
- Ve diğer büyük şehirler...

### 🌍 Dünya
- **Avrupa**: London, Paris, Berlin, Rome, Madrid
- **Amerika**: New York, Los Angeles, Chicago, Miami
- **Asya**: Tokyo, Beijing, Seoul, Mumbai, Bangkok
- **Okyanusya**: Sydney, Melbourne

## 🔐 API Key Yapılandırması

### OpenWeatherMap Ücretsiz Plan
- **Günlük**: 1,000 istek
- **Dakikalık**: 60 istek
- **Özellikler**: Güncel hava, 5 günlük tahmin

### API Key Olmadan
Agent API key olmadan da çalışır:
- Mock data kullanır
- Geliştirme ve test için idealdir
- Gerçek veriler için API key gereklidir

## 🔍 Hata Ayıklama

### Yaygın Sorunlar

**API bağlantı hatası:**
- İnternet bağlantınızı kontrol edin
- API key'in doğru olduğunu kontrol edin
- Rate limit aşılmadığından emin olun

**Şehir bulunamadı:**
- Şehir adını İngilizce deneyin
- Büyük/küçük harf farklılığı olmamalı
- Desteklenen şehirler listesini kontrol edin

**LM Studio formatı hatası:**
- LM Studio'nun çalıştığından emin olun
- Model yüklenmiş olduğunu kontrol edin
- Ham veri gösterilir, işlevsellik devam eder

## 📈 Performans

- **Yanıt Süresi**: ~1-3 saniye
- **Doğruluk**: OpenWeatherMap kalitesi
- **Dil Desteği**: Türkçe ve İngilizce
- **Güvenilirlik**: Hata toleranslı tasarım

## 🌟 Gelecek Özellikler

- [ ] 5 günlük hava tahmini
- [ ] Saatlik hava durumu
- [ ] Hava durumu uyarıları
- [ ] UV indeksi bilgisi
- [ ] Hava kalitesi verileri
- [ ] Görsel hava durumu haritaları
- [ ] Favori şehirler listesi
- [ ] Push notification desteği
- [ ] Tarihsel hava verileri

## 🤝 Katkıda Bulunma

Bu agent'ı geliştirmek için:
1. Yeni şehirler ekleyin
2. Hava durumu format iyileştirmeleri yapın
3. UI/UX geliştirmeleri ekleyin  
4. Yeni özellikler (tahmin, uyarılar) ekleyin
5. Test senaryoları geliştirin

---

**Not**: OpenWeatherMap API kullanımı için [Terms of Service](https://openweathermap.org/terms) geçerlidir. Bu agent eğitim ve araştırma amaçlıdır.