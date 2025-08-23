# 🌐 Translation Agent

Bu agent, LM Studio destekli çeviri hizmetleri sağlar. Türkçe-İngilizce arası çeviri yapar ve otomatik dil algılama özelliği vardır.

## 🎯 Özellikler

- **🔄 Çift Yönlü Çeviri**: Türkçe ↔ İngilizce
- **🤖 Otomatik Dil Algılama**: Metindeki Türkçe karakterleri algılar
- **🧠 LM Studio Entegrasyonu**: Yerel LLM modelleri kullanır
- **⚡ A2A Protokol Uyumlu**: JSON-RPC ve agent card desteği
- **🎨 Akıllı Komut Analizi**: "İngilizceye çevir" gibi komutları anlar

## 🚀 Kurulum

### Gereksinimler
```bash
pip install fastapi uvicorn httpx
```

### LM Studio Kurulumu
1. LM Studio'yu indirin ve başlatın
2. Bir dil modeli yükleyin (örn. Qwen, Gemma)
3. "OpenAI compatible server" özelliğini aktif edin

## 📋 Kullanım

### Agent'ı Başlatma
```bash
cd Translation-Agent
python translation_agent.py
```

Default port: `8004`

### Ortam Değişkenleri
```bash
export TRANSLATION_AGENT_PORT=8004
export LMSTUDIO_BASE_URL=http://localhost:1234
export LMSTUDIO_MODEL=qwen/qwen3-4b-2507
```

### Test Etme
```bash
# Agent card bilgisini görüntüle
curl http://localhost:8004/.well-known/agent-card.json | jq

# Çeviri testi
curl -X POST http://localhost:8004/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "agent.sendMessage",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Merhaba dünya ingilizceye çevir"}]
      }
    }
  }' | jq
```

## 🔄 Çeviri Komutları

### Türkçe → İngilizce
- "ingilizceye çevir [metin]"
- "inglizce çevir [metin]" 
- "ing çevir [metin]"
- Veya sadece Türkçe karakterli metin gönderin

### İngilizce → Türkçe
- "türkçeye çevir [metin]"
- "tr çevir [metin]"
- "türkçe yap [metin]"

### Örnekler
```
✅ "Merhaba dünya ingilizceye çevir"
✅ "Hello world türkçeye çevir"
✅ "Bu güzel bir gün" (otomatik İngilizceye)
✅ "This is a beautiful day" (otomatik Türkçeye)
```

## 🔧 Orchestrator Entegrasyonu

Bu agent'ı orchestrator.py'ye eklemek için:

```python
# orchestrator.py içinde
TRANSLATION_URL = os.getenv("TRANSLATION_AGENT_URL", "http://localhost:8004")

# Çeviri isteği algılama
wants_translation = any(s in user_text.lower() for s in [
    "çevir", "translate", "translation", "inglizce", "ingilizce", "türkçe", "turkce"
])

if wants_translation:
    translation_payload = {
        "jsonrpc": "2.0",
        "id": "translation-1",
        "method": "agent.sendMessage",
        "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": user_text}]}}
    }
    translation_resp = await send_message(TRANSLATION_URL, translation_payload)
    # ... handle response
```

## 🎨 UI Streamlit Entegrasyonu

UI'ye eklemek için agent listesine ekleyin:

```python
# ui_streamlit.py içinde
agents = {
    "Translation": {"url": "http://localhost:8004", "port": 8004},
    # ... diğer agents
}
```

## 🔍 Hata Ayıklama

### Yaygın Sorunlar

**LM Studio bağlantı hatası:**
- LM Studio'nun çalıştığından emin olun
- Model yüklenmiş olduğunu kontrol edin
- Base URL'in doğru olduğunu kontrol edin

**Port çakışması:**
- `TRANSLATION_AGENT_PORT` değişkenini kullanın
- `lsof -i :8004` ile port kullanımını kontrol edin

## 📊 Performans

- **Hız**: ~1-3 saniye (model boyutuna bağlı)
- **Kalite**: LM Studio modelinin kalitesine bağlı
- **Dil Desteği**: Şu anda TR-EN, gelecekte genişletilebilir

## 🌟 Gelecek Özellikler

- [ ] Çoklu dil desteği (Almanca, Fransızca, vb.)
- [ ] Döküman çevirisi (PDF, DOCX)
- [ ] Çeviri geçmişi
- [ ] Çeviri kalitesi skorlama
- [ ] Toplu çeviri
- [ ] Dil algılama güvenilirlik skoru

## 🤝 Katkıda Bulunma

Bu agent'ı geliştirmek için:
1. Yeni dil desteği ekleyin
2. Çeviri kalitesini artırın  
3. UI/UX iyileştirmeleri yapın
4. Test senaryoları ekleyin

---

**Not**: Bu agent eğitim ve araştırma amaçlıdır. Ticari kullanım için model lisanslarını kontrol edin.