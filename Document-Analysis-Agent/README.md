# 📄 Document Analysis Agent

Bu agent, LM Studio destekli metin ve döküman analizi hizmetleri sağlar. Özetleme, ana fikir çıkarma, duygu analizi ve kategorizasyon gibi işlemler yapar.

## 🎯 Özellikler

- **📋 Akıllı Özetleme**: Uzun metinleri kısa ve öz özetler
- **🎯 Ana Fikir Çıkarımı**: Metnin temel noktalarını madde halinde sunar
- **💭 Duygu Analizi**: Pozitif/negatif/nötr ton tespiti
- **📂 Kategori Belirleme**: Metnin konusunu ve kategorisini belirler
- **🔍 Genel Analiz**: Kapsamlı metin değerlendirmesi
- **🧠 LM Studio Entegrasyonu**: Yerel LLM modelleri kullanır
- **⚡ A2A Protokol Uyumlu**: JSON-RPC ve agent card desteği

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
cd Document-Analysis-Agent
python document_analysis_agent.py
```

Default port: `8005`

### Ortam Değişkenleri
```bash
export DOCUMENT_AGENT_PORT=8005
export LMSTUDIO_BASE_URL=http://localhost:1234
export LMSTUDIO_MODEL=qwen/qwen3-4b-2507
```

### Test Etme
```bash
# Agent card bilgisini görüntüle
curl http://localhost:8005/.well-known/agent-card.json | jq

# Analiz testi
curl -X POST http://localhost:8005/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "agent.sendMessage",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Bu uzun metnin ana fikirlerini çıkar: [UZUN_METIN]"}]
      }
    }
  }' | jq
```

## 🔍 Analiz Türleri

### 📋 Özetleme
Metni kısa ve öz şekilde özetler.

**Komutlar:**
- "özetle [metin]"
- "özetini çıkar [metin]"
- "özet yap [metin]"
- "kısaca anlat [metin]"

**Örnek:**
```
✅ "Bu makaleyi özetle: [UZUN_MAKALE_METNİ]"
```

### 🎯 Ana Fikirler
Metnin en önemli noktalarını madde halinde çıkarır.

**Komutlar:**
- "ana fikir [metin]"
- "ana nokta [metin]"
- "önemli nokta [metin]"
- "temel mesaj [metin]"

**Örnek:**
```
✅ "Bu sunumun ana fikirlerini çıkar: [SUNUM_METNİ]"
```

### 💭 Duygu Analizi
Metnin duygusal tonunu ve yoğunluğunu analiz eder.

**Komutlar:**
- "duygu analiz [metin]"
- "ton analiz [metin]"
- "pozitif mi negatif mi [metin]"

**Örnek:**
```
✅ "Bu müşteri yorumunun duygu analizini yap: [YORUM_METNİ]"
```

### 📂 Kategorizasyon
Metnin konusunu ve kategorisini belirler.

**Komutlar:**
- "kategori [metin]"
- "hangi konu [metin]"
- "sınıflandır [metin]"
- "hangi alan [metin]"

**Örnek:**
```
✅ "Bu haber metnini kategorize et: [HABER_METNİ]"
```

### 📄 Genel Analiz
Kapsamlı metin değerlendirmesi yapar.

**Varsayılan davranış** - özel komut belirtilmezse genel analiz yapar.

## 🔧 Orchestrator Entegrasyonu

Bu agent'ı orchestrator.py'ye eklemek için:

```python
# orchestrator.py içinde
DOCUMENT_URL = os.getenv("DOCUMENT_AGENT_URL", "http://localhost:8005")

# Döküman analizi isteği algılama
wants_document_analysis = any(s in user_text.lower() for s in [
    "özetle", "analiz", "duygu", "kategori", "ana fikir", "önemli nokta"
])

if wants_document_analysis:
    document_payload = {
        "jsonrpc": "2.0",
        "id": "document-1",
        "method": "agent.sendMessage",
        "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": user_text}]}}
    }
    document_resp = await send_message(DOCUMENT_URL, document_payload)
    # ... handle response
```

## 🎨 UI Streamlit Entegrasyonu

UI'ye eklemek için agent listesine ekleyin:

```python
# ui_streamlit.py içinde
agents = {
    "Document": {"url": "http://localhost:8005", "port": 8005},
    # ... diğer agents
}
```

## 📊 Analiz Örnekleri

### Özetleme Örneği
**Giriş:**
```
"Bu uzun makaleyi özetle: Yapay zeka teknolojileri son yıllarda... [500 kelimelik metin]"
```

**Çıkış:**
```
📋 Özet

Yapay zeka teknolojileri son yıllarda hızla gelişerek birçok sektörde devrim yarattı. 
Özellikle makine öğrenmesi ve dil modelleri, otomasyondan sağlığa kadar geniş bir 
yelpazede kullanılıyor. Gelecekte bu teknolojilerin daha da yaygınlaşması bekleniyor.
```

### Ana Fikirler Örneği
**Giriş:**
```
"Bu projenin ana fikirlerini çıkar: E-ticaret platformu geliştirme projesi..."
```

**Çıkış:**
```
🎯 Ana Fikirler

• Kullanıcı dostu arayüz tasarımı
• Güvenli ödeme sistemi entegrasyonu  
• Mobil uyumluluk
• Hızlı kargo ve teslimat seçenekleri
• Müşteri destek sistemi
```

## 🔍 Hata Ayıklama

### Yaygın Sorunlar

**LM Studio bağlantı hatası:**
- LM Studio'nun çalıştığından emin olun
- Model yüklenmiş olduğunu kontrol edin
- Base URL'in doğru olduğunu kontrol edin

**Kısa metin hatası:**
- En az 10 karakter uzunluğunda metin gönderin
- Anlamlı içerik sağlayın

**Port çakışması:**
- `DOCUMENT_AGENT_PORT` değişkenini kullanın
- `lsof -i :8005` ile port kullanımını kontrol edin

## 📈 Performans Optimizasyonu

- **Uzun metinler**: Çok uzun metinler için chunking uygulayın
- **Model seçimi**: Analiz türüne göre uygun model kullanın
- **Temperature ayarı**: Analitik işlemler için düşük temperature (0.1-0.3)

## 🌟 Gelecek Özellikler

- [ ] PDF dosya desteği
- [ ] Toplu analiz (batch processing)
- [ ] Özel kategori tanımlama
- [ ] Analiz geçmişi
- [ ] Karşılaştırmalı analiz
- [ ] Çoklu dil desteği
- [ ] Excel/CSV çıktı
- [ ] Görselleştirme desteği

## 🤝 Katkıda Bulunma

Bu agent'ı geliştirmek için:
1. Yeni analiz türleri ekleyin
2. Dosya format desteği genişletin
3. UI/UX iyileştirmeleri yapın
4. Performans optimizasyonları yapın
5. Test senaryoları ekleyin

---

**Not**: Bu agent eğitim ve araştırma amaçlıdır. Hassas verilerin analizinde güvenlik önlemlerini dikkate alın.