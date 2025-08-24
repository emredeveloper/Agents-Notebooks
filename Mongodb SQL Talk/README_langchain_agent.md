# MongoDB LangChain Agent with LM Studio

Bu proje, LangChain framework'ü kullanarak MongoDB veritabanları ile doğal dil etkileşimi sağlayan bir agent'tır. LM Studio ile yerel AI modelleri kullanır ve Flask web arayüzü ile birlikte gelir.

## Özellikler

- 🤖 **LangChain Agent**: ReAct pattern kullanarak akıllı MongoDB sorguları
- 🍃 **MongoDB Entegrasyonu**: Pymongo ile veritabanı bağlantısı
- 🎯 **LM Studio Desteği**: Yerel LLM modelleri (Llama, Gemma, Mistral, vb.)
- 🌐 **Flask Web Arayüzü**: Kullanıcı dostu web interface
- 💾 **Konversasyon Hafızası**: Önceki konuşmaları hatırlama
- 🔧 **Schema Analizi**: Otomatik koleksiyon şeması keşfi

## Kurulum

1. **Gerekli paketleri yükleyin:**

```bash
pip install -r requirements.txt
```

2. **MongoDB'yi başlatın:**

```bash
# Docker ile:
docker run -d -p 27017:27017 mongo

# Veya yerel MongoDB kurulumu
mongod --dbpath /data/db
```

3. **LM Studio'yu kurun ve bir model yükleyin:**

```
1. LM Studio'yu indirin: https://lmstudio.ai
2. Bir model indirin (örn: Llama 3.2, Gemma 2, Mistral)
3. Local Server sekmesine gidin
4. Server'ı başlatın (port 1234)
5. Model adını not edin
```

## Kullanım

### Web Arayüzü ile

```bash
python mongodb-langchain-agent.py
```

Tarayıcınızda `http://localhost:5000` adresine gidin.

### Programatik Kullanım

```python
from mongodb_langchain_agent import MongoDBLangChainAgent

# Agent'ı başlatın
agent = MongoDBLangChainAgent(
    mongo_uri="mongodb://localhost:27017/",
    lm_studio_url="http://localhost:1234/v1",
    model_name="llama-3.2-3b-instruct"
)

# Sorgu gönderin
response = agent.process_query("Show me all users", db_name="myapp")
print(response)
```

## Örnek Sorgular

- "Koleksiyonları listele"
- "İlk 10 kullanıcıyı göster"
- "25 yaşından büyük kullanıcıları bul"
- "En çok sipariş veren müşteriyi göster"
- "Son hafta eklenen ürünleri listele"

## Yapılandırma

### MongoDB Bağlantısı

```python
# Varsayılan
mongo_uri = "mongodb://localhost:27017/"

# Kimlik doğrulama ile
mongo_uri = "mongodb://username:password@localhost:27017/"

# MongoDB Atlas
mongo_uri = "mongodb+srv://username:password@cluster.mongodb.net/"
```

### LM Studio Modelleri

- `llama-3.2-3b-instruct`
- `gemma-2-9b-it`
- `mistral-7b-instruct`
- `qwen2.5-7b-instruct`

## LangChain Araçları

Agent aşağıdaki araçları kullanır:

1. **MongoDBConnectionTool**: Veritabanı bağlantısı yönetimi
2. **MongoDBQueryTool**: Sorgu çalıştırma
3. **MongoDBSchemaAnalyzer**: Şema analizi ve koleksiyon listesi

## Hata Ayıklama

### MongoDB Bağlantı Sorunları

```bash
# Bağlantıyı test edin
python -c "import pymongo; print(pymongo.MongoClient().admin.command('ping'))"
```

### LM Studio Sorunları

```bash
# LM Studio API'yi test edin
curl http://localhost:1234/v1/models
```

## Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.
