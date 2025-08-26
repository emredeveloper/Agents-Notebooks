# MongoDB Natural Language Agent

Bu proje, MongoDB veritabanlarıyla doğal dilde konuşmanızı sağlayan akıllı bir agent'tır. LM Studio entegrasyonu ile herhangi bir MongoDB koleksiyonu ve alanıyla çalışabilir.

## 🚀 Özellikler

- **Doğal Dil Anlama**: "adı Ahmet olanları bul" gibi doğal dil sorguları
- **Dinamik Koleksiyon Tespiti**: Herhangi bir koleksiyon adıyla çalışır
- **Akıllı Veri Ekleme**: "yeni kullanıcı ekle" gibi komutlarla veri ekleme
- **Otomatik Şema Analizi**: Mevcut alanları otomatik tespit eder
- **Web Arayüzü**: Kullanıcı dostu modern web arayüzü
- **LM Studio Entegrasyonu**: Yerel LLM desteği

## 📋 Gereksinimler

- Python 3.8+
- MongoDB (yerel veya uzak)
- LM Studio (yerel LLM için)
- Gerekli Python paketleri (requirements.txt'te)

## 🛠 Kurulum

1. **Repository'yi klonlayın:**
   ```bash
   git clone <repository-url>
   cd mongodb-langchain-agent
   ```

2. **Sanal ortam oluşturun:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # veya
   venv\Scripts\activate     # Windows
   ```

3. **Bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

4. **LM Studio'yu başlatın:**
   - LM Studio'yu indirin ve çalıştırın
   - Bir model yükleyin (örn: Qwen, Gemma)
   - Server'ı port 1234'te başlatın

5. **MongoDB'yi başlatın:**
   - MongoDB'nin çalıştığından emin olun
   - Varsayılan: `mongodb://localhost:27017/`

## 🎯 Kullanım

### Web Arayüzü ile

```bash
python mongodb-langchain-agent-clean.py
```

Tarayıcıda `http://localhost:5000` adresine gidin.

### Örnek Sorgular

#### Veri Sorgulama:
- "koleksiyonları listele"
- "users tablosundaki ilk 5 veriyi göster"
- "adı Ahmet olan kullanıcıları bul"
- "yaşı 25'ten büyük olanları listele"
- "fiyatı 100 ile 500 arasındaki ürünler"

#### Veri Ekleme:
- "adı Mehmet soyadı Kaya yaşı 30 olan kullanıcı ekle"
- "yeni ürün ekle: laptop, fiyat 15000"
- "sipariş ekle: müşteri Ayşe, tutar 250"

#### İstatistikler:
- "kaç kullanıcı var?"
- "products koleksiyonunda kaç kayıt var?"
- "toplam kayıt sayısı nedir?"

## ⚙️ Yapılandırma

### Veritabanı Bağlantısı

```python
# mongodb-langchain-agent-clean.py dosyasında
agent = MongoDBLangChainAgent(
    mongo_uri="mongodb://localhost:27017/",  # MongoDB URI
    lm_studio_url="http://localhost:1234/v1", # LM Studio URL
    model_name="your-model-name"              # LLM Model adı
)
```

### LM Studio Ayarları

```python
# LMStudioLLM sınıfında
base_url: str = "http://localhost:1234/v1"  # LM Studio URL
model_name: str = "qwen/qwen3-4b-2507"      # Model adı
```

## 📁 Proje Yapısı

```
mongodb-langchain-agent/
├── mongodb-langchain-agent-clean.py  # Ana uygulama
├── templates/
│   └── index.html                    # Web arayüzü
├── static/
│   ├── css/
│   │   └── style.css                # Stil dosyası
│   └── js/
│       └── app.js                   # JavaScript
├── requirements.txt                  # Python bağımlılıkları
└── README.md                        # Bu dosya
```

## 🔧 Özelleştirme

### Yeni Koleksiyon Türleri Ekleme

```python
# _detect_collection_from_query metodunda
collection_patterns = {
    'users': ['user', 'kullanıcı', 'kişi', 'person'],
    'products': ['product', 'ürün', 'item'],
    'orders': ['order', 'sipariş', 'purchase'],
    'customers': ['customer', 'müşteri', 'client'],
    # Yeni türler buraya eklenebilir
}
```

### Örnek Veri Şablonları

```python
# _add_sample_data metodunda yeni veri türleri ekleyebilirsiniz
elif 'custom_collection' in collection_name.lower():
    sample_data = [
        {"field1": "value1", "field2": 123, "created_at": datetime.now()},
        # Özel verileriniz
    ]
```

## 🐛 Sorun Giderme

### LM Studio Bağlantı Hatası
- LM Studio'nun çalıştığından emin olun
- Port 1234'ün açık olduğunu kontrol edin
- Model'in yüklendiğini doğrulayın

### MongoDB Bağlantı Hatası
- MongoDB servisinin çalıştığından emin olun
- Bağlantı string'ini kontrol edin
- Firewall ayarlarını kontrol edin

### Parsing Hataları
- LLM model'ini değiştirmeyi deneyin
- Temperature değerini düşürün
- Prompt'ları basitleştirin

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🙏 Teşekkürler

- [LangChain](https://github.com/langchain-ai/langchain) - Agent framework
- [LM Studio](https://lmstudio.ai/) - Yerel LLM desteği
- [MongoDB](https://www.mongodb.com/) - Veritabanı
- [Flask](https://flask.palletsprojects.com/) - Web framework
