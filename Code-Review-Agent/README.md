# 💻 Code Review Agent

Bu agent, LM Studio destekli kod analizi ve review hizmetleri sağlar. Güvenlik, performans, kod kalitesi ve bug tespiti gibi çeşitli analiz türleri yapar.

## 🎯 Özellikler

- **🔒 Güvenlik Analizi**: Vulnerability ve security açıkları tespiti
- **⚡ Performans İncelemesi**: Algoritma ve optimizasyon önerileri
- **🏆 Kod Kalitesi**: SOLID, DRY ve best practice değerlendirmesi
- **🐛 Bug Analizi**: Potansiyel hata ve edge case tespiti
- **🎨 Style İncelemesi**: Coding conventions ve formatting
- **🤖 Otomatik Dil Tespiti**: 10+ programlama dili desteği
- **⚡ A2A Protokol Uyumlu**: JSON-RPC ve agent card desteği

## 🚀 Kurulum

### Gereksinimler
```bash
pip install fastapi uvicorn httpx
```

### LM Studio Kurulumu
1. LM Studio'yu indirin ve başlatın
2. Kod analizi için uygun model yükleyin (Code-focused model önerilir)
3. "OpenAI compatible server" özelliğini aktif edin

## 📋 Kullanım

### Agent'ı Başlatma
```bash
cd Code-Review-Agent
python code_review_agent.py
```

Default port: `8007`

### Ortam Değişkenleri
```bash
export CODE_REVIEW_AGENT_PORT=8007
export LMSTUDIO_BASE_URL=http://localhost:1234
export LMSTUDIO_MODEL=qwen/qwen3-4b-2507
```

### Test Etme
```bash
# Agent card bilgisini görüntüle
curl http://localhost:8007/.well-known/agent-card.json | jq

# Code review testi
curl -X POST http://localhost:8007/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "agent.sendMessage",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Bu Python kodunu güvenlik açısından incele:\n```python\ndef login(username, password):\n    query = \"SELECT * FROM users WHERE name=\'\''\" + username + \"\'\'' AND pass=\'\''\" + password + \"\'\'';\"\n    return db.execute(query)\n```"}]
      }
    }
  }' | jq
```

## 🔍 Review Türleri

### 🔒 Güvenlik Analizi
Kod güvenlik açıklarını tespit eder.

**Komutlar:**
- "güvenlik [kod]"
- "security review [kod]"
- "zafiyet [kod]"
- "vulnerability [kod]"

**İnceleme Alanları:**
- SQL Injection, XSS
- Input validation
- Authentication/Authorization
- Sensitive data handling
- Crypto kullanımı

### ⚡ Performans İncelemesi
Kod performansını analiz eder.

**Komutlar:**
- "performans [kod]"
- "performance [kod]"
- "optimizasyon [kod]"
- "yavaş [kod]"

**İnceleme Alanları:**
- Big O complexity
- Memory usage
- I/O optimization
- Loop efficiency
- Database queries

### 🏆 Kod Kalitesi
Best practice ve clean code principles.

**Komutlar:**
- "kalite [kod]"
- "quality [kod]"
- "clean code [kod]"
- "best practice [kod]"

**İnceleme Alanları:**
- SOLID principles
- DRY principle
- Naming conventions
- Code readability
- Function design

### 🐛 Bug Analizi
Potansiyel hataları tespit eder.

**Komutlar:**
- "bug [kod]"
- "hata [kod]"
- "error [kod]"
- "problem [kod]"

**İnceleme Alanları:**
- Logic errors
- Null pointer exceptions
- Array index errors
- Type mismatches
- Resource leaks

### 🎨 Style İncelemesi
Code formatting ve conventions.

**Komutlar:**
- "stil [kod]"
- "style [kod]"
- "format [kod]"
- "convention [kod]"

**İnceleme Alanları:**
- Indentation
- Naming conventions
- Comment quality
- Line length
- Whitespace

## 🔤 Desteklenen Diller

### Ana Diller
- **🐍 Python**: def, class, import, if __name__
- **📜 JavaScript/TypeScript**: function, const, let, =>
- **☕ Java**: public class, static void main
- **🔷 C#**: using System, Console.WriteLine
- **⚙️ C++**: #include, std::, cout

### Diğer Diller
- **🌐 HTML**: HTML tags ve structure
- **🎨 CSS**: Selectors ve properties
- **🗃️ SQL**: SELECT, INSERT, UPDATE queries
- **🚀 Go**: package main, func
- **🦀 Rust**: fn main(), println!

## 📊 Kod Format Desteği

### Code Blocks
```python
def example():
    return "Hello World"
```

### Indented Code
    def example():
        return "Hello World"

### Inline Code
Agent metinden kod bloklarını otomatik tespit eder.

## 🔧 Orchestrator Entegrasyonu

Bu agent'ı orchestrator.py'ye eklemek için:

```python
# orchestrator.py içinde
CODE_REVIEW_URL = os.getenv("CODE_REVIEW_AGENT_URL", "http://localhost:8007")

# Code review isteği algılama
wants_code_review = any(s in user_text.lower() for s in [
    "kod", "code", "review", "incele", "analiz", "güvenlik", "performans", "bug"
])

if wants_code_review:
    review_payload = {
        "jsonrpc": "2.0",
        "id": "review-1",
        "method": "agent.sendMessage",
        "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": user_text}]}}
    }
    review_resp = await send_message(CODE_REVIEW_URL, review_payload)
    # ... handle response
```

## 🎨 UI Streamlit Entegrasyonu

UI'ye eklemek için agent listesine ekleyin:

```python
# ui_streamlit.py içinde
agents = {
    "CodeReview": {"url": "http://localhost:8007", "port": 8007},
    # ... diğer agents
}
```

## 📈 Review Örnekleri

### Güvenlik Analizi Örneği
**Giriş:**
```python
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}' AND pass='{password}'"
    return db.execute(query)
```

**Çıkış:**
```
🔒 Güvenlik İncelemesi

🔧 Dil: Python

❌ KRİTİK GÜVENLİK AÇIĞI - SQL Injection
Bu kod SQL injection saldırılarına açık. Kullanıcı girişi doğrudan SQL sorgusuna ekleniyor.

Öneriler:
• Parameterized queries kullanın
• Input validation ekleyin  
• ORM veya prepared statements tercih edin
```

### Performans Analizi Örneği
**Giriş:**
```python
def find_duplicates(lst):
    duplicates = []
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            if lst[i] == lst[j] and lst[i] not in duplicates:
                duplicates.append(lst[i])
    return duplicates
```

**Çıkış:**
```
⚡ Performans İncelemesi

🔧 Dil: Python

⚠️ PERFORMANS SORUNU - O(n³) Complexity
Nested loops ve 'not in' kontrolü algoritma karmaşıklığını artırıyor.

Öneriler:
• Set kullanarak O(n) complexity'e indirin
• Counter veya defaultdict kullanın
• Early termination ekleyin
```

## 🔍 Hata Ayıklama

### Yaygın Sorunlar

**LM Studio bağlantı hatası:**
- LM Studio'nun çalıştığından emin olun
- Kod analizi için uygun model kullanın
- Base URL'in doğru olduğunu kontrol edin

**Kod tespit edilemedi:**
- Kod bloklarını ``` ile çevrelerin
- En az 4 boşluk indentation kullanın
- Yeterli uzunlukta kod girin (10+ karakter)

**Dil tespit hatası:**
- Kod syntax'ının doğru olduğunu kontrol edin
- Belirgin language keywords kullanın
- Comment'lerde dil belirtin

## 🚀 En İyi Uygulamalar

### Model Seçimi
- **Code-focused models**: CodeLlama, StarCoder
- **General models**: GPT-4, Claude için yüksek temperature
- **Memory**: Büyük kod blokları için yeterli context

### Prompt Engineering
- Specific review type belirtin
- Kod context'i açıklayın
- Expected output format belirtin

## 🌟 Gelecek Özellikler

- [ ] Çoklu dosya review desteği
- [ ] Git diff analizi
- [ ] Automated testing suggestions
- [ ] Code refactoring önerileri
- [ ] Dependency security scanning
- [ ] Code complexity metrics
- [ ] Custom rule sets
- [ ] IDE extensions
- [ ] Batch processing
- [ ] Report generation

## 🤝 Katkıda Bulunma

Bu agent'ı geliştirmek için:
1. Yeni programlama dilleri ekleyin
2. Review türlerini genişletin
3. Advanced static analysis ekleyin
4. UI/UX iyileştirmeleri yapın
5. Performance optimizasyonları ekleyin

---

**Not**: Bu agent eğitim ve geliştirme amaçlıdır. Production kod review'ları için ek security tools ve human review önerilir.