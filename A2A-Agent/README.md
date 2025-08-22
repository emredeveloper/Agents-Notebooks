# A2A Demo (LM Studio entegrasyonlu çoklu ajan)

Bu demo, iki basit ajan (MathAgent, WriterAgent) ve bir Orchestrator ile A2A benzeri uçlar sağlar. LLM yanıtları için LM Studio'nun OpenAI uyumlu yerel sunucusu kullanılır.

## 🚀 Hızlı Başlangıç

```bash
cd A2A-Agent

# Ayrı terminallerde başlatın
python math_agent.py
python writer_agent.py
python embedding_agent.py
python orchestrator.py
```

## 📸 Ekran Görüntüleri

<p align="center">
  <img src="./A2A-Demo-08-22-2025_01_53_PM.png" alt="A2A Demo - Ekran 1" width="900">
  <br/>
  <em>Ekran 1</em>
  <br/><br/>
  <img src="./A2A-Demo-08-22-2025_01_54_PM.png" alt="A2A Demo - Ekran 2" width="900">
  <br/>
  <em>Ekran 2</em>
  <br/><br/>
  <img src="./A2A-Demo-08-22-2025_01_54_PM%20%281%29.png" alt="A2A Demo - Ekran 3" width="900">
  <br/>
  <em>Ekran 3</em>
  <br/><br/>
  <img src="./A2A-Demo-08-22-2025_01_55_PM.png" alt="A2A Demo - Ekran 4" width="900">
  <br/>
  <em>Ekran 4</em>
  <br/>

</p>

## Kurulum

1) Bağımlılıklar

```bash
pip install -r requirements.txt
```

2) LM Studio
- LM Studio'da bir model başlatın ve "OpenAI compatible server" özelliğini  `http://localhost:1234/v1` gibi bir adreste çalıştırın.

## Çalıştırma

Üç terminalde:

```bash
python math_agent.py
```

```bash
python writer_agent.py
```

```bash
python orchestrator.py
```

Ardından orchestrator'a istek atabilirsiniz:

```bash
curl -s http://localhost:8100/.well-known/agent-card.json | jq

curl -s -X POST http://localhost:8100/ -H "Content-Type: application/json" -d '{
  "jsonrpc":"2.0",
  "id":"1",
  "method":"agent.sendMessage",
  "params":{
    "message":{"role":"user","parts":[{"kind":"text","text":"10 USD kaç TRY eder ve kısa bir paragraf yaz."}],"messageId":"m1"}
  }
}' | jq
```

Not: Bu demo güvenlik/kimlik doğrulama içermemektedir.

## 🖥️ Opsiyonel Web UI

Ajanları otomatik başlatan ve test etmenizi sağlayan basit bir UI de mevcuttur:

```bash
cd A2A-Agent
streamlit run ui_streamlit.py
```

Varsayılan portlar:
- MathAgent: 8001
- WriterAgent: 8002
- EmbeddingSearchAgent: 8003
- Orchestrator: 8100

Not: LM Studio embedding modeli için varsayılan env: `LMSTUDIO_EMBED_MODEL=text-embedding-mxbai-embed-large-v1`.


