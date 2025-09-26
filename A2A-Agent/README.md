# A2A Demo (Multi-agent with LM Studio)

This demo provides A2A-like endpoints with two simple agents (MathAgent, WriterAgent) and an Orchestrator. It uses LM Studio's OpenAI-compatible local server for LLM responses.

## 🚀 Quickstart

```bash
cd A2A-Agent

# Start in separate terminals
python math_agent.py
python writer_agent.py
python embedding_agent.py
python orchestrator.py
```

## 📸 Screenshots

<p align="center">
  <img src="./A2A-Demo-08-22-2025_01_53_PM.png" alt="A2A Demo - Screen 1" width="900">
  <br/>
  <em>Screen 1</em>
  <br/><br/>
  <img src="./A2A-Demo-08-22-2025_01_54_PM.png" alt="A2A Demo - Screen 2" width="900">
  <br/>
  <em>Screen 2</em>
  <br/><br/>
  <img src="./A2A-Demo-08-22-2025_01_54_PM%20%281%29.png" alt="A2A Demo - Screen 3" width="900">
  <br/>
  <em>Screen 3</em>
  <br/><br/>
  <img src="./A2A-Demo-08-22-2025_01_55_PM.png" alt="A2A Demo - Screen 4" width="900">
  <br/>
  <em>Screen 4</em>
  <br/>

</p>

## Setup

1) Dependencies

```bash
pip install -r requirements.txt
```

2) LM Studio
- Start a model in LM Studio and run the "OpenAI compatible server" at an address such as `http://localhost:1234/v1`.

## Run

Use three terminals:

```bash
python math_agent.py
```

```bash
python writer_agent.py
```

```bash
python orchestrator.py
```

Then you can send requests to the orchestrator:

```bash
curl -s http://localhost:8100/.well-known/agent-card.json | jq

curl -s -X POST http://localhost:8100/ -H "Content-Type: application/json" -d '{
  "jsonrpc":"2.0",
  "id":"1",
  "method":"agent.sendMessage",
  "params":{
    "message":{"role":"user","parts":[{"kind":"text","text":"How much is 10 USD in TRY? Write a short paragraph."}],"messageId":"m1"}
  }
}' | jq
```

Note: This demo has no auth/security.

## 🖥️ Optional Web UI

A simple UI is available that auto-starts agents and lets you test them:

```bash
cd A2A-Agent
streamlit run ui_streamlit.py
```

Default ports:
- MathAgent: 8001
- WriterAgent: 8002
- EmbeddingSearchAgent: 8003
- Orchestrator: 8100

Note: default env for LM Studio embedding model: `LMSTUDIO_EMBED_MODEL=text-embedding-mxbai-embed-large-v1`.

