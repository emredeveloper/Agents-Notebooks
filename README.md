Langraph agent demo

This small script runs a LangGraph StateGraph that calls a local OpenAI-compatible LLM (LM Studio, LM Deploy, etc.).

Quick start (Windows cmd.exe)

Set environment variables if you need a custom endpoint or model:

```cmd
set LG_BASE_URL=http://127.0.0.1:1234/v1
set LG_API_KEY=lm-studio
set LG_MODEL=google/gemma-3n-e4b
```

Activate your venv and run:

```cmd
python langraph_app.py
```

Features added:
- Configurable via env vars
- Retry on transient network errors
- Proper role mapping for messages (user/assistant/system)
- Max-turn guard to avoid infinite loops
- Logging for easier debugging

### In-memory checkpoint / thread demo

`langraph_stream_memory.py` shows how to use `InMemorySaver` so each `thread_id` keeps an isolated conversation. Run it:

```cmd
python langraph_stream_memory.py
```

It will:
 
1. Start thread 1, introduce a name.
2. Ask if the model remembers the name (same thread, history present).
3. Ask the same question in thread 2 (fresh history) to contrast behavior.


Adjust environment variables (`LG_BASE_URL`, `LG_MODEL`, etc.) as in the basic example if needed.

### Persona Branching Demo

`langraph_branch_personas.py` aynı prompt'u farklı persona (system mesajı) ile paralel thread'lerde çalıştırır ve cevapları karşılaştırmalı diff ile gösterir.

Çalıştır:

```cmd
python langraph_branch_personas.py --prompt "Kısa bir motivasyon cümlesi yaz" --temperature 0.7
```

Persona listesini görmek için:

```cmd
python langraph_branch_personas.py --list-personas
```

Örnek personelar: friendly, formal, teacher, skeptic.

### Dinamik Sıcaklık (Temperature) Demo

`langraph_dynamic_temperature.py` prompt türünü basit kurallarla sınıflandırıp modele gönderilen `temperature` değerini otomatik seçer.

Çalıştırma örneği:

```cmd
python langraph_dynamic_temperature.py --prompt "Kısa bir motivasyon cümlesi yaz" --show-rationale --compare
```

Parametreler:

- `--show-rationale` sınıflandırma gerekçesini yazdırır.
- `--compare` dinamik ve sabit sıcaklık çıktısını yan yana gösterir.
- `--fixed-temperature 0.7` sabit referans sıcaklık.

If you don't have a local LLM, point `LG_BASE_URL` to a reachable OpenAI-compatible server and set `LG_API_KEY` appropriately.
