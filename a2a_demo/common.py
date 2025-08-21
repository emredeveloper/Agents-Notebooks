import os
import httpx
from typing import Optional, Tuple, List


LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")


def _lmstudio_base_variants() -> Tuple[str, str]:
    base = LMSTUDIO_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        base_v1 = base
        base_root = base[: -len("/v1")]
    else:
        base_root = base
        base_v1 = f"{base_root}/v1"
    return base_root, base_v1


async def list_lmstudio_models() -> List[str]:
    """LM Studio'dan model listesini getirir (GET /v1/models veya /models)."""
    base_root, base_v1 = _lmstudio_base_variants()
    headers = {"Authorization": f"Bearer {LMSTUDIO_API_KEY}"}
    endpoints = [f"{base_v1}/models", f"{base_root}/models"]
    async with httpx.AsyncClient(timeout=30) as client:
        last_err: Optional[str] = None
        for url in endpoints:
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 404:
                    last_err = f"404 Not Found: {url}"
                    continue
                resp.raise_for_status()
                data = resp.json()
                items = data.get("data") or []
                return [m.get("id") for m in items if isinstance(m, dict) and m.get("id")]
            except Exception as e:
                last_err = f"{type(e).__name__}: {e} at {url}"
                continue
    # Model listesi alınamazsa boş liste döndür.
    return []


async def select_lmstudio_model(preferred: Optional[str]) -> str:
    """Tercih edilen model uygunsa onu; değilse mevcut modellerden ilkini döndürür."""
    preferred_model = preferred or "qwen/qwen3-4b-2507"
    try:
        models = await list_lmstudio_models()
        if preferred_model in models:
            return preferred_model
        if models:
            return models[0]
    except Exception:
        pass
    return preferred_model


async def call_lmstudio(prompt: str, model: Optional[str] = None, temperature: float = 0.2) -> str:
    headers = {"Authorization": f"Bearer {LMSTUDIO_API_KEY}", "Content-Type": "application/json"}
    model_name = await select_lmstudio_model(model)
    base_root, base_v1 = _lmstudio_base_variants()

    chat_payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    comp_payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
    }

    candidates = [
        (f"{base_v1}/chat/completions", chat_payload, "chat"),
        (f"{base_root}/chat/completions", chat_payload, "chat"),
        (f"{base_v1}/completions", comp_payload, "text"),
        (f"{base_root}/completions", comp_payload, "text"),
    ]

    last_err: Optional[str] = None
    async with httpx.AsyncClient(timeout=60) as client:
        for url, payload, mode in candidates:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 404:
                    last_err = f"404 Not Found: {url}"
                    continue
                resp.raise_for_status()
                data = resp.json()
                if mode == "chat":
                    return data["choices"][0]["message"]["content"]
                else:
                    # text completions
                    return data["choices"][0]["text"]
            except Exception as e:
                last_err = f"{type(e).__name__}: {e} at {url}"
                continue

    raise RuntimeError(last_err or "LM Studio endpoint not reachable")


async def select_lmstudio_embedding_model(preferred: Optional[str]) -> str:
    """Embedding için uygun modeli seç (id içinde embed/embedding geçen)."""
    # Eğer belirli bir model geçildiyse doğrudan dön (UI tarafından sabitlenmiş olabilir)
    if preferred:
        return preferred
    try:
        models = await list_lmstudio_models()
        candidates = [m for m in models if any(k in m.lower() for k in ["embed", "embedding", "text-embedding"])]
        if candidates:
            return candidates[0]
    except Exception:
        pass
    # Fallback: bilinen iyi bir embed modeli
    return "text-embedding-mxbai-embed-large-v1"


async def embed_with_lmstudio(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """Embeddings uçlarını dener (/v1/embeddings, /embeddings) ve vektörleri döndürür.

    Not: LM Studio'da embed modeli yüklü olmalı (ör. text-embedding-mxbai-embed-large-v1).
    """
    headers = {"Authorization": f"Bearer {LMSTUDIO_API_KEY}", "Content-Type": "application/json"}
    model_name = await select_lmstudio_embedding_model(model)
    base_root, base_v1 = _lmstudio_base_variants()
    payload = {"model": model_name, "input": texts}
    endpoints = [f"{base_v1}/embeddings", f"{base_root}/embeddings"]
    last_err: Optional[str] = None
    async with httpx.AsyncClient(timeout=60) as client:
        for url in endpoints:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 404:
                    last_err = f"404 Not Found: {url}"
                    continue
                resp.raise_for_status()
                data = resp.json()
                # OpenAI uyumlu: data.data[0].embedding
                arr: List[List[float]] = []
                for item in data.get("data", []):
                    emb = item.get("embedding")
                    if emb:
                        arr.append(emb)
                if len(arr) != len(texts):
                    raise RuntimeError(
                        f"Embedding count mismatch: expected {len(texts)} got {len(arr)} for model {model_name}"
                    )
                return arr
            except Exception as e:
                last_err = f"{type(e).__name__}: {e} at {url}"
                continue
    raise RuntimeError(last_err or "LM Studio embedding endpoint not reachable")


AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"
EXTENDED_AGENT_CARD_PATH = "/agent/authenticatedExtendedCard"


def build_public_agent_card(name: str, description: str, base_url: str, streaming: bool = False) -> dict:
    return {
        "name": name,
        "description": description,
        "capabilities": {
            "streaming": streaming,
            "input": {"modalities": ["text"]},
            "output": {"modalities": ["text"]},
        },
        "endpoints": {
            "rpcUrl": f"{base_url}/",
            "agentCardUrl": f"{base_url}{AGENT_CARD_WELL_KNOWN_PATH}",
        },
        "supportsAuthenticatedExtendedCard": False,
    }


def jsonrpc_success(id_val: str, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": id_val, "result": result}


def jsonrpc_error(id_val: str, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_val, "error": {"code": code, "message": message}}


