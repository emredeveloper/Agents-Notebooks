import os
from typing import List

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

try:
    from a2a_demo.common import (
        AGENT_CARD_WELL_KNOWN_PATH,
        build_public_agent_card,
        jsonrpc_error,
        jsonrpc_success,
        embed_with_lmstudio,
    )
except ImportError:  # allow running from inside a2a_demo directory
    from common import (
        AGENT_CARD_WELL_KNOWN_PATH,
        build_public_agent_card,
        jsonrpc_error,
        jsonrpc_success,
        embed_with_lmstudio,
    )


PORT = int(os.getenv("EMBED_AGENT_PORT", 8003))
BASE_URL = os.getenv("EMBED_AGENT_BASE_URL", f"http://localhost:{PORT}")
EMBED_MODEL = os.getenv("LMSTUDIO_EMBED_MODEL", "text-embedding-mxbai-embed-large-v1")

app = FastAPI(title="EmbeddingSearchAgent")


DOCS: List[str] = [
    "A2A, ajanlar arası birlikte çalışabilirlik sağlayan açık bir protokoldür.",
    "LM Studio yerel LLM çalıştırmayı ve OpenAI uyumlu API'yi destekler.",
    "MathAgent basit aritmetik işlemleri yapar.",
    "WriterAgent kısa metin üretir ve LM Studio ile konuşur.",
    "Orchestrator, kullanıcı niyetine göre ajanlara yönlendirir.",
]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


@app.get(AGENT_CARD_WELL_KNOWN_PATH)
async def agent_card():
    card = build_public_agent_card(
        name="EmbeddingSearchAgent",
        description="Basit vektör arama (LM Studio embeddings ile)",
        base_url=BASE_URL,
        streaming=False,
    )
    return JSONResponse(card)


@app.post("/")
async def rpc_root(request: Request):
    body = await request.json()
    method = body.get("method")
    id_val = str(body.get("id", "0"))

    if method != "agent.sendMessage":
        return JSONResponse(jsonrpc_error(id_val, -32601, "Method not found"), status_code=404)

    params = body.get("params", {})
    message = params.get("message", {})
    parts = message.get("parts", [])
    query = ""
    for p in parts:
        if p.get("kind") == "text":
            query += p.get("text", "") + "\n"

    try:
        docs = [d for d in DOCS if d and d.strip()]
        if not docs:
            raise RuntimeError("Belge kümesi boş")
        # LM Studio embed modeli yüklemesini tetiklemek için küçük bir gecikme ile iki aşamada deneyelim
        doc_embeddings = await embed_with_lmstudio(docs, model=EMBED_MODEL)
        q_payload = query.strip() or "A2A protokolü nedir?"
        q_list = await embed_with_lmstudio([q_payload], model=EMBED_MODEL)
        if not q_list:
            raise RuntimeError("Sorgu embedding alınamadı (boş yanıt)")
        query_embedding = q_list[0]
        if not query_embedding:
            raise RuntimeError("Sorgu embedding alınamadı (None)")
        doc_vecs = np.array(doc_embeddings, dtype=float)
        qvec = np.array(query_embedding, dtype=float)
        if doc_vecs.ndim != 2 or qvec.ndim != 1:
            raise RuntimeError("Beklenmeyen embedding tensör şekli")
        sims = [(_cosine_sim(qvec, dvec), idx) for idx, dvec in enumerate(doc_vecs)]
        sims.sort(reverse=True)
        topk = sims[:3]
        results = [
            {"doc": docs[i], "score": round(score, 4)} for score, i in topk
        ]
        text = "\n".join([f"- ({r['score']}) {r['doc']}" for r in results])
    except Exception as e:
        text = f"EmbeddingSearchAgent hatası: {e}"

    result = {
        "contextId": message.get("contextId"),
        "message": {
            "role": "assistant",
            "parts": [{"kind": "text", "text": text}],
        },
    }
    return JSONResponse(jsonrpc_success(id_val, result))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)


