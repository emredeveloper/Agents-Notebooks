import os
import sys
import json
import time
import socket
import subprocess
import httpx
import streamlit as st


ORCH_URL = os.getenv("ORCH_URL", "http://localhost:8100")
MATH_URL = os.getenv("MATH_URL", "http://localhost:8001")
WRITER_URL = os.getenv("WRITER_URL", "http://localhost:8002")
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8003")


def call_agent(base_url: str, text: str, req_id: str = "1") -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "agent.sendMessage",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
                "messageId": req_id,
            }
        },
    }
    try:
        with httpx.Client(timeout=10) as client:
            r = client.post(f"{base_url}/", json=payload)
            return r.json()
    except Exception as e:
        # UI çökmesin; hata bilgisini döndür.
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32002, "message": f"Bağlantı hatası: {e}", "target": base_url},
        }


def extract_text(resp: dict) -> str:
    try:
        return resp["result"]["message"]["parts"][0]["text"]
    except Exception:
        return json.dumps(resp)


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _script_path(filename: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))


def _start_agent(script_filename: str, port: int) -> subprocess.Popen | None:
    if _is_port_open("127.0.0.1", port):
        return None
    py = sys.executable
    script = _script_path(script_filename)
    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        start_new_session = True
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{os.path.splitext(os.path.basename(script_filename))[0]}.log")
    log_file = open(log_path, "a", encoding="utf-8")
    # Ortamı kopyala; embedding modeli varsayılanını geçir
    env = os.environ.copy()
    env.setdefault("LMSTUDIO_EMBED_MODEL", "text-embedding-mxbai-embed-large-v1")
    proc = subprocess.Popen(
        [py, script],
        stdout=log_file,
        stderr=log_file,
        creationflags=creationflags,
        start_new_session=start_new_session,
        env=env,
    )
    # Wait for readiness (agent card)
    base = f"http://localhost:{port}"
    url = f"{base}/.well-known/agent-card.json"
    for _ in range(30):  # ~6s
        try:
            with httpx.Client(timeout=0.2) as client:
                r = client.get(url)
                if r.status_code == 200:
                    break
        except Exception:
            pass
        time.sleep(0.2)
    return proc


def ensure_agents_started():
    if "_agent_procs" not in st.session_state:
        st.session_state["_agent_procs"] = {}

    specs = [
        ("MathAgent", "math_agent.py", 8001),
        ("WriterAgent", "writer_agent.py", 8002),
        ("EmbeddingSearchAgent", "embedding_agent.py", 8003),
        ("Orchestrator", "orchestrator.py", 8100),
    ]
    for name, script, port in specs:
        if not _is_port_open("127.0.0.1", port):
            proc = _start_agent(script, port)
            if proc is not None:
                st.session_state["_agent_procs"][name] = proc.pid


def _kill_pid(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.call(["taskkill", "/PID", str(pid), "/F", "/T"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, 9)
    except Exception:
        pass


def restart_agent(name: str, script: str, port: int) -> None:
    # Mevcut PID'i kapat
    pid = st.session_state.get("_agent_procs", {}).get(name)
    if pid:
        _kill_pid(pid)
        time.sleep(0.5)
    # Port dinliyorsa mümkünse sistem PID'ini kapat (Windows)
    if _is_port_open("127.0.0.1", port) and os.name == "nt":
        try:
            out = subprocess.check_output(["cmd", "/c", f"netstat -ano | findstr :{port}"], text=True)
            for line in out.splitlines():
                parts = line.split()
                if parts and parts[-1].isdigit():
                    _kill_pid(int(parts[-1]))
        except Exception:
            pass
        time.sleep(0.5)
    # Yeniden başlat
    proc = _start_agent(script, port)
    if proc is not None:
        st.session_state.setdefault("_agent_procs", {})[name] = proc.pid


def list_agent_statuses():
    statuses = []
    for name, port in [("Orchestrator", 8100), ("MathAgent", 8001), ("WriterAgent", 8002), ("Embedding", 8003)]:
        up = _is_port_open("127.0.0.1", port)
        statuses.append((name, port, up))
    return statuses


def main():
    st.set_page_config(page_title="A2A Demo", layout="wide")
    st.title("A2A Çoklu Ajan Demo")
    st.caption("Orchestrator, MathAgent, WriterAgent, EmbeddingSearchAgent")

    # Agents auto-start
    ensure_agents_started()

    with st.sidebar:
        st.header("Ajan Uçları")
        st.write(ORCH_URL)
        st.write(MATH_URL)
        st.write(WRITER_URL)
        st.write(EMBED_URL)
        st.subheader("Durum")
        for name, port, up in list_agent_statuses():
            st.write(f"{name} ({port}): {'✅' if up else '⛔'}")
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        st.caption(f"Loglar: {logs_dir}")
        st.caption("Ajanlar başlamazsa önce `pip install -r requirements.txt` çalıştırın.")
        st.divider()
        if st.button("Tüm Ajanları Yeniden Başlat"):
            restart_agent("MathAgent", "math_agent.py", 8001)
            restart_agent("WriterAgent", "writer_agent.py", 8002)
            restart_agent("EmbeddingSearchAgent", "embedding_agent.py", 8003)
            restart_agent("Orchestrator", "orchestrator.py", 8100)
            st.rerun()

    tabs = st.tabs([
        "Orchestrator",
        "MathAgent",
        "WriterAgent",
        "EmbeddingSearch",
        "Hakkında",
    ])

    with tabs[0]:
        st.info("Kullanıcı mesajını uygun ajanlara yönlendirir. İşlem varsa önce Math, sonra Writer çağrılır.")
        q = st.text_area("Mesaj", "12 + 4 topla ve sonucu bir paragrafta açıkla")
        if st.button("Gönder", key="orch"):
            resp = call_agent(ORCH_URL, q, "orch-1")
            st.code(json.dumps(resp, ensure_ascii=False, indent=2))
            st.success(extract_text(resp))

    with tabs[1]:
        st.info("Basit aritmetik işlemleri yanıtlar (topla/çıkar/çarp/böl). Dosya: `a2a_demo/math_agent.py`")
        q = st.text_input("MathAgent prompt", "12 + 4 topla")
        if st.button("Gönder", key="math"):
            resp = call_agent(MATH_URL, q, "math-1")
            st.code(json.dumps(resp, ensure_ascii=False, indent=2))
            st.success(extract_text(resp))

    with tabs[2]:
        st.info("LM Studio ile kısa metin üretir. Model doğrulama ve uç fallback içerir. Dosya: `a2a_demo/writer_agent.py`")
        q = st.text_area("WriterAgent prompt", "Kısa bir paragraf yaz: çoklu ajan demo")
        if st.button("Gönder", key="writer"):
            resp = call_agent(WRITER_URL, q, "writer-1")
            st.code(json.dumps(resp, ensure_ascii=False, indent=2))
            st.success(extract_text(resp))

    with tabs[3]:
        st.info("Embedding modeli ile küçük bir bilgi kümesinde vektör arama yapar. Dosya: `a2a_demo/embedding_agent.py`")
        q = st.text_input("Arama ifadesi", "A2A nedir?")
        if st.button("Gönder", key="embed"):
            resp = call_agent(EMBED_URL, q, "embed-1")
            st.code(json.dumps(resp, ensure_ascii=False, indent=2))
            st.success(extract_text(resp))

    with tabs[4]:
        st.markdown("""
        - **Orchestrator (`a2a_demo/orchestrator.py`)**: İstekleri uygun ajanlara yönlendirir, sonuçları birleştirir.
        - **MathAgent (`a2a_demo/math_agent.py`)**: Basit aritmetik işlemler.
        - **WriterAgent (`a2a_demo/writer_agent.py`)**: LM Studio ile metin üretimi; `/v1/chat/completions` ve alternatif uçlar.
        - **EmbeddingSearchAgent (`a2a_demo/embedding_agent.py`)**: `text-embedding-mxbai-embed-large-v1` ile vektör arama.
        - **Ortak Yardımcılar (`a2a_demo/common.py`)**: LM Studio istemcisi, agent card, JSON-RPC yardımcıları.
        """)


if __name__ == "__main__":
    # streamlit run a2a_demo/ui_streamlit.py
    main()


