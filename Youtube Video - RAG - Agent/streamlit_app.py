"""
YouTube Video Soru-Cevap Agent - Streamlit Web Arayüzü
Bu uygulama YouTube videolarından transcript çıkarıp, içerik hakkında soru-cevap yapar.
"""

import streamlit as st
import os
import time
from typing import Optional
import traceback

# Streamlit sayfası yapılandırması
st.set_page_config(
    page_title="YouTube QA Agent",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .video-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .model-info {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    .question {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-left: 4px solid #4285f4;
    }
    .answer {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        border-left: 4px solid #34a853;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .video-embed {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stats-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .chunk-preview {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Agent'i import et
try:
    from youtube_qa_agent import YouTubeQAAgent, VideoQAState
except ImportError as e:
    st.error(f"Agent modülü yüklenemedi: {e}")
    st.stop()

def extract_video_id_for_embed(url: str) -> Optional[str]:
    """YouTube URL'den video ID'sini çıkarır (embed için)"""
    import re
    from urllib.parse import urlparse, parse_qs
    
    try:
        parsed_url = urlparse(url)
        
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query).get('v', [None])[0]
            elif parsed_url.path.startswith('/embed/'):
                return parsed_url.path.split('/embed/')[1]
        elif parsed_url.hostname in ['youtu.be']:
            return parsed_url.path[1:].split('?')[0]  # Query parametrelerini temizle
        
        return None
    except Exception:
        return None

def initialize_session_state():
    """Session state'i başlat"""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "video_state" not in st.session_state:
        st.session_state.video_state = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "video_processed" not in st.session_state:
        st.session_state.video_processed = False
    if "model_configured" not in st.session_state:
        st.session_state.model_configured = False

def create_agent(provider: str, **kwargs) -> Optional[YouTubeQAAgent]:
    """Agent oluştur"""
    try:
        agent = YouTubeQAAgent(provider=provider, **kwargs)
        return agent
    except Exception as e:
        st.error(f"Agent oluşturulamadı: {e}")
        return None

def process_video(agent: YouTubeQAAgent, video_url: str) -> Optional[VideoQAState]:
    """Video'yu işle"""
    try:
        with st.spinner("Video işleniyor... Bu birkaç dakika sürebilir."):
            state = agent.process_video(video_url)
            if state["error_message"]:
                st.error(f"Video işleme hatası: {state['error_message']}")
                return None
            return state
    except Exception as e:
        st.error(f"Video işleme sırasında hata: {e}")
        st.error(traceback.format_exc())
        return None

def ask_question(agent: YouTubeQAAgent, state: VideoQAState, question: str) -> Optional[VideoQAState]:
    """Soru sor"""
    try:
        with st.spinner("Cevap hazırlanıyor..."):
            updated_state = agent.ask_question(state, question)
            if updated_state["error_message"]:
                st.error(f"Soru yanıtlama hatası: {updated_state['error_message']}")
                return None
            return updated_state
    except Exception as e:
        st.error(f"Soru yanıtlama sırasında hata: {e}")
        return None

def main():
    """Ana uygulama"""
    initialize_session_state()
    
    # Başlık
    st.markdown("""
    # 🎥 YouTube Video Soru-Cevap Agent
    YouTube videolarından transcript çıkarıp, içerik hakkında sorularınızı yanıtlar!
    """)
    
    # Sidebar - Model Konfigürasyonu
    with st.sidebar:
        st.markdown("## ⚙️ Model Ayarları")
        
        # Provider seçimi
        provider = st.selectbox(
            "Model Sağlayıcısı",
            options=["lm_studio", "gemini"],
            format_func=lambda x: "🏠 LM Studio (Yerel)" if x == "lm_studio" else "☁️ Google Gemini (Bulut)",
            help="Yerel model için LM Studio, bulut model için Gemini seçin"
        )
        
        if provider == "lm_studio":
            st.markdown("### 🏠 LM Studio Ayarları")
            lm_studio_url = st.text_input(
                "LM Studio URL",
                value="http://localhost:1234/v1",
                help="LM Studio API endpoint'i"
            )
            model_name = st.text_input(
                "Model Adı",
                value="gemma-3n-e4b",
                help="Kullanılacak LLM model adı"
            )
            embedding_model = st.text_input(
                "Embedding Model",
                value="text-embedding-mxbai-embed-large-v1",
                help="Kullanılacak embedding model adı"
            )
            api_key = None
            
        else:  # gemini
            st.markdown("### ☁️ Gemini Ayarları")
            api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                help="Google AI Studio'dan alacağınız API key"
            )
            model_name = st.selectbox(
                "Gemini Model",
                options=["gemini-2.5-flash", "gemini-2.5-pro"],
                format_func=lambda x: f"⚡ {x} (Hızlı)" if "flash" in x else f"🧠 {x} (Güçlü)",
                help="Flash hızlı, Pro daha güçlü"
            )
            lm_studio_url = None
            embedding_model = None
        
        # Agent oluştur butonu
        if st.button("🚀 Agent'i Yapılandır", type="primary"):
            if provider == "gemini" and not api_key:
                st.error("Gemini için API key gerekli!")
            else:
                with st.spinner("Agent yapılandırılıyor..."):
                    agent_kwargs = {
                        "api_key": api_key,
                        "model_name": model_name,
                    }
                    if provider == "lm_studio":
                        agent_kwargs.update({
                            "lm_studio_url": lm_studio_url,
                            "embedding_model": embedding_model
                        })
                    
                    agent = create_agent(provider, **agent_kwargs)
                    if agent:
                        st.session_state.agent = agent
                        st.session_state.model_configured = True
                        st.success("✅ Agent başarıyla yapılandırıldı!")
                        st.rerun()
    
    # Ana içerik
    if not st.session_state.model_configured:
        st.info("👈 Önce sol menüden model ayarlarını yapılandırın.")
        return
    
    # Model bilgisi göster
    if st.session_state.agent:
        provider_name = "🏠 LM Studio" if st.session_state.agent.provider == "lm_studio" else "☁️ Google Gemini"
        st.markdown(f"""
        <div class="model-info">
            <h3>🤖 Aktif Model: {provider_name}</h3>
            <p><strong>Model:</strong> {st.session_state.agent.model_name}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Video işleme bölümü
    st.markdown("## 📹 Video İşleme")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_url = st.text_input(
            "YouTube Video URL'si",
            placeholder="https://www.youtube.com/watch?v=...",
            help="İşlemek istediğiniz YouTube video linkini girin"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Boşluk için
        process_button = st.button("🎬 Video'yu İşle", type="primary")
    
    if process_button and video_url and st.session_state.agent:
        state = process_video(st.session_state.agent, video_url)
        if state:
            st.session_state.video_state = state
            st.session_state.video_processed = True
            st.session_state.chat_history = []  # Chat geçmişini temizle
            st.success("✅ Video başarıyla işlendi!")
            st.rerun()
    
    # Video bilgisi göster
    if st.session_state.video_processed and st.session_state.video_state:
        state = st.session_state.video_state
        
        # Video embed ve bilgiler
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Video embed
            video_id = extract_video_id_for_embed(state['video_url'])
            if video_id:
                st.markdown(f"""
                <div class="video-embed">
                    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
                        <iframe src="https://www.youtube.com/embed/{video_id}" 
                                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
                                frameborder="0" allowfullscreen>
                        </iframe>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Video önizlemesi mevcut değil")
        
        with col2:
            st.markdown(f"""
            <div class="video-info">
                <h3>📊 Video Bilgileri</h3>
                <p><strong>📹 Başlık:</strong> {state['video_title']}</p>
                <p><strong>🔗 URL:</strong> <a href="{state['video_url']}" target="_blank">YouTube'da Aç</a></p>
                <p><strong>📝 Transcript:</strong> {len(state['transcript'])} karakter</p>
                <p><strong>🧩 Parça Sayısı:</strong> {len(state['chunks'])}</p>
                <p><strong>🎯 Ana Fikir:</strong> {len(state.get('key_insights', []))} adet</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Ana Fikirler Bölümü
        if state.get('key_insights'):
            st.markdown("### 🎯 Video'nun Ana Fikirleri")
            
            # Ana fikirleri güzel bir şekilde göster
            for i, insight in enumerate(state['key_insights'], 1):
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 1rem;
                    border-radius: 10px;
                    margin: 0.5rem 0;
                    border-left: 4px solid #4285f4;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                ">
                    <strong>💡 Ana Fikir {i}:</strong> {insight}
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("🎯 Ana fikirler çıkarılıyor...")
        
        # Transcript detayları - expandable
        with st.expander("📋 Transcript Detayları", expanded=False):
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Özet", "📝 Tam Metin", "🧩 Parçalar", "🎯 Ana Fikirler"])
            
            with tab1:
                st.info(f"""
                **📈 İstatistikler:**
                - **Toplam Karakter:** {len(state['transcript']):,}
                - **Toplam Kelime:** {len(state['transcript'].split()):,}
                - **Parça Sayısı:** {len(state['chunks'])}
                - **Ortalama Parça Uzunluğu:** {len(state['transcript']) // len(state['chunks']) if state['chunks'] else 0} karakter
                """)
            
            with tab2:
                st.text_area(
                    "Tam Transcript",
                    value=state['transcript'],
                    height=300,
                    help="Video'nun tam transcript metni",
                    key="full_transcript"
                )
            
            with tab3:
                st.write(f"**Toplam {len(state['chunks'])} parça:**")
                for i, chunk in enumerate(state['chunks'], 1):
                    with st.expander(f"Parça {i} ({len(chunk.page_content)} karakter)", expanded=False):
                        st.text_area(
                            f"Parça {i} İçeriği",
                            value=chunk.page_content,
                            height=150,
                            disabled=True,
                            key=f"transcript_chunk_{i}"
                        )
                        st.caption(f"Metadata: {chunk.metadata}")
            
            with tab4:
                if state.get('key_insights'):
                    st.write("**🎯 Video'nun Ana Fikirleri:**")
                    for i, insight in enumerate(state['key_insights'], 1):
                        st.markdown(f"""
                        <div style="
                            background-color: #f8f9fa;
                            border-left: 4px solid #667eea;
                            padding: 1rem;
                            margin: 0.5rem 0;
                            border-radius: 5px;
                        ">
                            <strong>{i}.</strong> {insight}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Ana fikirleri kopyalama butonu
                    insights_text = "\n".join([f"{i}. {insight}" for i, insight in enumerate(state['key_insights'], 1)])
                    st.text_area(
                        "Ana Fikirler (Kopyalamak için)",
                        value=insights_text,
                        height=150,
                        key="insights_copy"
                    )
                else:
                    st.info("Ana fikirler henüz çıkarılmadı.")
    
    # Soru-Cevap bölümü
    if st.session_state.video_processed:
        st.markdown("## 💬 Soru-Cevap")
        
        # Chat geçmişini göster
        chat_container = st.container()
        
        with chat_container:
            for i, chat_item in enumerate(st.session_state.chat_history):
                if isinstance(chat_item, tuple) and len(chat_item) == 2:
                    # Eski format (sadece soru-cevap)
                    question, answer = chat_item
                    relevant_chunks = None
                else:
                    # Yeni format (soru-cevap-chunks)
                    question = chat_item.get('question', '')
                    answer = chat_item.get('answer', '')
                    relevant_chunks = chat_item.get('relevant_chunks', None)
                
                # Soru
                st.markdown(f"""
                <div class="chat-message question">
                    <strong>🤔 Soru {i+1}:</strong> {question}
                </div>
                """, unsafe_allow_html=True)
                
                # Cevap
                st.markdown(f"""
                <div class="chat-message answer">
                    <strong>🤖 Cevap:</strong> {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Kullanılan parçaları göster
                if relevant_chunks:
                    with st.expander(f"📄 Cevap için kullanılan içerik parçaları ({len(relevant_chunks)} parça)", expanded=False):
                        for j, chunk in enumerate(relevant_chunks, 1):
                            st.text_area(
                                f"Parça {j}",
                                value=chunk,
                                height=100,
                                disabled=True,
                                key=f"chunk_q{i}_p{j}"
                            )
        
        # Yeni soru formu
        with st.form("question_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                question = st.text_input(
                    "Sorunuz",
                    placeholder="Video hakkında sormak istediğiniz soruyu yazın...",
                    label_visibility="collapsed"
                )
            
            with col2:
                submit_button = st.form_submit_button("🚀 Sor", type="primary")
        
        if submit_button and question and st.session_state.agent and st.session_state.video_state:
            updated_state = ask_question(st.session_state.agent, st.session_state.video_state, question)
            if updated_state:
                st.session_state.video_state = updated_state
                # Chat geçmişine soru, cevap ve kullanılan parçaları ekle
                chat_item = {
                    'question': question,
                    'answer': updated_state["answer"],
                    'relevant_chunks': updated_state.get("relevant_chunks", [])
                }
                st.session_state.chat_history.append(chat_item)
                st.rerun()
    
    else:
        st.info("📹 Soru sorabilmek için önce bir video işleyin.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🤖 YouTube QA Agent - Video içeriklerini analiz eden AI asistanınız</p>
        <p>LM Studio ve Google Gemini desteği ile güçlendirilmiştir</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
