"""
YouTube Video Soru-Cevap Agent Sistemi
Bu sistem YouTube video linkinden transcript çıkarıp, içerik hakkında soru-cevap yapar.
"""

import os
import re
from typing import List, Dict, Any, Optional, TypedDict
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import tiktoken

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import google.generativeai as genai

from langgraph.graph import StateGraph, END

import httpx

class LMStudioEmbeddings(Embeddings):
    """LM Studio için özel embedding sınıfı"""
    
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
        self.client = httpx.Client(timeout=60.0)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Metinleri embedding'e çevirir"""
        embeddings = []
        for text in texts:
            try:
                response = self.client.post(
                    f"{self.base_url}/embeddings",
                    json={
                        "model": self.model_name,
                        "input": text
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    embedding = data["data"][0]["embedding"]
                    embeddings.append(embedding)
                else:
                    print(f"❌ Embedding hatası: {response.status_code}")
                    # Hata durumunda dummy embedding
                    embeddings.append([0.0] * 1024)  # Varsayılan boyut
            except Exception as e:
                print(f"❌ Embedding isteği hatası: {e}")
                embeddings.append([0.0] * 1024)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Tek bir metni embedding'e çevirir"""
        return self.embed_documents([text])[0]

class VideoQAState(TypedDict):
    """Agent state yapısı"""
    video_url: str
    video_title: str
    transcript: str
    chunks: List[Document]
    vectorstore: Optional[Any]
    question: str
    relevant_chunks: List[str]
    answer: str
    conversation_history: List[Dict[str, str]]
    key_insights: List[str]  # Ana fikirler
    error_message: str

class YouTubeQAAgent:
    """YouTube Video Soru-Cevap Agent Sistemi"""
    
    def __init__(self, 
                 api_key: str = None,
                 provider: str = "lm_studio",  # "lm_studio" veya "gemini"
                 lm_studio_url: str = "http://localhost:1234/v1",
                 model_name: str = "gemma-3n-e4b",
                 embedding_model: str = "text-embedding-mxbai-embed-large-v1"):
        """
        Args:
            api_key: Gemini API key (sadece Gemini modu için gerekli)
            provider: "lm_studio" veya "gemini"
            lm_studio_url: LM Studio API endpoint'i
            model_name: Kullanılacak LLM model adı
            embedding_model: Kullanılacak embedding model adı
        """
        self.provider = provider
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        if provider == "lm_studio":
            # LM Studio için yapılandırma
            print("🔵 LM Studio modu etkinleştirildi")
            print(f"🔗 Endpoint: {lm_studio_url}")
            print(f"🤖 LLM Model: {model_name}")
            print(f"🧠 Embedding Model: {embedding_model}")
            
            # LM Studio için ChatOpenAI benzeri client
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                base_url=lm_studio_url,
                api_key="lm-studio",  # LM Studio için dummy key
                model=model_name,
                temperature=0.1,
                max_tokens=2000
            )
            
            # LM Studio embeddings kullan
            self.embeddings = LMStudioEmbeddings(lm_studio_url, embedding_model)
            print("✅ LM Studio embeddings yapılandırıldı")
            
        elif provider == "gemini":
            # Gemini için yapılandırma
            if not api_key:
                raise ValueError("Gemini API key gerekli")
            
            print("🔵 Google Gemini modu etkinleştirildi")
            print(f"🤖 Model: {model_name}")
            
            # Gemini API key'i ayarla
            genai.configure(api_key=api_key)
            os.environ["GOOGLE_API_KEY"] = api_key
            
            # Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model=model_name if model_name.startswith("gemini") else "gemini-2.5-flash",
                temperature=0.1,
                max_tokens=2000
            )
            
            # Gemini embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
            print("✅ Gemini LLM ve embeddings yapılandırıldı")
            
        else:
            raise ValueError(f"Desteklenmeyen provider: {provider}")
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Token sayacı
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        
        # Graph'ı oluştur
        self.graph = self._build_graph()
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """YouTube URL'den video ID'sini çıkarır"""
        try:
            parsed_url = urlparse(url)
            
            if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
                if parsed_url.path == '/watch':
                    return parse_qs(parsed_url.query).get('v', [None])[0]
                elif parsed_url.path.startswith('/embed/'):
                    return parsed_url.path.split('/embed/')[1]
            elif parsed_url.hostname in ['youtu.be']:
                return parsed_url.path[1:]
            
            return None
        except Exception as e:
            print(f"❌ URL parse hatası: {e}")
            return None
    
    def get_video_info(self, state: VideoQAState) -> VideoQAState:
        """Video bilgilerini alır"""
        try:
            print("🔵 Video bilgileri alınıyor...")
            
            video_id = self.extract_video_id(state["video_url"])
            if not video_id:
                state["error_message"] = "Geçersiz YouTube URL"
                return state
            
            # Video başlığını al - farklı yöntemler dene
            try:
                # İlk yöntem: pytube
                yt = YouTube(state["video_url"])
                state["video_title"] = yt.title
                print(f"✅ Video bulundu (pytube): {state['video_title']}")
            except Exception as e1:
                print(f"⚠️ Pytube hatası: {e1}")
                # Son çare: sadece video ID
                state["video_title"] = f"YouTube Video ({video_id})"
                print(f"⚠️ Video ID kullanılıyor: {state['video_title']}")
            
        except Exception as e:
            state["error_message"] = f"Video bilgisi alınamadı: {str(e)}"
            print(f"❌ Hata: {state['error_message']}")
        
        return state
    
    def extract_transcript(self, state: VideoQAState) -> VideoQAState:
        """YouTube video transcript'ini çıkarır"""
        try:
            print("🔵 Video transcript'i çıkarılıyor...")
            
            video_id = self.extract_video_id(state["video_url"])
            if not video_id:
                state["error_message"] = "Video ID çıkarılamadı"
                return state
            
            # Yeni API kullanımı
            ytt_api = YouTubeTranscriptApi()
            fetched_transcript = None
            
            # Farklı dilleri dene
            languages_to_try = ['tr', 'en']
            
            for lang in languages_to_try:
                try:
                    print(f"📝Dil deneniyor: {lang}")
                    fetched_transcript = ytt_api.fetch(video_id, languages=[lang])
                    print(f"✅Transcript bulundu ({lang})")
                    break
                except Exception as lang_error:
                    print(f"📝{lang} dili başarısız: {lang_error}")
                    continue
            
            # Eğer hiç dil bulunamazsa, mevcut transcript'leri listele ve ilkini al
            if not fetched_transcript:
                try:
                    print("⚠️Mevcut transcript'ler listeleniyor...")
                    transcript_list = ytt_api.list(video_id)
                    
                    # İlk mevcut transcript'i al
                    for transcript in transcript_list:
                        try:
                            fetched_transcript = transcript.fetch()
                            print(f"✅Transcript alındı: {transcript.language}")
                            break
                        except Exception as e:
                            print(f"📝Transcript fetch hatası: {e}")
                            continue
                            
                except Exception as list_error:
                    print(f"[red]Transcript listelenemedi: {list_error}[/red]")
            
            if not fetched_transcript:
                state["error_message"] = "Video için transcript bulunamadı"
                return state
            
            # Raw data'yı al ve birleştir
            raw_data = fetched_transcript.to_raw_data()
            full_transcript = " ".join([item['text'] for item in raw_data])
            state["transcript"] = full_transcript
            
            # Token sayısını göster
            token_count = len(self.encoding.encode(full_transcript))
            print(f"✅Transcript çıkarıldı: {len(full_transcript)} karakter, {token_count} token")
            
        except Exception as e:
            state["error_message"] = f"Transcript çıkarılamadı: {str(e)}"
            print(f"❌ Hata: {state['error_message']}")
        
        return state
    
    def process_content(self, state: VideoQAState) -> VideoQAState:
        """İçeriği parçalara ayırır ve vector store oluşturur"""
        try:
            print("🔵İçerik işleniyor ve vector store oluşturuluyor...")
            
            if not state["transcript"]:
                state["error_message"] = "İşlenecek transcript bulunamadı"
                return state
            
            # Metni parçalara ayır
            chunks = self.text_splitter.split_text(state["transcript"])
            
            # Document objelerine çevir
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "video_title": state["video_title"],
                        "video_url": state["video_url"],
                        "chunk_id": i,
                        "source": "youtube_transcript"
                    }
                )
                documents.append(doc)
            
            state["chunks"] = documents
            
            # Vector store oluştur (embeddings varsa)
            if self.embeddings:
                vectorstore = FAISS.from_documents(documents, self.embeddings)
                state["vectorstore"] = vectorstore
                print(f"✅İçerik işlendi: {len(documents)} parça oluşturuldu")
            else:
                print("⚠️⚠️  Embeddings olmadan devam ediliyor (basit metin araması)")
                state["vectorstore"] = None
                print(f"✅İçerik işlendi: {len(documents)} parça oluşturuldu")
            
        except Exception as e:
            state["error_message"] = f"İçerik işleme hatası: {str(e)}"
            print(f"❌ Hata: {state['error_message']}")
        
        return state
    
    def search_relevant_content(self, state: VideoQAState) -> VideoQAState:
        """Soruya uygun içerikleri arar"""
        try:
            if not state["question"]:
                state["error_message"] = "Soru bulunamadı"
                return state
            
            print(f"🔵Soru için ilgili içerikler aranıyor: {state['question']}")
            
            if state["vectorstore"]:
                # Vector search ile alakalı içerikleri bul
                docs = state["vectorstore"].similarity_search(
                    state["question"], 
                    k=4  # En alakalı 4 parçayı al
                )
                relevant_chunks = [doc.page_content for doc in docs]
                print(f"✅Vector search ile {len(relevant_chunks)} alakalı içerik bulundu")
            else:
                # Basit keyword search (embeddings yoksa)
                print("⚠️Basit metin araması yapılıyor...")
                question_words = state["question"].lower().split()
                relevant_chunks = []
                
                for doc in state["chunks"]:
                    content_lower = doc.page_content.lower()
                    # Soru kelimelerinin herhangi biri içerikte varsa al
                    if any(word in content_lower for word in question_words if len(word) > 2):
                        relevant_chunks.append(doc.page_content)
                
                # En fazla 4 parça al
                relevant_chunks = relevant_chunks[:4]
                
                # Eğer hiç bulamazsa, ilk birkaç parçayı al
                if not relevant_chunks and state["chunks"]:
                    relevant_chunks = [doc.page_content for doc in state["chunks"][:3]]
                
                print(f"✅Basit arama ile {len(relevant_chunks)} alakalı içerik bulundu")
            
            state["relevant_chunks"] = relevant_chunks
            
        except Exception as e:
            state["error_message"] = f"İçerik arama hatası: {str(e)}"
            print(f"❌ Hata: {state['error_message']}")
        
        return state
    
    def extract_key_insights(self, state: VideoQAState) -> VideoQAState:
        """Video'nun ana fikirlerini çıkarır"""
        try:
            print("🔵Ana fikirler çıkarılıyor...")
            
            if not state["transcript"]:
                state["error_message"] = "Ana fikirler için transcript bulunamadı"
                return state
            
            # Transcript'in ilk yarısını al (çok uzunsa)
            transcript = state["transcript"]
            if len(transcript) > 3000:
                # İlk 3000 karakteri al ve son cümleyi tamamla
                truncated = transcript[:3000]
                last_period = truncated.rfind('.')
                if last_period > 2000:
                    transcript = truncated[:last_period + 1]
                else:
                    transcript = truncated
            
            # Ana fikirler için prompt
            insights_prompt = f"""Sen bir video içerik analiz uzmanısın. Aşağıdaki video transcript'ini analiz ederek ana fikirleri çıkar.

Video: {state["video_title"]}

Transcript:
{transcript}

Görev: Bu video'nun 3-5 temel mesajını/ana fikrini çıkar.

Format:
- Her ana fikir tek satırda olsun
- Net, öz ve anlaşılır olsun
- Türkçe olsun
- Sadece ana fikirleri listele, açıklama yapma

Ana Fikirler:
1.
2.
3.
4.
5."""
            
            response = self.llm.invoke(insights_prompt)
            insights_text = response.content
            
            # Response'u parse et
            insights = []
            lines = insights_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')) or line[0].isdigit()):
                    # Numarayı ve işaretleri temizle
                    clean_line = line
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '•']:
                        if clean_line.startswith(prefix):
                            clean_line = clean_line[len(prefix):].strip()
                            break
                    if clean_line:
                        insights.append(clean_line)
            
            # En fazla 5 ana fikir
            state["key_insights"] = insights[:5]
            
            print(f"✅✅ {len(state['key_insights'])} ana fikir çıkarıldı")
            
        except Exception as e:
            state["error_message"] = f"Ana fikir çıkarma hatası: {str(e)}"
            print(f"❌ Hata: {state['error_message']}")
            # Hata durumunda boş liste
            state["key_insights"] = []
        
        return state
    
    def generate_answer(self, state: VideoQAState) -> VideoQAState:
        """Soruya cevap üretir"""
        try:
            if not state["relevant_chunks"] or not state["question"]:
                state["error_message"] = "Alakalı içerik veya soru bulunamadı"
                return state
            
            print("🔵Cevap üretiliyor...")
            
            # Alakalı içerikleri birleştir
            context = "\n\n".join(state["relevant_chunks"])
            
            # Gemma için optimize edilmiş prompt
            prompt = f"""Sen bir video içerik analiz uzmanısın. Aşağıdaki video içeriğini analiz ederek soruyu yanıtla.

Video: {state["video_title"]}

İçerik:
{context}

Soru: {state["question"]}

Talimatlar:
1. Sadece verilen video içeriğindeki bilgileri kullan
2. Cevabını Türkçe ver
3. Açık ve net ol
4. Eğer bilgi içerikte yoksa "Bu bilgi video içeriğinde mevcut değil" de

Cevap:"""
            
            response = self.llm.invoke(prompt)
            state["answer"] = response.content
            
            # Konuşma geçmişine ekle
            if "conversation_history" not in state:
                state["conversation_history"] = []
            
            state["conversation_history"].append({
                "question": state["question"],
                "answer": state["answer"]
            })
            
            print("✅Cevap üretildi")
            
        except Exception as e:
            state["error_message"] = f"Cevap üretme hatası: {str(e)}"
            print(f"❌ Hata: {state['error_message']}")
        
        return state
    
    def _build_graph(self) -> StateGraph:
        """LangGraph workflow'unu oluşturur"""
        workflow = StateGraph(VideoQAState)
        
        # Node'ları ekle
        workflow.add_node("get_video_info", self.get_video_info)
        workflow.add_node("extract_transcript", self.extract_transcript)
        workflow.add_node("process_content", self.process_content)
        workflow.add_node("extract_insights", self.extract_key_insights)
        workflow.add_node("search_content", self.search_relevant_content)
        workflow.add_node("generate_answer", self.generate_answer)
        
        # Edge'leri tanımla
        workflow.add_edge("get_video_info", "extract_transcript")
        workflow.add_edge("extract_transcript", "process_content")
        workflow.add_edge("process_content", "extract_insights")
        workflow.add_edge("extract_insights", "search_content")
        workflow.add_edge("search_content", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        # Başlangıç noktası
        workflow.set_entry_point("get_video_info")
        
        return workflow.compile()
    
    def process_video(self, video_url: str) -> VideoQAState:
        """Video'yu işler ve soru-cevap için hazırlar"""
        initial_state = VideoQAState(
            video_url=video_url,
            video_title="",
            transcript="",
            chunks=[],
            vectorstore=None,
            question="",
            relevant_chunks=[],
            answer="",
            conversation_history=[],
            key_insights=[],
            error_message=""
        )
        
        # Video işleme workflow'unu çalıştır (ana fikirler dahil)
        for step in ["get_video_info", "extract_transcript", "process_content", "extract_insights"]:
            if step == "get_video_info":
                result = self.get_video_info(initial_state)
            elif step == "extract_transcript":
                result = self.extract_transcript(result)
            elif step == "process_content":
                result = self.process_content(result)
            elif step == "extract_insights":
                result = self.extract_key_insights(result)
            
            if result["error_message"]:
                return result
        
        return result
    
    def ask_question(self, state: VideoQAState, question: str) -> VideoQAState:
        """Video hakkında soru sorar"""
        state["question"] = question
        
        # Soru-cevap workflow'unu çalıştır
        state = self.search_relevant_content(state)
        if state["error_message"]:
            return state
        
        state = self.generate_answer(state)
        return state
    
    def display_answer(self, state: VideoQAState):
        """Cevabı güzel bir şekilde gösterir"""
        if state["error_message"]:
            print(f"❌ Hata: {state['error_message']}")
            return
        
        # Video bilgisi
        print(f"\n🎥 Video: {state['video_title']}")
        print(f"🔗 URL: {state['video_url']}")
        
        # Soru
        print(f"\n❓ Soru: {state['question']}")
        
        # Cevap
        print(f"\n💬 Cevap: {state['answer']}")
        print("-" * 50)

def main():
    """Ana fonksiyon"""
    print("🤖 YouTube Video Soru-Cevap Agent Sistemi")
    print("=" * 50)
    print("Bu sistem YouTube videolarından transcript çıkarıp, içerik hakkında sorularınızı yanıtlar.")
    print("=" * 50)
    
    # Model seçimi
    print("\n🔧 Model seçenekleri:")
    print("1. LM Studio (Yerel model)")
    print("2. Google Gemini (Bulut model)")
    
    choice = input("\nSeçiminiz (1 veya 2): ").strip()
    
    if choice == "1":
        # LM Studio modu
        lm_studio_url = input("LM Studio URL'si [http://localhost:1234/v1]: ").strip()
        if not lm_studio_url:
            lm_studio_url = "http://localhost:1234/v1"
        
        model_name = input("Model adı [gemma-3n-e4b]: ").strip()
        if not model_name:
            model_name = "gemma-3n-e4b"
        
        # Embedding model adı
        embedding_model = input("Embedding model adı [text-embedding-mxbai-embed-large-v1]: ").strip()
        if not embedding_model:
            embedding_model = "text-embedding-mxbai-embed-large-v1"
        
        print(f"✅ LM Studio modu: {lm_studio_url}")
        print(f"✅ LLM Model: {model_name}")
        print(f"✅ Embedding Model: {embedding_model}")
        
        agent = YouTubeQAAgent(
            provider="lm_studio",
            lm_studio_url=lm_studio_url,
            model_name=model_name,
            embedding_model=embedding_model
        )
    else:
        # Gemini modu
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            api_key = input("Google Gemini API Key giriniz: ")
        
        # Gemini model seçimi
        print("\n🔧 Gemini model seçenekleri:")
        print("1. gemini-2.5-flash (Hızlı)")
        print("2. gemini-2.5-pro (Güçlü)")
        
        model_choice = input("Model seçin (1 veya 2) [1]: ").strip()
        if model_choice == "2":
            model_name = "gemini-2.5-pro"
        else:
            model_name = "gemini-2.5-flash"
        
        print(f"✅ Gemini modu")
        print(f"✅ Model: {model_name}")
        
        agent = YouTubeQAAgent(
            api_key=api_key,
            provider="gemini",
            model_name=model_name
        )
    
    # Video URL al
    video_url = input("\nYouTube video URL'sini giriniz: ")
    
    print("\n🔵 Video işleniyor...")
    
    # Video'yu işle
    state = agent.process_video(video_url)
    
    if state["error_message"]:
        print(f"❌ Hata: {state['error_message']}")
        return
    
    print(f"\n✅ Video başarıyla işlendi: {state['video_title']}")
    print("💬 Artık video hakkında sorular sorabilirsiniz!")
    
    # Soru-cevap döngüsü
    while True:
        try:
            question = input("\n🤔 Sorunuz (çıkmak için 'q'): ")
            
            if question.lower() in ['q', 'quit', 'exit', 'çık']:
                print("👋 Görüşürüz!")
                break
            
            if not question.strip():
                continue
            
            # Soruyu yanıtla
            state = agent.ask_question(state, question)
            agent.display_answer(state)
            
        except KeyboardInterrupt:
            print("\n👋 Görüşürüz!")
            break
        except Exception as e:
            print(f"❌ Beklenmeyen hata: {e}")

if __name__ == "__main__":
    main()
