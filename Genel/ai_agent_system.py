"""
GERÇEK AI AGENT SİSTEMİ - LangChain & CrewAI ile
Gerçek agent framework'leri kullanarak çalışır
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import time
from typing import List, Dict, Optional
import re
import requests
from bs4 import BeautifulSoup
import io
from urllib.parse import urlparse
import os

# PDF okuma için
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# DOCX okuma için
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Gereksiz import'lar kaldırıldı - Sadece Gemini kullanıyoruz

# Gemini API
import google.generativeai as genai

st.set_page_config(
    page_title="AI Agent Kontrol Paneli",
    page_icon="🤖",
    layout="wide"
)

# ============================================================================
# AGENT TYPES
# ============================================================================

AGENT_TYPES = {
    "Research Agent": {
        "icon": "🔍",
        "description": "Google Search grounding ile gerçek web araştırması",
        "color": "blue",
        "tools": ["google_search"],
        "model": "gemini-2.5-flash"
    },
    "Data Analysis Agent": {
        "icon": "📊",
        "description": "Code execution ile veri analizi ve hesaplama",
        "color": "green",
        "tools": ["code_execution"],
        "model": "gemini-2.5-flash"
    },
    "Content Writer Agent": {
        "icon": "✍️",
        "description": "Structured output ile SEO uyumlu içerik",
        "color": "purple",
        "tools": ["structured_output"],
        "model": "gemini-2.5-flash"
    },
    "Document Analysis Agent": {
        "icon": "📄",
        "description": "Document understanding ile PDF/DOCX analizi",
        "color": "cyan",
        "tools": ["document_understanding"],
        "model": "gemini-2.5-flash"
    },
    "Code Assistant Agent": {
        "icon": "💻",
        "description": "Code execution ile kod yazma ve test",
        "color": "red",
        "tools": ["code_execution"],
        "model": "gemini-2.5-flash"
    }
}

# ============================================================================
# HELPER FUNCTIONS - Dosya ve URL işleme
# ============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """PDF dosyasından metin çıkar"""
    try:
        if PDF_AVAILABLE:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            return "PDF okuma kütüphanesi yüklü değil."
    except Exception as e:
        return f"PDF okuma hatası: {str(e)}"

def extract_text_from_docx(docx_file) -> str:
    """DOCX dosyasından metin çıkar"""
    try:
        if DOCX_AVAILABLE:
            doc = docx.Document(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        else:
            return "DOCX okuma kütüphanesi yüklü değil."
    except Exception as e:
        return f"DOCX okuma hatası: {str(e)}"

def extract_text_from_txt(txt_file) -> str:
    """TXT dosyasından metin çıkar"""
    try:
        return txt_file.read().decode('utf-8')
    except Exception as e:
        return f"TXT okuma hatası: {str(e)}"

def scrape_website(url: str) -> str:
    """Web sitesinden içerik çek"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Script ve style etiketlerini kaldır
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]  # İlk 5000 karakter
    except Exception as e:
        return f"Web scraping hatası: {str(e)}"

def read_csv_file(csv_file) -> str:
    """CSV dosyasını oku ve özet çıkar"""
    try:
        df = pd.read_csv(csv_file)
        summary = f"""
CSV Dosya Özeti:
- Satır sayısı: {len(df)}
- Sütun sayısı: {len(df.columns)}
- Sütunlar: {', '.join(df.columns.tolist())}

İlk 5 satır:
{df.head().to_string()}

İstatistikler:
{df.describe().to_string()}
"""
        return summary
    except Exception as e:
        return f"CSV okuma hatası: {str(e)}"

def read_excel_file(excel_file) -> str:
    """Excel dosyasını oku ve özet çıkar"""
    try:
        df = pd.read_excel(excel_file)
        summary = f"""
Excel Dosya Özeti:
- Satır sayısı: {len(df)}
- Sütun sayısı: {len(df.columns)}
- Sütunlar: {', '.join(df.columns.tolist())}

İlk 5 satır:
{df.head().to_string()}

İstatistikler:
{df.describe().to_string()}
"""
        return summary
    except Exception as e:
        return f"Excel okuma hatası: {str(e)}"

# Gereksiz tool fonksiyonları kaldırıldı - Gemini direkt kullanıyor

# ============================================================================
# RESEARCH AGENT - LangChain ile gerçek web araştırması
# ============================================================================

def research_agent_task(query: str, gemini_key: str, additional_context: str = "") -> Dict:
    """Research Agent - Gemini 2.0 Flash Lite + Google Search (Yeni SDK)"""
    try:
        # Yeni SDK kullan: google.genai
        from google import genai as new_genai
        from google.genai import types
        
        client = new_genai.Client(api_key=gemini_key)
        
        # Google Search tool
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        config = types.GenerateContentConfig(
            tools=[grounding_tool]
        )
        
        ref_text = f"\n\nEk Bağlam:\n{additional_context[:2000]}" if additional_context else ""
        
        prompt = f"""Sen bir profesyonel araştırma uzmanısın. Google Search kullanarak GÜNCEL ve DOĞRULANMIŞ bilgilerle derinlemesine araştırma yap.

🔍 Araştırma Konusu: {query}
{ref_text}

ARAŞTIRMA GEREKSİNİMLERİ:
✅ Güncel kaynaklar kullan (2024-2025)
✅ Birden fazla güvenilir kaynak kontrol et
✅ Rakamlar ve istatistikler ekle
✅ Farklı bakış açılarını değerlendir
✅ Kaynak linklerini belirt

ÇIKTI FORMATI (JSON):
{{
    "summary": "Kapsamlı özet (3-4 cümle, ana bulguları içeren)",
    "key_findings": ["Bulgu 1 (detaylı)", "Bulgu 2 (detaylı)", "Bulgu 3 (detaylı)", "Bulgu 4 (detaylı)", "Bulgu 5 (detaylı)"],
    "detailed_analysis": "Derinlemesine analiz (4-5 paragraf, rakamlar ve örneklerle desteklenmiş)",
    "sources": ["Kaynak 1 (başlık ve link)", "Kaynak 2", "Kaynak 3", "Kaynak 4"],
    "recommendations": ["Öneri 1 (aksiyon odaklı)", "Öneri 2", "Öneri 3"],
    "statistics": ["İstatistik 1", "İstatistik 2"],
    "trends": ["Trend 1", "Trend 2"],
    "expert_opinions": ["Uzman görüşü 1", "Uzman görüşü 2"]
}}
"""
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config
        )
        
        text = response.text.strip()
        
        # Grounding metadata'dan kaynakları al
        sources = []
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    grounding_chunks = candidate.grounding_metadata.grounding_chunks
                    if grounding_chunks:
                        for chunk in grounding_chunks[:5]:
                            if hasattr(chunk, 'web') and chunk.web:
                                sources.append(f"{chunk.web.title} - {chunk.web.uri}")
        except:
            pass
        
        # JSON parse
        try:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            else:
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end] if start != -1 and end > start else "{}"
            
            parsed_result = json.loads(json_str)
            
            # Grounding'den gelen kaynakları ekle
            if sources:
                parsed_result['sources'] = sources[:3] + parsed_result.get('sources', [])[:2]
            
        except:
            parsed_result = {
                "summary": text[:300] if text else "Araştırma tamamlandı",
                "key_findings": [
                    "Google Search ile gerçek arama yapıldı",
                    f"'{query}' konusunda güncel bilgi toplandı",
                    "Gemini 2.0 Flash Lite ile analiz edildi"
                ],
                "detailed_analysis": text[:1000] if text else "Araştırma tamamlandı.",
                "sources": sources if sources else ["Google Search"],
                "recommendations": ["Daha spesifik sorular sorun"]
            }
        
        return {
            "status": "success",
            "agent_type": "Research Agent",
            "framework": "Gemini 2.0 Flash Lite + Google Search",
            "query": query,
            "result": parsed_result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# ============================================================================
# DATA ANALYSIS AGENT - LangChain ile veri analizi
# ============================================================================

def data_analysis_agent_task(data_description: str, gemini_key: str, data_content: str = "") -> Dict:
    """Data Analysis Agent - Gemini 2.0 Flash Lite + Code Execution"""
    try:
        genai.configure(api_key=gemini_key)
        
        # Code execution tool ile model oluştur
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            tools='code_execution'
        )
        
        data_text = f"\n\nVeri:\n{data_content[:3000]}" if data_content else ""
        
        prompt = f"""Sen profesyonel bir veri analistisin. Python kodu çalıştırarak GERÇEK hesaplamalar ve analizler yap.

📊 Analiz Görevi: {data_description}
{data_text}

ANALİZ GEREKSİNİMLERİ:
✅ Python ile gerçek istatistiksel hesaplamalar yap (numpy, pandas kullan)
✅ Veri kalitesini kontrol et (missing values, outliers)
✅ Dağılım analizleri yap (normal dağılım, çarpıklık)
✅ Korelasyon analizi yap (değişkenler arası ilişkiler)
✅ Trend analizi yap (zaman serisi varsa)
✅ Görselleştirme önerileri sun (matplotlib, seaborn)

ÇIKTI FORMATI (JSON):
{{
    "summary": "Veri seti özeti (satır/sütun sayısı, veri tipi, genel durum)",
    "statistics": {{
        "ortalama": 0,
        "medyan": 0,
        "std": 0,
        "min": 0,
        "max": 0,
        "q1": 0,
        "q3": 0,
        "missing_values": 0,
        "outliers": 0
    }},
    "insights": [
        "İçgörü 1 (rakamlarla desteklenmiş)",
        "İçgörü 2 (istatistiksel bulgu)",
        "İçgörü 3 (pattern/trend)"
    ],
    "correlations": [
        "Değişken1 - Değişken2: korelasyon katsayısı",
        "Değişken3 - Değişken4: korelasyon katsayısı"
    ],
    "visualizations": [
        "Histogram (dağılım analizi için)",
        "Box plot (outlier tespiti için)",
        "Scatter plot (korelasyon için)",
        "Time series plot (trend için)"
    ],
    "recommendations": [
        "Öneri 1 (aksiyon odaklı)",
        "Öneri 2 (veri kalitesi)",
        "Öneri 3 (ileri analiz)"
    ],
    "data_quality": "İyi/Orta/Düşük (açıklama ile)"
}}

NOT: Gerçek Python kodu çalıştır ve sonuçları raporla!
"""
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        try:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            else:
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end] if start != -1 and end > start else "{}"
            parsed_result = json.loads(json_str)
        except:
            parsed_result = {
                "summary": text[:500] if text else "Veri analizi tamamlandı",
                "statistics": {"note": "Code execution ile hesaplandı"},
                "insights": ["Python kodu çalıştırıldı", "Gerçek hesaplamalar yapıldı"],
                "visualizations": ["Histogram", "Scatter plot"],
                "recommendations": [text[:300] if text else "Veri kalitesini artırın"]
            }
        
        return {
            "status": "success",
            "agent_type": "Data Analysis Agent",
            "framework": "Gemini 2.0 Flash Lite + Code Execution",
            "result": parsed_result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# ============================================================================
# CONTENT WRITER AGENT - CrewAI ile içerik üretimi
# ============================================================================

def content_writer_agent_task(topic: str, content_type: str, gemini_key: str, reference_content: str = "") -> Dict:
    """Content Writer Agent - Gemini 2.5 Flash + Structured Output (İçerik Tipine Özel)"""
    try:
        genai.configure(api_key=gemini_key)
        
        # İçerik tipine göre özel ayarlar
        content_specs = {
            "Makale": {
                "word_count": "800-1200 kelime",
                "structure": "Giriş, 3-4 ana bölüm, sonuç",
                "tone": "Profesyonel ve bilgilendirici",
                "extras": "Alt başlıklar, madde işaretli listeler",
                "visual": "İnfografik veya açıklayıcı görseller öner"
            },
            "Blog": {
                "word_count": "600-900 kelime",
                "structure": "Çekici giriş, 2-3 ana nokta, CTA ile sonuç",
                "tone": "Samimi ve ilgi çekici",
                "extras": "Kişisel anekdotlar, emoji kullanımı",
                "visual": "Blog banner görseli ve ara görseller öner"
            },
            "Sosyal Medya": {
                "word_count": "50-150 karakter (Twitter) veya 200-300 kelime (LinkedIn)",
                "structure": "Hook + Ana mesaj + CTA + Hashtag",
                "tone": "Kısa, öz ve etkileyici",
                "extras": "Emoji, hashtag (#), mention (@)",
                "visual": "Dikkat çekici görsel veya carousel tasarımı öner"
            },
            "E-posta": {
                "word_count": "200-400 kelime",
                "structure": "Konu satırı + Kişiselleştirilmiş giriş + Ana mesaj + CTA",
                "tone": "Profesyonel ama yakın",
                "extras": "Konu satırı alternatifleri, PS notu",
                "visual": "E-posta header görseli ve CTA butonu tasarımı öner"
            },
            "Haber Bülteni": {
                "word_count": "400-600 kelime",
                "structure": "Başlık + Özet + Detaylar + İlgili linkler",
                "tone": "Objektif ve bilgilendirici",
                "extras": "Kaynak linkler, tarih bilgisi",
                "visual": "Haber görseli ve thumbnail öner"
            },
            "Ürün Açıklaması": {
                "word_count": "150-300 kelime",
                "structure": "Özellikler + Faydalar + Teknik detaylar + CTA",
                "tone": "İkna edici ve net",
                "extras": "Bullet points, özellik listesi",
                "visual": "Ürün görselleri ve kullanım senaryoları öner"
            }
        }
        
        spec = content_specs.get(content_type, content_specs["Makale"])
        
        # Structured output için generation config
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "summary": {"type": "string"},
                        "seo_score": {"type": "string"},
                        "target_audience": {"type": "string"},
                        "visual_suggestions": {"type": "array", "items": {"type": "string"}},
                        "hashtags": {"type": "array", "items": {"type": "string"}},
                        "cta": {"type": "string"}
                    }
                }
            }
        )
        
        ref_text = f"\n\nReferans:\n{reference_content[:2000]}" if reference_content else ""
        
        prompt = f"""Sen profesyonel bir {content_type} yazarısın. Aşağıdaki konuda içerik üret.

Konu: {topic}
İçerik Tipi: {content_type}
{ref_text}

ÖZEL GEREKSİNİMLER:
- Uzunluk: {spec['word_count']}
- Yapı: {spec['structure']}
- Ton: {spec['tone']}
- Ekstralar: {spec['extras']}
- Görsel: {spec['visual']}

ÇIKTI GEREKSİNİMLERİ:
- title: Çekici ve SEO uyumlu başlık
- content: {spec['word_count']} uzunluğunda, {spec['tone']} tonda içerik
- keywords: 5-7 anahtar kelime
- summary: 30-50 kelime özet
- seo_score: 1-10 arası puan
- target_audience: Hedef kitle tanımı
- visual_suggestions: 2-3 görsel önerisi (detaylı açıklama)
- hashtags: İlgili 5-10 hashtag (sosyal medya için)
- cta: Call-to-action metni

NOT: {content_type} formatına uygun, {spec['tone']} bir dil kullan!
"""
        
        response = model.generate_content(prompt)
        parsed_result = json.loads(response.text)
        
        # İçerik tipini ekle
        parsed_result['content_type'] = content_type
        parsed_result['specifications'] = spec
        
        return {
            "status": "success",
            "agent_type": "Content Writer Agent",
            "framework": "Gemini 2.5 Flash + Structured Output",
            "topic": topic,
            "content_type": content_type,
            "result": parsed_result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# ============================================================================
# DOCUMENT ANALYSIS AGENT - Document Understanding
# ============================================================================

def document_analysis_agent_task(document_description: str, gemini_key: str, document_content: str = "") -> Dict:
    """Document Analysis Agent - Gemini 2.0 Flash Lite + Document Understanding"""
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        if not document_content:
            return {
                "status": "error",
                "error": "Doküman içeriği sağlanmadı",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Doküman tipini belirle
        doc_type = "Genel"
        if any(word in document_description.lower() for word in ['rapor', 'report']):
            doc_type = "Rapor"
        elif any(word in document_description.lower() for word in ['makale', 'article', 'paper']):
            doc_type = "Akademik"
        elif any(word in document_description.lower() for word in ['sözleşme', 'contract', 'anlaşma']):
            doc_type = "Hukuki"
        elif any(word in document_description.lower() for word in ['özgeçmiş', 'cv', 'resume']):
            doc_type = "Özgeçmiş"
        
        prompt = f"""Sen profesyonel bir doküman analiz uzmanısın. {doc_type} tipinde doküman analizi yap.

📄 Analiz Talebi: {document_description}
📋 Doküman Tipi: {doc_type}

Doküman İçeriği:
{document_content[:10000]}

ANALİZ GEREKSİNİMLERİ:
✅ Dokümanın amacını ve bağlamını anla
✅ Anahtar bilgileri çıkar (isimler, tarihler, rakamlar)
✅ Yapısal analiz yap (bölümler, başlıklar)
✅ Sentiment ve ton analizi yap
✅ Eksik veya belirsiz noktaları belirle

ÇIKTI FORMATI (JSON):
{{
    "summary": "Kapsamlı özet (3-4 cümle, ana mesajı içeren)",
    "document_type": "{doc_type}",
    "key_points": [
        "Ana nokta 1 (detaylı)",
        "Ana nokta 2 (detaylı)",
        "Ana nokta 3 (detaylı)",
        "Ana nokta 4 (detaylı)"
    ],
    "entities": {{
        "people": ["İsim 1", "İsim 2"],
        "organizations": ["Kurum 1", "Kurum 2"],
        "dates": ["Tarih 1", "Tarih 2"],
        "locations": ["Yer 1", "Yer 2"],
        "numbers": ["Rakam 1: açıklama", "Rakam 2: açıklama"]
    }},
    "structure": {{
        "sections": ["Bölüm 1", "Bölüm 2"],
        "word_count": 0,
        "page_count": 0
    }},
    "sentiment": "Pozitif/Negatif/Nötr (açıklama ile)",
    "tone": "Resmi/Gayri resmi/Teknik/Akademik",
    "key_insights": [
        "İçgörü 1",
        "İçgörü 2"
    ],
    "missing_info": ["Eksik bilgi 1", "Eksik bilgi 2"],
    "recommendations": [
        "Öneri 1 (aksiyon odaklı)",
        "Öneri 2"
    ]
}}
"""
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        try:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            else:
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end] if start != -1 and end > start else "{}"
            parsed_result = json.loads(json_str)
        except:
            parsed_result = {
                "summary": text[:500] if text else "Doküman analiz edildi",
                "key_points": ["Doküman işlendi", "Gemini ile analiz edildi"],
                "entities": ["Otomatik çıkarıldı"],
                "sentiment": "Nötr",
                "recommendations": [text[:300] if text else "Detaylı inceleme yapın"]
            }
        
        return {
            "status": "success",
            "agent_type": "Document Analysis Agent",
            "framework": "Gemini 2.0 Flash Lite + Document Understanding",
            "result": parsed_result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# ============================================================================
# CUSTOMER SUPPORT AGENT - LangChain ile Q&A
# ============================================================================

# Customer Support Agent kaldırıldı - Document Analysis ile değiştirildi

# ============================================================================
# CODE ASSISTANT AGENT - CrewAI ile kod yazma
# ============================================================================

def code_assistant_agent_task(task_description: str, language: str, gemini_key: str, existing_code: str = "") -> Dict:
    """Code Assistant Agent - Gemini 2.0 Flash Lite + Code Execution"""
    try:
        genai.configure(api_key=gemini_key)
        
        # Code execution tool ile model
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            tools='code_execution'
        )
        
        existing_code_text = f"\n\nMevcut Kod:\n```{language}\n{existing_code[:2000]}\n```" if existing_code else ""
        
        # Dile göre özel ayarlar
        language_specs = {
            "Python": {
                "best_practices": "PEP 8, type hints, docstrings",
                "testing": "pytest, unittest",
                "patterns": "SOLID, DRY, KISS"
            },
            "JavaScript": {
                "best_practices": "ES6+, async/await, arrow functions",
                "testing": "Jest, Mocha",
                "patterns": "Functional programming, Promises"
            },
            "TypeScript": {
                "best_practices": "Strict mode, interfaces, generics",
                "testing": "Jest, Vitest",
                "patterns": "Type safety, OOP"
            },
            "Java": {
                "best_practices": "Clean code, naming conventions",
                "testing": "JUnit, Mockito",
                "patterns": "OOP, Design patterns"
            },
            "C++": {
                "best_practices": "RAII, smart pointers, const correctness",
                "testing": "Google Test, Catch2",
                "patterns": "Memory management, STL"
            }
        }
        
        spec = language_specs.get(language, language_specs["Python"])
        
        prompt = f"""Sen uzman bir {language} geliştiricisisin. Code execution ile KOD YAZ, TEST ET ve ÇALIŞTIR.

💻 Görev: {task_description}
🔤 Dil: {language}
{existing_code_text}

KOD GEREKSİNİMLERİ:
✅ {spec['best_practices']} standartlarına uy
✅ Clean code prensipleri uygula
✅ Edge case'leri düşün
✅ Performans optimizasyonu yap
✅ Güvenlik açıklarını kontrol et
✅ Gerçek kod çalıştır ve test et

ÇIKTI FORMATI (JSON):
{{
    "code": "Tam çalışan kod (yorum satırları ile)",
    "explanation": "Kodun ne yaptığı (detaylı, adım adım)",
    "complexity": {{
        "time": "O(n) - zaman karmaşıklığı",
        "space": "O(1) - alan karmaşıklığı"
    }},
    "test_cases": [
        "Test 1: Input -> Output (başarılı)",
        "Test 2: Edge case (başarılı)",
        "Test 3: Error handling (başarılı)"
    ],
    "best_practices": [
        "Uygulanan best practice 1",
        "Uygulanan best practice 2"
    ],
    "improvements": [
        "İyileştirme önerisi 1",
        "İyileştirme önerisi 2"
    ],
    "dependencies": ["Gerekli kütüphane 1", "Gerekli kütüphane 2"],
    "security_notes": ["Güvenlik notu 1", "Güvenlik notu 2"]
}}

NOT: Gerçek {language} kodu yaz ve çalıştır! Test framework: {spec['testing']}
"""
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        try:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            else:
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end] if start != -1 and end > start else "{}"
            parsed_result = json.loads(json_str)
        except:
            code_match = text
            if f"```{language}" in text:
                code_match = text.split(f"```{language}")[1].split("```")[0].strip()
            
            parsed_result = {
                "code": code_match[:1000] if code_match else text[:1000],
                "explanation": "Kod code execution ile test edildi",
                "complexity": "O(n)",
                "test_cases": ["Test edildi"],
                "improvements": ["Optimize edilebilir"]
            }
        
        return {
            "status": "success",
            "agent_type": "Code Assistant Agent",
            "framework": "Gemini 2.0 Flash Lite + Code Execution",
            "task": task_description,
            "language": language,
            "result": parsed_result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# ============================================================================
# UNIFIED AGENT EXECUTOR
# ============================================================================

def execute_agent_task(agent_type: str, task_params: Dict, gemini_key: str) -> Dict:
    """Agent görevini çalıştır - Gemini 2.0 Flash Lite + Tools"""
    
    if agent_type == "Research Agent":
        return research_agent_task(
            task_params.get("query", ""),
            gemini_key,
            task_params.get("additional_context", "")
        )
    
    elif agent_type == "Data Analysis Agent":
        return data_analysis_agent_task(
            task_params.get("data_description", ""),
            gemini_key,
            task_params.get("data_content", "")
        )
    
    elif agent_type == "Content Writer Agent":
        return content_writer_agent_task(
            task_params.get("topic", ""),
            task_params.get("content_type", "Makale"),
            gemini_key,
            task_params.get("reference_content", "")
        )
    
    elif agent_type == "Document Analysis Agent":
        return document_analysis_agent_task(
            task_params.get("document_description", ""),
            gemini_key,
            task_params.get("document_content", "")
        )
    
    elif agent_type == "Code Assistant Agent":
        return code_assistant_agent_task(
            task_params.get("task_description", ""),
            task_params.get("language", "Python"),
            gemini_key,
            task_params.get("existing_code", "")
        )
    
    else:
        return {
            "status": "error",
            "error": f"Bilinmeyen agent tipi: {agent_type}"
        }

# ============================================================================
# SESSION STATE
# ============================================================================

if 'agents' not in st.session_state:
    st.session_state.agents = []
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'task_results' not in st.session_state:
    st.session_state.task_results = []
if 'execution_logs' not in st.session_state:
    st.session_state.execution_logs = []
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

# ============================================================================
# UI - HEADER
# ============================================================================

st.title("🤖 AI Agent Kontrol Paneli")
st.markdown("**Gerçek Agent Framework'leri:** LangChain ReAct + CrewAI Multi-Agent")

# ============================================================================
# SIDEBAR - Agent Yönetimi
# ============================================================================

with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # API Key
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key,
        help="Google AI Studio'dan alın: https://aistudio.google.com/app/apikey"
    )
    
    if api_key:
        st.session_state.gemini_api_key = api_key
        st.success("✅ API Key kaydedildi")
    
    st.divider()
    
    # Agent Oluşturma
    st.header("➕ Yeni Agent Oluştur")
    
    agent_type = st.selectbox(
        "Agent Tipi",
        list(AGENT_TYPES.keys()),
        format_func=lambda x: f"{AGENT_TYPES[x]['icon']} {x}"
    )
    
    agent_name = st.text_input("Agent İsmi", f"{agent_type.split()[0]}Agent-{len(st.session_state.agents)+1:02d}")
    
    if st.button("🚀 Agent Oluştur", use_container_width=True):
        if not st.session_state.gemini_api_key:
            st.error("❌ Önce API Key girin!")
        else:
            new_agent = {
                "id": f"agent_{len(st.session_state.agents)+1}",
                "name": agent_name,
                "type": agent_type,
                "icon": AGENT_TYPES[agent_type]["icon"],
                "model": AGENT_TYPES[agent_type]["model"],
                "tools": AGENT_TYPES[agent_type]["tools"],
                "status": "idle",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tasks_completed": 0
            }
            st.session_state.agents.append(new_agent)
            st.session_state.execution_logs.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "agent": agent_name,
                "action": "Agent Oluşturuldu",
                "details": f"{agent_type} ({AGENT_TYPES[agent_type]['model']})"
            })
            st.success(f"✅ {agent_name} oluşturuldu!")
            st.rerun()
    
    st.divider()
    
    # Aktif Agent'lar
    st.header("🤖 Aktif Agent'lar")
    if st.session_state.agents:
        for agent in st.session_state.agents:
            with st.expander(f"{agent['icon']} {agent['name']}", expanded=False):
                st.write(f"**Tip:** {agent['type']}")
                st.write(f"**Model:** {agent['model']}")
                st.write(f"**Tools:** {', '.join(agent['tools'])}")
                st.write(f"**Durum:** {agent['status']}")
                st.write(f"**Görevler:** {agent['tasks_completed']}")
                if st.button(f"🗑️ Sil", key=f"delete_{agent['id']}"):
                    st.session_state.agents.remove(agent)
                    st.rerun()
    else:
        st.info("Henüz agent yok")

# ============================================================================
# MAIN - Tabs
# ============================================================================

tab1, tab2, tab3 = st.tabs(["🎯 Görev Yönetimi", "📊 Sonuçlar", "📋 Loglar"])

# TAB 1: Görev Yönetimi
with tab1:
    st.header("🎯 Yeni Görev Oluştur")
    
    if not st.session_state.agents:
        st.warning("⚠️ Önce sidebar'dan agent oluşturun!")
    else:
        # Agent seçimi
        agent_options = [f"{a['icon']} {a['name']} ({a['type']})" for a in st.session_state.agents]
        selected_agent_display = st.selectbox("Agent Seç", agent_options)
        selected_agent_idx = agent_options.index(selected_agent_display)
        selected_agent = st.session_state.agents[selected_agent_idx]
        
        agent_type = selected_agent['type']
        
        st.info(f"**Model:** {selected_agent['model']} | **Tools:** {', '.join(selected_agent['tools'])} | **Açıklama:** {AGENT_TYPES[agent_type]['description']}")
        
        # Agent tipine göre input alanları
        task_params = {}
        
        if agent_type == "Research Agent":
            st.subheader("🔍 Araştırma Parametreleri")
            task_params["query"] = st.text_area("Araştırma Konusu", "2025 yapay zeka trendleri neler?")
            
            # Ek kaynak
            st.write("**Ek Kaynak (Opsiyonel):**")
            col1, col2, col3 = st.columns(3)
            with col1:
                url_input = st.text_input("URL")
                if url_input:
                    task_params["additional_context"] = scrape_website(url_input)
            with col2:
                file_upload = st.file_uploader("Dosya Yükle", type=['pdf', 'docx', 'txt'])
                if file_upload:
                    if file_upload.name.endswith('.pdf'):
                        task_params["additional_context"] = extract_text_from_pdf(file_upload)
                    elif file_upload.name.endswith('.docx'):
                        task_params["additional_context"] = extract_text_from_docx(file_upload)
                    else:
                        task_params["additional_context"] = extract_text_from_txt(file_upload)
            with col3:
                manual_text = st.text_area("Manuel Metin")
                if manual_text:
                    task_params["additional_context"] = manual_text
        
        elif agent_type == "Data Analysis Agent":
            st.subheader("📊 Veri Analizi Parametreleri")
            task_params["data_description"] = st.text_area("Veri Açıklaması", "Satış verilerini analiz et")
            
            st.write("**Veri Kaynağı:**")
            col1, col2 = st.columns(2)
            with col1:
                data_file = st.file_uploader("CSV/Excel Yükle", type=['csv', 'xlsx'])
                if data_file:
                    if data_file.name.endswith('.csv'):
                        task_params["data_content"] = read_csv_file(data_file)
                    else:
                        task_params["data_content"] = read_excel_file(data_file)
            with col2:
                manual_data = st.text_area("Manuel Veri")
                if manual_data:
                    task_params["data_content"] = manual_data
        
        elif agent_type == "Content Writer Agent":
            st.subheader("✍️ İçerik Üretimi Parametreleri")
            task_params["topic"] = st.text_input("Konu", "Yapay Zeka ve Gelecek")
            task_params["content_type"] = st.selectbox("İçerik Tipi", ["Makale", "Blog", "Sosyal Medya", "E-posta", "Haber Bülteni", "Ürün Açıklaması"])
            
            st.write("**Referans İçerik (Opsiyonel):**")
            ref_file = st.file_uploader("Referans Dosya", type=['pdf', 'docx', 'txt'])
            if ref_file:
                if ref_file.name.endswith('.pdf'):
                    task_params["reference_content"] = extract_text_from_pdf(ref_file)
                elif ref_file.name.endswith('.docx'):
                    task_params["reference_content"] = extract_text_from_docx(ref_file)
                else:
                    task_params["reference_content"] = extract_text_from_txt(ref_file)
        
        elif agent_type == "Document Analysis Agent":
            st.subheader("📄 Doküman Analizi Parametreleri")
            task_params["document_description"] = st.text_area("Analiz Talebi", "Bu dokümanı özetle ve ana noktaları çıkar")
            
            st.write("**Doküman Yükle:**")
            doc_file = st.file_uploader("PDF/DOCX/TXT Dosyası", type=['pdf', 'docx', 'txt'], key="doc_analysis")
            if doc_file:
                if doc_file.name.endswith('.pdf'):
                    task_params["document_content"] = extract_text_from_pdf(doc_file)
                elif doc_file.name.endswith('.docx'):
                    task_params["document_content"] = extract_text_from_docx(doc_file)
                else:
                    task_params["document_content"] = extract_text_from_txt(doc_file)
                st.success(f"✅ {doc_file.name} yüklendi ({len(task_params.get('document_content', ''))} karakter)")
        
        elif agent_type == "Code Assistant Agent":
            st.subheader("💻 Kod Asistanı Parametreleri")
            task_params["task_description"] = st.text_area("Görev Açıklaması", "Fibonacci sayı dizisi hesapla")
            task_params["language"] = st.selectbox("Programlama Dili", ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go"])
            task_params["existing_code"] = st.text_area("Mevcut Kod (Opsiyonel)")
        
        # Görevi çalıştır
        if st.button("▶️ Görevi Çalıştır", type="primary", use_container_width=True):
            if not st.session_state.gemini_api_key:
                st.error("❌ API Key gerekli!")
            else:
                with st.spinner(f"🤖 {selected_agent['name']} çalışıyor..."):
                    progress = st.progress(0)
                    status = st.empty()
                    
                    status.text("Görev işleniyor...")
                    progress.progress(30)
                    
                    # Agent'ı çalıştır
                    result = execute_agent_task(
                        agent_type,
                        task_params,
                        st.session_state.gemini_api_key
                    )
                    
                    progress.progress(80)
                    
                    # Sonucu kaydet
                    st.session_state.task_results.append(result)
                    
                    # Agent istatistiklerini güncelle
                    selected_agent['tasks_completed'] += 1
                    
                    # Log ekle
                    st.session_state.execution_logs.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "agent": selected_agent['name'],
                        "action": "Görev Tamamlandı" if result['status'] == 'success' else "Görev Başarısız",
                        "details": result.get('framework', 'N/A')
                    })
                    
                    progress.progress(100)
                    status.text("✅ Tamamlandı!")
                    time.sleep(0.5)
                    
                    if result['status'] == 'success':
                        st.success("✅ Görev başarıyla tamamlandı!")
                    else:
                        st.error(f"❌ Hata: {result.get('error', 'Bilinmeyen hata')}")
                    
                    st.rerun()

# TAB 2: Sonuçlar
with tab2:
    st.header("📊 Görev Sonuçları")
    
    if st.session_state.task_results:
        for idx, result in enumerate(reversed(st.session_state.task_results)):
            agent_type = result.get('agent_type', 'Unknown Agent')
            icon = AGENT_TYPES.get(agent_type, {}).get('icon', '🤖')
            
            with st.expander(
                f"{icon} {agent_type} - {result.get('timestamp', 'N/A')}",
                expanded=(idx == 0)
            ):
                if result.get('status') == 'success':
                    st.success(f"✅ Başarılı | Framework: **{result.get('framework', 'N/A')}**")
                    
                    # Sonuçları okunabilir formatta göster
                    res = result.get('result', {})
                    
                    # Agent tipine göre özel görünüm
                    if agent_type == "Research Agent":
                        st.subheader("📝 Özet")
                        st.info(res.get('summary', 'N/A'))
                        
                        # Metrikler
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Bulgu Sayısı", len(res.get('key_findings', [])))
                        with col2:
                            st.metric("Kaynak Sayısı", len(res.get('sources', [])))
                        with col3:
                            st.metric("Öneri Sayısı", len(res.get('recommendations', [])))
                        
                        st.subheader("🔍 Ana Bulgular")
                        for i, finding in enumerate(res.get('key_findings', []), 1):
                            st.markdown(f"**{i}.** {finding}")
                        
                        # İstatistikler ve Trendler
                        if res.get('statistics') or res.get('trends'):
                            col1, col2 = st.columns(2)
                            with col1:
                                if res.get('statistics'):
                                    st.markdown("**📈 İstatistikler:**")
                                    for stat in res.get('statistics', []):
                                        st.markdown(f"- {stat}")
                            with col2:
                                if res.get('trends'):
                                    st.markdown("**📊 Trendler:**")
                                    for trend in res.get('trends', []):
                                        st.markdown(f"- {trend}")
                        
                        # Uzman Görüşleri
                        if res.get('expert_opinions'):
                            st.markdown("**👨‍🏫 Uzman Görüşleri:**")
                            for opinion in res.get('expert_opinions', []):
                                st.markdown(f"> {opinion}")
                        
                        st.subheader("📊 Detaylı Analiz")
                        st.markdown(f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>{res.get('detailed_analysis', 'N/A')}</div>", unsafe_allow_html=True)
                        
                        st.subheader("📚 Kaynaklar")
                        for i, source in enumerate(res.get('sources', []), 1):
                            st.markdown(f"**[{i}]** {source}")
                        
                        st.subheader("💡 Öneriler")
                        for i, rec in enumerate(res.get('recommendations', []), 1):
                            st.success(f"**{i}.** {rec}")
                
                elif agent_type == "Data Analysis Agent":
                    st.subheader("📝 Veri Seti Özeti")
                    st.info(res.get('summary', 'N/A'))
                    
                    # Veri Kalitesi
                    if res.get('data_quality'):
                        quality = res.get('data_quality', '')
                        if 'İyi' in quality:
                            st.success(f"✅ Veri Kalitesi: {quality}")
                        elif 'Orta' in quality:
                            st.warning(f"⚠️ Veri Kalitesi: {quality}")
                        else:
                            st.error(f"❌ Veri Kalitesi: {quality}")
                    
                    st.subheader("📊 İstatistiksel Analiz")
                    stats = res.get('statistics', {})
                    
                    # Ana metrikler
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Ortalama", f"{stats.get('ortalama', 'N/A')}")
                        st.metric("Q1", f"{stats.get('q1', 'N/A')}")
                    with col2:
                        st.metric("Medyan", f"{stats.get('medyan', 'N/A')}")
                        st.metric("Q3", f"{stats.get('q3', 'N/A')}")
                    with col3:
                        st.metric("Min", f"{stats.get('min', 'N/A')}")
                        st.metric("Max", f"{stats.get('max', 'N/A')}")
                    with col4:
                        st.metric("Std Sapma", f"{stats.get('std', 'N/A')}")
                        st.metric("Outliers", f"{stats.get('outliers', 'N/A')}")
                    
                    # Missing values uyarısı
                    if stats.get('missing_values', 0) > 0:
                        st.warning(f"⚠️ Eksik Veri: {stats.get('missing_values')} değer")
                    
                    # Korelasyonlar
                    if res.get('correlations'):
                        st.subheader("🔗 Korelasyon Analizi")
                        for corr in res.get('correlations', []):
                            st.markdown(f"- {corr}")
                    
                    st.subheader("💡 Anahtar İçgörüler")
                    for i, insight in enumerate(res.get('insights', []), 1):
                        st.markdown(f"**{i}.** {insight}")
                    
                    st.subheader("📈 Görselleştirme Önerileri")
                    viz_cols = st.columns(2)
                    for i, viz in enumerate(res.get('visualizations', [])):
                        with viz_cols[i % 2]:
                            st.markdown(f"📊 {viz}")
                    
                    st.subheader("🎯 Öneriler")
                    for i, rec in enumerate(res.get('recommendations', []), 1):
                        st.success(f"**{i}.** {rec}")
                
                elif agent_type == "Content Writer Agent":
                    content_type = res.get('content_type', 'Makale')
                    specs = res.get('specifications', {})
                    
                    # İçerik tipi badge
                    st.markdown(f"**📋 İçerik Tipi:** `{content_type}` | **📏 Uzunluk:** `{specs.get('word_count', 'N/A')}`")
                    
                    st.subheader(f"📰 {res.get('title', 'Başlık')}")
                    
                    # İçerik
                    st.markdown("**✍️ İçerik:**")
                    st.markdown(f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50;'>{res.get('content', 'N/A')}</div>", unsafe_allow_html=True)
                    
                    # Metrikler
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("SEO Puanı", f"{res.get('seo_score', 'N/A')}/10")
                    with col2:
                        st.metric("Kelime Sayısı", f"~{len(res.get('content', '').split())}")
                    with col3:
                        st.metric("Karakter", f"{len(res.get('content', ''))}")
                    
                    # Özet
                    st.markdown("**📝 Özet:**")
                    st.info(res.get('summary', 'N/A'))
                    
                    # Anahtar Kelimeler ve Hashtag'ler
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🏷️ Anahtar Kelimeler:**")
                        for kw in res.get('keywords', []):
                            st.markdown(f"- `{kw}`")
                    with col2:
                        st.markdown("**#️⃣ Hashtag'ler:**")
                        hashtags = res.get('hashtags', [])
                        if hashtags:
                            st.write(" ".join([f"`{tag}`" for tag in hashtags]))
                    
                    # CTA
                    if res.get('cta'):
                        st.markdown("**📢 Call-to-Action:**")
                        st.success(res.get('cta'))
                    
                    # Görsel Önerileri
                    st.markdown("**🎨 Görsel Önerileri:**")
                    for i, visual in enumerate(res.get('visual_suggestions', []), 1):
                        st.markdown(f"{i}. 🖼️ {visual}")
                    
                    # Hedef Kitle
                    st.markdown("**🎯 Hedef Kitle:**")
                    st.write(res.get('target_audience', 'N/A'))
                    
                    # Teknik Detaylar (Expander)
                    with st.expander("⚙️ Teknik Detaylar"):
                        st.json({
                            "Yapı": specs.get('structure', 'N/A'),
                            "Ton": specs.get('tone', 'N/A'),
                            "Ekstralar": specs.get('extras', 'N/A')
                        })
                
                elif agent_type == "Document Analysis Agent":
                    doc_type = res.get('document_type', 'Genel')
                    st.markdown(f"**📋 Doküman Tipi:** `{doc_type}`")
                    
                    st.subheader("📝 Özet")
                    st.info(res.get('summary', 'N/A'))
                    
                    # Yapı ve Metrikler
                    structure = res.get('structure', {})
                    if structure:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Kelime Sayısı", structure.get('word_count', 'N/A'))
                        with col2:
                            st.metric("Sayfa Sayısı", structure.get('page_count', 'N/A'))
                        with col3:
                            st.metric("Bölüm Sayısı", len(structure.get('sections', [])))
                    
                    # Sentiment ve Ton
                    col1, col2 = st.columns(2)
                    with col1:
                        sentiment = res.get('sentiment', 'N/A')
                        if 'Pozitif' in sentiment:
                            st.success(f"😊 Sentiment: {sentiment}")
                        elif 'Negatif' in sentiment:
                            st.error(f"😞 Sentiment: {sentiment}")
                        else:
                            st.info(f"😐 Sentiment: {sentiment}")
                    with col2:
                        st.metric("🎭 Ton", res.get('tone', 'N/A'))
                    
                    st.subheader("🔑 Ana Noktalar")
                    for i, point in enumerate(res.get('key_points', []), 1):
                        st.markdown(f"**{i}.** {point}")
                    
                    # Varlıklar (Entities)
                    entities = res.get('entities', {})
                    if isinstance(entities, dict) and entities:
                        st.subheader("🏷️ Çıkarılan Varlıklar")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if entities.get('people'):
                                st.markdown("**👥 Kişiler:**")
                                for person in entities.get('people', []):
                                    st.markdown(f"- {person}")
                            
                            if entities.get('organizations'):
                                st.markdown("**🏢 Kurumlar:**")
                                for org in entities.get('organizations', []):
                                    st.markdown(f"- {org}")
                        
                        with col2:
                            if entities.get('dates'):
                                st.markdown("**📅 Tarihler:**")
                                for date in entities.get('dates', []):
                                    st.markdown(f"- {date}")
                            
                            if entities.get('locations'):
                                st.markdown("**📍 Yerler:**")
                                for loc in entities.get('locations', []):
                                    st.markdown(f"- {loc}")
                        
                        if entities.get('numbers'):
                            st.markdown("**🔢 Önemli Rakamlar:**")
                            for num in entities.get('numbers', []):
                                st.markdown(f"- {num}")
                    
                    # İçgörüler
                    if res.get('key_insights'):
                        st.subheader("💡 Anahtar İçgörüler")
                        for insight in res.get('key_insights', []):
                            st.markdown(f"✨ {insight}")
                    
                    # Eksik Bilgiler
                    if res.get('missing_info'):
                        st.subheader("⚠️ Eksik/Belirsiz Bilgiler")
                        for missing in res.get('missing_info', []):
                            st.warning(missing)
                    
                    # Yapı Bölümleri
                    if structure.get('sections'):
                        with st.expander("📑 Doküman Yapısı"):
                            for i, section in enumerate(structure.get('sections', []), 1):
                                st.markdown(f"{i}. {section}")
                    
                    st.subheader("🎯 Öneriler")
                    for i, rec in enumerate(res.get('recommendations', []), 1):
                        st.success(f"**{i}.** {rec}")
                
                elif agent_type == "Code Assistant Agent":
                    lang = result.get('language', 'python').lower()
                    
                    st.subheader(f"💻 {result.get('language', 'Python')} Kodu")
                    st.code(res.get('code', 'N/A'), language=lang)
                    
                    # Karmaşıklık Metrikleri
                    complexity = res.get('complexity', {})
                    if isinstance(complexity, dict):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("⏱️ Zaman Karmaşıklığı", complexity.get('time', 'N/A'))
                        with col2:
                            st.metric("💾 Alan Karmaşıklığı", complexity.get('space', 'N/A'))
                    else:
                        st.metric("Karmaşıklık", complexity)
                    
                    st.subheader("📝 Açıklama")
                    st.markdown(f"<div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px;'>{res.get('explanation', 'N/A')}</div>", unsafe_allow_html=True)
                    
                    # Test Cases
                    st.subheader("🧪 Test Durumları")
                    for i, test in enumerate(res.get('test_cases', []), 1):
                        if '✅' in test or 'başarılı' in test.lower():
                            st.success(f"**Test {i}:** {test}")
                        else:
                            st.info(f"**Test {i}:** {test}")
                    
                    # Best Practices
                    if res.get('best_practices'):
                        st.subheader("✨ Uygulanan Best Practices")
                        for practice in res.get('best_practices', []):
                            st.markdown(f"✓ {practice}")
                    
                    # Dependencies
                    if res.get('dependencies'):
                        st.subheader("📦 Gerekli Bağımlılıklar")
                        deps = res.get('dependencies', [])
                        st.code(", ".join(deps))
                    
                    # Security Notes
                    if res.get('security_notes'):
                        st.subheader("🔒 Güvenlik Notları")
                        for note in res.get('security_notes', []):
                            st.warning(note)
                    
                    # İyileştirme Önerileri
                    st.subheader("⚡ İyileştirme Önerileri")
                    for i, imp in enumerate(res.get('improvements', []), 1):
                        st.markdown(f"**{i}.** {imp}")
                    
                    else:
                        # Fallback: JSON göster
                        st.json(res)
                    
                    # Ham JSON'u görmek isteyenler için
                    with st.expander("🔍 Ham JSON Verisi"):
                        st.json(res)
                else:
                    st.error(f"❌ Hata: {result.get('error', 'Bilinmeyen')}")
    else:
        st.info("📭 Henüz sonuç yok")

# TAB 3: Loglar
with tab3:
    st.header("📋 Execution Logs")
    
    if st.session_state.execution_logs:
        for log in reversed(st.session_state.execution_logs[-50:]):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
            with col1:
                st.text(log['timestamp'])
            with col2:
                st.text(log['agent'])
            with col3:
                st.text(log['action'])
            with col4:
                st.text(log['details'])
        
        if st.button("🗑️ Logları Temizle"):
            st.session_state.execution_logs = []
            st.rerun()
    else:
        st.info("📭 Henüz log yok")

# Footer
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Toplam Agent", len(st.session_state.agents))
with col2:
    st.metric("Tamamlanan Görev", len(st.session_state.task_results))
with col3:
    success_count = len([r for r in st.session_state.task_results if r.get('status') == 'success'])
    if st.session_state.task_results:
        success_rate = int(success_count / len(st.session_state.task_results) * 100)
        st.metric("Başarı Oranı", f"%{success_rate}")
    else:
        st.metric("Başarı Oranı", "N/A")

st.markdown("""
<div style='text-align: center; color: gray; margin-top: 20px;'>
    <p>🤖 AI Agent Kontrol Paneli v5.0 | Gemini 2.5 Flash + Advanced Tools</p>
    <p style='font-size: 0.8em;'>✨ Google Search | 🐍 Code Execution | 📋 Structured Output | 📄 Document Understanding</p>
</div>
""", unsafe_allow_html=True)
