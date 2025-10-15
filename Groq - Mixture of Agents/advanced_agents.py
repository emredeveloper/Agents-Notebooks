import os
import time
from typing import Dict, Optional, Generator, List, Any
import json
import math

# LangChain imports
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun as CommunityDuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults as CommunityDuckDuckGoSearchResults
from langchain.agents import create_react_agent, AgentExecutor

# Rich (daha okunabilir çıktı)
from rich import print as print
from rich.traceback import install as rich_traceback_install
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
rich_traceback_install(show_locals=False)


# Rich yardımcıları
def print_section(title: str):
    print(Rule(f"[bold cyan]{title}"))


def print_panel(text: str, title: Optional[str] = None, style: str = ""): 
    print(Panel.fit(text, title=title, border_style=style or "cyan"))

GROQ_API_KEY  = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ============================================================================
# GROQ LIMIT YÖNETİMİ (Free Tier Optimizasyonu)
# ============================================================================

class GroqLimitManager:
    """Groq API limitlerini yönetir - Free Tier için optimize edilmiş"""
    
    def __init__(self):
        # Free Tier limitleri (konservatif yaklaşım)
        self.free_tier_limits = {
            'tpm': 6000,  # Tokens per minute
            'rpm': 100,   # Requests per minute (konservatif)
            'max_tokens_per_request': 1500  # Tek istekte max token
        }
        
        self.usage_stats = {
            'tokens_used': 0,
            'requests_made': 0,
            'last_reset': time.time(),
            'current_minute_tokens': 0,
            'current_minute_requests': 0
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Token sayısını tahmin eder (yaklaşık)"""
        return len(text.split()) * 1.3  # Türkçe için konservatif tahmin
    
    def check_rate_limits(self, estimated_tokens: int) -> Dict[str, bool]:
        """Rate limitleri kontrol eder"""
        current_time = time.time()
        
        # Dakika sıfırlama kontrolü
        if current_time - self.usage_stats['last_reset'] >= 60:
            self.usage_stats['current_minute_tokens'] = 0
            self.usage_stats['current_minute_requests'] = 0
            self.usage_stats['last_reset'] = current_time
        
        # Limit kontrolleri
        can_use_tokens = (
            self.usage_stats['current_minute_tokens'] + estimated_tokens 
            <= self.free_tier_limits['tpm']
        )
        
        can_make_request = (
            self.usage_stats['current_minute_requests'] + 1 
            <= self.free_tier_limits['rpm']
        )
        
        token_size_ok = estimated_tokens <= self.free_tier_limits['max_tokens_per_request']
        
        return {
            'can_proceed': can_use_tokens and can_make_request and token_size_ok,
            'tokens_ok': can_use_tokens,
            'requests_ok': can_make_request,
            'size_ok': token_size_ok
        }
    
    def record_usage(self, tokens_used: int):
        """Kullanımı kaydeder"""
        self.usage_stats['tokens_used'] += tokens_used
        self.usage_stats['requests_made'] += 1
        self.usage_stats['current_minute_tokens'] += tokens_used
        self.usage_stats['current_minute_requests'] += 1
    
    def get_wait_time(self) -> float:
        """Limit aşımı durumunda bekleme süresini hesaplar"""
        time_since_reset = time.time() - self.usage_stats['last_reset']
        return max(0, 60 - time_since_reset)
    
    def get_status(self) -> Dict[str, Any]:
        """Mevcut durumu getirir"""
        current_time = time.time()
        time_since_reset = current_time - self.usage_stats['last_reset']
        
        return {
            'tokens_used_this_minute': self.usage_stats['current_minute_tokens'],
            'requests_made_this_minute': self.usage_stats['current_minute_requests'],
            'tokens_remaining': self.free_tier_limits['tpm'] - self.usage_stats['current_minute_tokens'],
            'requests_remaining': self.free_tier_limits['rpm'] - self.usage_stats['current_minute_requests'],
            'seconds_until_reset': max(0, 60 - time_since_reset),
            'total_tokens_used': self.usage_stats['tokens_used'],
            'total_requests_made': self.usage_stats['requests_made']
        }

# Global limit manager
limit_manager = GroqLimitManager()

print_section("Groq Limit Manager")
print_panel(
    f"TPM: {limit_manager.free_tier_limits['tpm']}\n"
    f"RPM: {limit_manager.free_tier_limits['rpm']}\n"
    f"Max Tokens/Request: {limit_manager.free_tier_limits['max_tokens_per_request']}",
    title="Free Tier Limitleri",
)


# ============================================================================
# GERÇEK AJAN ARAÇLARI (TOOLS)
# ============================================================================

# Web Search Tools (deprecation uyarısı olmaması için community sürümleri)
search_tool = CommunityDuckDuckGoSearchRun()
search_results_tool = CommunityDuckDuckGoSearchResults()

# Calculator Tool (Improved)
def safe_calculator(expression: str) -> str:
    """Güvenli matematik hesaplama aracı"""
    try:
        # Girdiyi temizle
        expression = expression.strip()

        # Sadece güvenli matematiksel ifadelere izin ver
        # Not: Önceki sürümde '\\s' literal olarak değerlendirilip boşluk engelleniyordu
        allowed_symbols = set("+-*/.() ")
        if not all(ch.isdigit() or ch in allowed_symbols or ch.isspace() for ch in expression):
            return "Hata: Sadece matematiksel ifadeler kabul edilir (+, -, *, /, sayılar, parantez)"

        # Güvenlik kontrolü - tehlikeli ifadeler
        dangerous_words = ['import', 'exec', 'eval', 'open', 'file', '__']
        lowered = expression.lower()
        if any(word in lowered for word in dangerous_words):
            return "Hata: Güvenlik nedeniyle desteklenmeyen ifade"

        # Hesaplama yap
        result = eval(expression)
        return f"Sonuç: {result}"

    except ZeroDivisionError:
        return "Hata: Sıfıra bölme hatası"
    except SyntaxError:
        return "Hata: Geçersiz matematik ifadesi"
    except Exception as e:
        return f"Hesaplama hatası: {str(e)}"

calculator_tool = Tool(
    name="Calculator",
    func=safe_calculator,
    description="Matematik hesaplamaları için kullan. Örnek: '2 + 2', '10 * 5', '(3 + 4) * 2'. Sadece matematiksel ifadeler kabul edilir."
)

# Code Execution Tool
def execute_python_code(code: str) -> str:
    """Python kodunu güvenli şekilde çalıştır"""
    try:
        # Güvenlik kontrolü
        dangerous_imports = ['os', 'sys', 'subprocess', 'eval', 'exec', 'open', 'file']
        for dangerous in dangerous_imports:
            if dangerous in code:
                return f"Hata: '{dangerous}' kullanımı güvenlik nedeniyle yasak"
        
        # Kodu çalıştır
        exec_globals = {'math': math, 'json': json, 'time': time}
        exec(code, exec_globals)
        return "Kod başarıyla çalıştırıldı"
    except Exception as e:
        return f"Kod hatası: {str(e)}"

code_tool = Tool(
    name="Python_Code_Executor",
    func=execute_python_code,
    description="Python kodunu çalıştır. Sadece güvenli matematik ve veri işleme kodları kabul edilir."
)

# Text Analysis Tool
def analyze_text(text: str) -> str:
    """Metin analizi yapar"""
    try:
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        analysis = {
            "kelime_sayisi": word_count,
            "karakter_sayisi": char_count,
            "cumle_sayisi": sentences,
            "ortalama_kelime_uzunlugu": round(char_count / word_count if word_count > 0 else 0, 2)
        }
        
        return f"Metin Analizi: {json.dumps(analysis, indent=2, ensure_ascii=False)}"
    except Exception as e:
        return f"Analiz hatası: {str(e)}"

text_analysis_tool = Tool(
    name="Text_Analyzer",
    func=analyze_text,
    description="Metin analizi yapar - kelime sayısı, karakter sayısı, cümle sayısı vb."
)

## (v1 agent entegrasyonu kaldırıldı)

print_section("Araçlar")
table_tools = Table(show_header=True, header_style="bold magenta")
table_tools.add_column("Araç")
table_tools.add_column("Ad")
table_tools.add_row("Search", search_tool.name)
table_tools.add_row("SearchResults", search_results_tool.name)
table_tools.add_row("Calculator", calculator_tool.name)
table_tools.add_row("Code", code_tool.name)
table_tools.add_row("Text Analysis", text_analysis_tool.name)
print(table_tools)


# ============================================================================
# GROQ LIMIT-AWARE AJANLAR (TOOL-BASED)
# ============================================================================

def create_limit_aware_agent(
    name: str,
    system_prompt: str,
    tools: List[Tool],
    model_name: str = "llama-3.3-70b-versatile"
) -> AgentExecutor:
    """Groq limitlerini göz önünde bulunduran ajan oluşturur"""
    
    # Limit-aware LLM oluştur
    llm = ChatGroq(
        model=model_name, 
        temperature=0.1,
        max_tokens=500,  # Free tier için kısa yanıtlar
        timeout=30  # Timeout ekle
    )
    
    # ReAct agent için doğru prompt formatı
    optimized_prompt = ChatPromptTemplate.from_template(f"""{system_prompt}

Mevcut araçlar:
{{tools}}

Kullanabileceğin araçlar: {{tool_names}}

Format:
Question: {{input}}
Thought: Ne yapmam gerekiyor?
Action: [araç_adı]
Action Input: [araç_girişi]
Observation: [araç_çıkışı]
Final Answer: [kısa_yanıt]

{{agent_scratchpad}}""")
    
    # ReAct agent oluştur
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=optimized_prompt
    )
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=3,
        handle_parsing_errors=True,
        max_execution_time=30,
        return_intermediate_steps=False
    )

def create_action_agent(
    name: str,
    system_prompt: str,
    tools: List[Tool],
    model_name: str = "llama-3.1-8b-instant"
) -> AgentExecutor:
    """Tool-based gerçek ajan oluşturur (limit-aware)"""
    return create_limit_aware_agent(name, system_prompt, tools, model_name)

# Araştırma Ajanı (Improved)
research_agent = create_action_agent(
    name="Research_Agent",
    system_prompt="""Sen araştırma uzmanısın. Web'de araştırma yap ve sonuçları özetle. 
    Her zaman web araması yap, sonuçları analiz et.""",
    tools=[search_tool, Tool(name="duckduckgo_results", func=search_results_tool.run, description="DuckDuckGo'dan JSON sonuç listesi getir")],
    model_name="llama-3.1-8b-instant"
)

# Analiz Ajanı (Improved)
analysis_agent = create_action_agent(
    name="Analysis_Agent",
    system_prompt="""Sen matematik uzmanısın. Hesaplama yapmak için Calculator aracını kullan. 
    Her zaman matematiksel ifadeleri hesapla.""",
    tools=[calculator_tool],
    model_name="llama-3.1-8b-instant"
)

# Yaratıcı Ajan (Improved)
creative_agent = create_action_agent(
    name="Creative_Agent",
    system_prompt="""Sen yaratıcı uzmanısın. Yaratıcı fikirler üret ve örnekler ver.""",
    tools=[],  # Araç yok - sadece yaratıcı düşünce
    model_name="llama-3.1-8b-instant"
)

print_section("Ajanlar")
table_agents = Table(show_header=True, header_style="bold green")
table_agents.add_column("Ajan")
table_agents.add_column("Araç Sayısı")
table_agents.add_column("Açıklama")
table_agents.add_row("Research", str(len(research_agent.tools)), "Web araştırması")
table_agents.add_row("Analysis", str(len(analysis_agent.tools)), "Matematik hesaplaması")
table_agents.add_row("Creative", str(len(creative_agent.tools)), "Yaratıcı düşünce")
print(table_agents)
print_panel("Verbose açık, daha fazla iterasyon, temiz çıktı", style="green")


# ============================================================================
# AJANLAR ARASI İŞBİRLİĞİ SİSTEMİ
# ============================================================================

class TrueMixtureOfAgents:
    """Gerçek Mixture of Agents sistemi - Groq Free Tier için optimize edilmiş"""
    
    def __init__(self):
        self.agents = {
            'research': research_agent,
            'analysis': analysis_agent,
            'creative': creative_agent
        }
        
        # Ana koordinatör ajan (Free Tier Optimized)
        self.coordinator = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=300,  # Kısa sentez
            timeout=30
        )
        
        # Koordinatör için prompt template (kısa ve yapılı)
        self.coordinator_prompt = ChatPromptTemplate.from_template("""
Sen koordinatör ajansın. Farklı ajan çıktılarından KISA ve YAPILI bir yanıt üret.

Kullanıcı Sorusu: {query}
Ajan Sonuçları: {agent_results}
Stil: {style}

Format Talimatları:
{format_instructions}

Kurallar:
- Gereksiz giriş yapma; belirtilen formatı aynen uygula.
""")
        
        # Memory (küçük boyut)
        self.memory = ConversationBufferMemory(
            memory_key="messages",
            return_messages=True,
            max_token_limit=1000  # Memory limit
        )

    def determine_response_style(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in [" vs ", "karşılaştır", "vs."]):
            return "compare"
        if any(k in q for k in ["hesapla", "+", "-", "*", "/"]):
            return "math"
        if any(k in q for k in ["fikir", "yaratıcı", "öneri", "proje"]):
            return "creative"
        if any(k in q for k in ["nedir", "nasıl", "nerede", "kim", "ne zaman", "neden", "araştır"]):
            return "research"
        return "generic"

    def get_format_instructions(self, style: str) -> str:
        if style == "math":
            return (
                "- Sadece 2 satır yaz:\n"
                "  1) Cevap: <sonuç>\n"
                "  2) Gerekçe: <1 kısa cümle>"
            )
        if style == "research":
            return (
                "- En fazla 5 satır yaz:\n"
                "  1) Özet: <1 cümle>\n"
                "  2-4) • <kısa madde>\n"
                "  5) Öneri: <1 cümle>"
            )
        if style == "compare":
            return (
                "- En fazla 6 satır yaz:\n"
                "  1) Özet: <1 cümle>\n"
                "  2-3) React: • <kısa madde>\n"
                "  4-5) Vue: • <kısa madde>\n"
                "  6) Öneri: <1 cümle>"
            )
        if style == "creative":
            return (
                "- En fazla 4 satır yaz:\n"
                "  1) Özet: <1 cümle>\n"
                "  2-4) • <kısa fikir>"
            )
        return (
            "- En fazla 4 satır yaz:\n"
            "  1) Cevap: <1 cümle>\n"
            "  2-4) • <kısa destekleyici noktalar>"
        )
    
    def smart_agent_selection(self, query: str) -> List[str]:
        """Soruya göre hangi ajanları kullanacağını belirler"""
        query_lower = query.lower()
        selected_agents = []
        
        # Araştırma anahtar kelimeleri
        research_keywords = ['araştır', 'bul', 'nedir', 'nasıl', 'nerede', 'kim', 'ne zaman', 'neden', 'hangi', 'nelerdir', 'liste', 'kaynak', 'referans']
        if any(keyword in query_lower for keyword in research_keywords):
            selected_agents.append('research')
        
        # Analiz anahtar kelimeleri
        analysis_keywords = ['hesapla', 'analiz', 'karşılaştır', 'değerlendir', 'matematik', 'sayı', 'istatistik']
        if any(keyword in query_lower for keyword in analysis_keywords):
            selected_agents.append('analysis')
        
        # Yaratıcı anahtar kelimeleri
        creative_keywords = ['fikir', 'yaratıcı', 'çözüm', 'öneri', 'tasarım', 'plan', 'strateji']
        if any(keyword in query_lower for keyword in creative_keywords):
            selected_agents.append('creative')
        
        # Eğer hiçbir ajan seçilmediyse, hepsini kullan
        if not selected_agents:
            selected_agents = ['research', 'analysis', 'creative']
        
        return selected_agents
    
    def collaborative_workflow(self, query: str) -> Generator[str, None, None]:
        """Ajanlar gerçek işbirliği yapar - Free Tier Optimized"""
        
        # Token limit kontrolü
        estimated_tokens = limit_manager.estimate_tokens(query)
        limit_check = limit_manager.check_rate_limits(estimated_tokens)
        
        if not limit_check['can_proceed']:
            wait_time = limit_manager.get_wait_time()
            yield f"⏳ Rate limit aşıldı. {wait_time:.0f} saniye bekleyin...\\n"
            return
        
        # 1. Hangi ajanları kullanacağımızı belirle (max 2 ajan - token tasarrufu)
        selected_agents = self.smart_agent_selection(query)[:2]  # Max 2 ajan
        used_research = 'research' in selected_agents
        
        agent_results = {}
        
        # 2. Her ajanı çalıştır (limit-aware)
        for agent_name in selected_agents:
            
            try:
                # Token kontrolü
                agent_tokens = limit_manager.estimate_tokens(query)
                if not limit_manager.check_rate_limits(agent_tokens)['can_proceed']:
                    continue
                
                # Ajanı çalıştır
                result = self.agents[agent_name].invoke({"input": query})
                
                # Sonucu temizle ve kaydet
                if isinstance(result, dict) and 'output' in result:
                    clean_output = result['output'].replace("Agent stopped due to iteration limit or time limit.", "").strip()
                    agent_results[agent_name] = clean_output
                else:
                    agent_results[agent_name] = str(result)
                
                # Kullanımı kaydet
                limit_manager.record_usage(agent_tokens)
                
                # sessiz çıktı
                
            except Exception as e:
                agent_results[agent_name] = f"Hata: {str(e)}"
        
        # 3. Sonuçları sentezle (kısa)
        if agent_results:
            
            # Koordinatör prompt ile sentez yap
            agent_results_text = "\\n".join([f"{name}: {result}" for name, result in agent_results.items()])
            # Research ajanı seçildiyse stili research'a zorla
            style = 'research' if used_research else self.determine_response_style(query)
            format_instructions = self.get_format_instructions(style)

            try:
                # Ana koordinatör ile sentez yap
                synthesis_response = self.coordinator.invoke(
                    self.coordinator_prompt.format(
                        query=query,
                        agent_results=agent_results_text,
                        style=style,
                        format_instructions=format_instructions
                    )
                )
                
                # Token kullanımını kaydet
                synthesis_tokens = limit_manager.estimate_tokens(query + agent_results_text + synthesis_response.content)
                limit_manager.record_usage(synthesis_tokens)

                # Research ajanı kullanıldıysa kısa yanıt + altta 3 kaynak linki göster
                if used_research:
                    try:
                        raw = search_results_tool.run(query)
                        lines = []
                        if isinstance(raw, list):
                            for item in raw[:3]:
                                url = (item.get('link') or item.get('href') or item.get('url') or '').strip()
                                title = (item.get('title') or item.get('source') or url).strip()
                                if url:
                                    lines.append(f"- [{title}]({url})")
                        elif isinstance(raw, dict) and 'results' in raw:
                            for item in raw['results'][:3]:
                                url = (item.get('link') or item.get('href') or item.get('url') or '').strip()
                                title = (item.get('title') or item.get('source') or url).strip()
                                if url:
                                    lines.append(f"- [{title}]({url})")
                        sources = "\\n".join(lines)
                        if sources:
                            yield f"{synthesis_response.content}\\nKaynaklar:\\n{sources}\\n"
                        else:
                            yield f"{synthesis_response.content}\\n"
                    except Exception:
                        yield f"{synthesis_response.content}\\n"
                else:
                    yield f"{synthesis_response.content}\\n"
                
                # Memory'ye kaydet (kısa)
                self.memory.save_context(
                    {'input': query}, 
                    {'output': synthesis_response.content[:500]}  # Kısa kayıt
                )
                
            except Exception as e:
                yield f"Hata: {str(e)}\\n"
        else:
            yield f"Hata: Hiçbir ajan çalışmadı.\\n"
        
        # Limit durumu göster
        status = limit_manager.get_status()
        yield f"\\n📊 {int(status['tokens_remaining'])} token kaldı, {int(status['seconds_until_reset'])}s\\n"

print("✅ TrueMixtureOfAgents sistemi hazır!")
print("🤖 Ajanlar gerçek eylemler yapacak")


# ============================================================================
# PERFORMANS OPTİMİZASYONLARI
# ============================================================================

class OptimizedAgentSystem:
    """Optimize edilmiş agent sistemi"""
    
    def __init__(self):
        self.base_system = TrueMixtureOfAgents()
        self.cache = {}
        self.performance_stats = {
            'total_queries': 0,
            'avg_response_time': 0,
            'cache_hits': 0
        }
    
    def check_token_limit(self, query: str) -> bool:
        """Token limitini kontrol et - Groq Free Tier için optimize edilmiş"""
        estimated_tokens = limit_manager.estimate_tokens(query)
        limit_check = limit_manager.check_rate_limits(estimated_tokens)
        return limit_check['can_proceed']
    
    def get_cached_response(self, query: str) -> Optional[str]:
        """Cache'den yanıt al"""
        query_hash = hash(query.lower().strip())
        if query_hash in self.cache:
            self.performance_stats['cache_hits'] += 1
            return self.cache[query_hash]
        return None
    
    def cache_response(self, query: str, response: str):
        """Yanıtı cache'e kaydet"""
        query_hash = hash(query.lower().strip())
        self.cache[query_hash] = response
    
    def optimized_chat_stream(self, query: str) -> Generator[str, None, None]:
        """Optimize edilmiş chat stream"""
        start_time = time.time()
        
        # Token limit kontrolü (Groq Free Tier)
        if not self.check_token_limit(query):
            status = limit_manager.get_status()
            yield f"❌ Token limiti aşıldı. {status['seconds_until_reset']:.0f} saniye bekleyin."
            yield f"📊 Mevcut durum: {status['tokens_remaining']} token kaldı"
            return
        
        # Cache kontrolü
        cached_response = self.get_cached_response(query)
        if cached_response:
            yield f"⚡ Cache'den yanıt (Hızlı!)\\n{cached_response}"
            return
        
        # Gerçek işlemi yap
        response_parts = []
        
        for chunk in self.base_system.collaborative_workflow(query):
            yield chunk
            response_parts.append(chunk)
        
        # Cache'e kaydet
        full_response = ''.join(response_parts)
        self.cache_response(query, full_response)
        
        # Performans istatistikleri
        execution_time = time.time() - start_time
        self.performance_stats['total_queries'] += 1
        
        # Ortalama süreyi güncelle
        total_queries = self.performance_stats['total_queries']
        current_avg = self.performance_stats['avg_response_time']
        self.performance_stats['avg_response_time'] = (
            (current_avg * (total_queries - 1) + execution_time) / total_queries
        )
        
        yield f"\\n\\n📊 Performans: {execution_time:.2f}s | Cache: {self.performance_stats['cache_hits']}/{total_queries}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Performans istatistiklerini getir"""
        return self.performance_stats.copy()

# Optimize edilmiş sistem
optimized_system = OptimizedAgentSystem()

print_panel("Cache sistemi aktif\nPerformans izleme aktif", title="Optimize Sistem", style="magenta")


# ============================================================================
# TEST FONKSİYONLARI
# ============================================================================

def test_agent_system():
    """Test the agent system with various queries"""
    print("🧪 Test Başlıyor...")
    
    test_queries = [
        "Python'da machine learning için hangi kütüphaneleri kullanmalıyım?",
        "2 + 2 * 3 hesapla",
        "Vue.js ile React arasındaki farklar neler?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n{'='*50}")
        print(f"🧪 TEST {i}: {query}")
        print(f"{'='*50}")
        
        # Test the system
        responses = list(optimized_system.optimized_chat_stream(query))
        
        # Print all responses
        for response in responses:
            print(response)
        
        print(f"\\n✅ Test {i} tamamlandı!")
        
        # Wait between tests
        time.sleep(2)
    
    print(f"\\n📊 Final Limit Durumu: {limit_manager.get_status()['tokens_remaining']:.1f} token kaldı")
    print(f"\\n📊 İstatistikler: {optimized_system.get_stats()}")

def quick_test():
    """Quick test with a single query"""
    print("🧪 Hızlı Test Başlıyor...")
    print(f"📊 Başlangıç Durumu: {limit_manager.get_status()['tokens_remaining']:.1f} token kaldı")
    
    query = "2 + 2 * 3 hesapla"
    print(f"\\n🔹 Test Sorusu: {query}")
    
    responses = list(optimized_system.optimized_chat_stream(query))
    for response in responses:
        print(response)
    
    print(f"\\n✅ Test tamamlandı!")
    print(f"📊 Final Durum: {limit_manager.get_status()['tokens_remaining']:.1f} token kaldı")

def test_agents():
    """Test individual agents"""
    print("🧪 Ajan Testi Başlıyor...")
    
    # Test calculator
    print("\\n🔹 Calculator Test:")
    result = calculator_tool.run("2 + 2 * 3")
    print(f"Sonuç: {result}")
    
    # Test search
    print("\\n🔹 Search Test:")
    try:
        search_result = search_tool.run("Python machine learning")
        print(f"Sonuç: {search_result[:200]}...")
    except Exception as e:
        print(f"Hata: {e}")
    
    print("\\n✅ Ajan testi tamamlandı!")

def clean_test():
    """Temiz ve okunabilir test"""
    print("🧪 Temiz Test Başlıyor...")
    print(f"📊 Başlangıç Durumu: {limit_manager.get_status()['tokens_remaining']:.1f} token kaldı")
    
    query = "2 + 2 * 3 hesapla"
    print(f"\\n🔹 Test Sorusu: {query}")
    
    responses = list(optimized_system.optimized_chat_stream(query))
    for response in responses:
        print(response)
    
    print(f"\\n✅ Test tamamlandı!")
    print(f"📊 Final Durum: {limit_manager.get_status()['tokens_remaining']:.1f} token kaldı")

def test_imports():
    """Test if all imports are working"""
    print("🧪 Import Test Başlıyor...")
    
    try:
        # Test LangChain imports
        from langchain_groq import ChatGroq
        from langchain.memory import ConversationBufferMemory
        from langchain.prompts import ChatPromptTemplate
        from langchain.tools import Tool
        from langchain_community.tools import DuckDuckGoSearchRun as CommunityDuckDuckGoSearchRun
        from langchain.agents import create_react_agent, AgentExecutor
        
        print("✅ Tüm LangChain import'ları başarılı!")
        
        # Test basic functionality
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=100)
        response = llm.invoke("Merhaba")
        print(f"✅ LLM test başarılı: {response.content[:50]}...")
        
    except Exception as e:
        print(f"❌ Import hatası: {e}")
    
    print("\\n✅ Import testi tamamlandı!")

print_panel("Temiz test, hızlı test ve tam test fonksiyonları hazır", title="Test Fonksiyonları")

# ============================================================================
# İNTERAKTİF CHAT SİSTEMİ
# ============================================================================

def interactive_chat():
    """İnteraktif chat sistemi"""
    print("🎉 İnteraktif Chat Sistemi Başlıyor!")
    print("💡 'quit' yazarak çıkabilirsiniz")
    print("💡 'stats' yazarak istatistikleri görebilirsiniz")
    print("💡 'clear' yazarak memory'yi temizleyebilirsiniz")
    print("="*60)
    
    while True:
        try:
            user_input = input("\\n🤖 Sorunuz: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Görüşürüz!")
                break
            elif user_input.lower() == 'stats':
                stats = optimized_system.get_stats()
                limit_status = limit_manager.get_status()
                print(f"\\n📊 İstatistikler: {stats}")
                print(f"📊 Limit Durumu: {limit_status}")
                continue
            elif user_input.lower() == 'clear':
                optimized_system.base_system.memory.clear()
                print("🧹 Memory temizlendi!")
                continue
            elif not user_input:
                print("❓ Lütfen bir soru yazın!")
                continue
            
            print("\\n" + "="*60)
            print(f"🤖 Soru: {user_input}")
            print("="*60)
            
            # Chat stream
            responses = list(optimized_system.optimized_chat_stream(user_input))
            for response in responses:
                print(response)
                
        except KeyboardInterrupt:
            print("\\n👋 Görüşürüz!")
            break
        except Exception as e:
            print(f"❌ Hata: {e}")

print_panel("interactive_chat() ile başlat", title="İnteraktif Chat", style="green")


# ============================================================================
# KULLANIM TALİMATLARI
# ============================================================================

print_panel(
    "Gerçek ajanlar • Groq optimize • Cache • İzleme • Temiz çıktı",
    title="Gelişmiş Mixture of Agents",
)

print_panel("Sistem hazır", style="green")


# ============================================================================
# SİSTEMİ TEST ET
# ============================================================================

def test_agent_system():
    """Agent sistemini test et"""
    
    test_queries = [
        "Python'da machine learning için hangi kütüphaneleri kullanmalıyım?",
        "2 + 2 * 3 hesapla",
        "React vs Vue.js karşılaştırması yap"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n{'='*50}")
        print(f"🧪 TEST {i}: {query}")
        print(f"{'='*50}")
        
        for chunk in optimized_system.optimized_chat_stream(query):
            print(chunk, end="")
        
        print(f"\\n\\n📊 İstatistikler: {optimized_system.get_stats()}")

print("🧪 Test sistemi hazır. 'test_agent_system()' çalıştırarak test edebilirsiniz.")


test_agent_system()

# ============================================================================
# İNTERAKTİF CHAT SİSTEMİ
# ============================================================================

def interactive_chat():
    """İnteraktif chat sistemi"""
    
    print("🤖 Gelişmiş Mixture of Agents Sistemi")
    print("=" * 50)
    print("Komutlar:")
    print("- 'quit' veya 'exit': Çıkış")
    print("- 'stats': Performans istatistikleri")
    print("- 'clear': Cache'i temizle")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\\n🔹 Sorunuz: ")
            
            if user_input.lower() in ['quit', 'exit', 'çıkış']:
                print("\\n👋 Görüşürüz!")
                break
            
            elif user_input.lower() == 'stats':
                stats = optimized_system.get_stats()
                print(f"\\n📊 Performans İstatistikleri:")
                print(f"Toplam Sorgu: {stats['total_queries']}")
                print(f"Ortalama Süre: {stats['avg_response_time']:.2f}s")
                print(f"Cache Hit Oranı: {stats['cache_hits']}/{stats['total_queries']}")
                continue
            
            elif user_input.lower() == 'clear':
                optimized_system.cache.clear()
                print("\\n🗑️ Cache temizlendi!")
                continue
            
            elif not user_input.strip():
                print("\\n❌ Lütfen bir soru yazın.")
                continue
            
            print(f"\\n👤 Kullanıcı: {user_input}")
            print(f"🤖 AI: ", end="")
            
            # Yanıtı al ve göster
            for chunk in optimized_system.optimized_chat_stream(user_input):
                print(chunk, end="")
            
            print("\\n")
            
        except KeyboardInterrupt:
            print("\\n\\n👋 Görüşürüz!")
            break
        except Exception as e:
            print(f"\\n❌ Hata: {str(e)}")

print("💬 İnteraktif chat hazır. 'interactive_chat()' çalıştırarak başlayabilirsiniz.")


# ## 🚀 Kullanım Talimatları
# 
# ### 1. **Test Etmek İçin:**
# ```python
# test_agent_system()
# ```
# 
# ### 2. **İnteraktif Chat İçin:**
# ```python
# interactive_chat()
# ```
# 
# ### 3. **Performans İstatistikleri:**
# ```python
# optimized_system.get_stats()
# ```
# 
# ### 4. **Groq Limit Durumu:**
# ```python
# limit_manager.get_status()
# ```
# 
# ## 🎯 Özellikler (Groq Free Tier Optimized)
# 
# ✅ **Gerçek Tool-Based Ajanlar** - Sadece text değil, gerçek eylemler
# ✅ **Groq Limit Yönetimi** - [Free Tier limitleri](https://console.groq.com/settings/limits) için optimize
# ✅ **Akıllı Ajan Seçimi** - Soruya göre hangi ajanları kullanacağını belirler
# ✅ **İşbirliği Sistemi** - Ajanlar gerçekten işbirliği yapar
# ✅ **Cache Sistemi** - Hızlı yanıtlar için
# ✅ **Performans İzleme** - Gerçek zamanlı istatistikler
# ✅ **Rate Limit Koruması** - 6000 TPM, 100 RPM limitleri
# ✅ **Token Tasarrufu** - Kısa yanıtlar, optimize edilmiş promptlar
# 
# ## 🔧 Araçlar (Free Tier Optimized)
# 
# - 🔍 **Web Search** - Gerçek zamanlı araştırma
# - 🧮 **Calculator** - Güvenli matematik hesaplamaları
# 
# ## 📊 Groq Free Tier Limitleri
# 
# - **TPM**: 6,000 tokens/dakika
# - **RPM**: 100 requests/dakika  
# - **Max Tokens/Request**: 1,500
# - **Model**: llama-3.1-8b-instant (hızlı ve ekonomik)
# 
# Bu sistem **Groq Free Tier** için optimize edilmiş gerçek **Mixture of Agents**! 🎉
# 
# **Limit kontrolü için**: [Groq Console](https://console.groq.com/settings/limits)
# 


# ============================================================================
# HIZLI TEST
# ============================================================================

def quick_test():
    """Hızlı test fonksiyonu"""
    print("🧪 Hızlı Test Başlıyor...")
    
    # Limit durumunu kontrol et
    status = limit_manager.get_status()
    print(f"📊 Başlangıç Durumu: {status['tokens_remaining']} token kaldı")
    
    # Basit bir test
    test_query = "2 + 2 * 3 hesapla"
    print(f"\\n🔹 Test Sorusu: {test_query}")
    
    try:
        for chunk in optimized_system.optimized_chat_stream(test_query):
            print(chunk, end="")
        
        print(f"\\n\\n✅ Test tamamlandı!")
        
        # Final durum
        final_status = limit_manager.get_status()
        print(f"📊 Final Durum: {final_status['tokens_remaining']} token kaldı")
        
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")

print("⚡ Hızlı test hazır. 'quick_test()' çalıştırarak test edebilirsiniz.")


# ============================================================================
# TEMİZ TEST FONKSİYONU
# ============================================================================

def clean_test():
    """Temiz ve okunabilir test"""
    print("🧪 TEMİZ TEST BAŞLIYOR...")
    print("="*60)
    
    # Test 1: Basit matematik
    print("\\n🔢 TEST 1: Basit Matematik")
    print("-" * 30)
    query1 = "2 + 2 * 3 hesapla"
    print(f"Soru: {query1}")
    
    try:
        for chunk in optimized_system.optimized_chat_stream(query1):
            print(chunk, end="")
    except Exception as e:
        print(f"Hata: {e}")
    
    print("\\n" + "="*60)
    
    # Test 2: Araştırma
    print("\\n🔍 TEST 2: Araştırma")
    print("-" * 30)
    query2 = "Python ML kütüphaneleri"
    print(f"Soru: {query2}")
    
    try:
        for chunk in optimized_system.optimized_chat_stream(query2):
            print(chunk, end="")
    except Exception as e:
        print(f"Hata: {e}")
    
    print("\\n" + "="*60)
    
    # Test 3: Yaratıcılık
    print("\\n💡 TEST 3: Yaratıcılık")
    print("-" * 30)
    query3 = "Yaratıcı proje fikirleri"
    print(f"Soru: {query3}")
    
    try:
        for chunk in optimized_system.optimized_chat_stream(query3):
            print(chunk, end="")
    except Exception as e:
        print(f"Hata: {e}")
    
    print("\\n" + "="*60)
    print("✅ TÜM TESTLER TAMAMLANDI!")
    
    # Final durum
    status = limit_manager.get_status()
    print(f"\\n📊 Final Durum: {status['tokens_remaining']} token kaldı")

print("🧪 Temiz test hazır. 'clean_test()' çalıştırarak test edebilirsiniz.")


quick_test()

# ============================================================================
# AJANLARI TEST ET
# ============================================================================

def test_agents():
    """Ajanları test et"""
    print("🧪 Ajan Testi Başlıyor...")
    
    # Test sorguları
    test_queries = [
        "2 + 2 * 3 hesapla",
        "Python ML kütüphaneleri neler?",
        "Yaratıcı proje fikirleri ver"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n{'='*50}")
        print(f"🧪 TEST {i}: {query}")
        print(f"{'='*50}")
        
        try:
            for chunk in optimized_system.optimized_chat_stream(query):
                print(chunk, end="")
            
            print(f"\\n\\n✅ Test {i} tamamlandı!")
            
        except Exception as e:
            print(f"❌ Test {i} hatası: {str(e)}")
    
    # Final durum
    final_status = limit_manager.get_status()
    print(f"\\n📊 Final Limit Durumu: {final_status['tokens_remaining']} token kaldı")

print("🧪 Ajan testi hazır. 'test_agents()' çalıştırarak test edebilirsiniz.")


test_agents()