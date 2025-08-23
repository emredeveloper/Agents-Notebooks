import os
import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import sys

# Import common utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'A2A-Agent'))
from common import (
    AGENT_CARD_WELL_KNOWN_PATH,
    build_public_agent_card,
    jsonrpc_error,
    jsonrpc_success,
    call_lmstudio,
)

PORT = int(os.getenv("CODE_REVIEW_AGENT_PORT", 8007))
BASE_URL = os.getenv("CODE_REVIEW_AGENT_BASE_URL", f"http://localhost:{PORT}")
MODEL = os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-4b-2507")

app = FastAPI(title="CodeReviewAgent")


@app.get(AGENT_CARD_WELL_KNOWN_PATH)
async def agent_card():
    card = build_public_agent_card(
        name="CodeReviewAgent",
        description="Kod analizi, review ve iyileştirme önerileri yapan ajan (LM Studio)",
        base_url=BASE_URL,
        streaming=False,
    )
    return JSONResponse(card)


def detect_programming_language(code: str) -> str:
    """
    Kod içeriğinden programlama dilini tespit eder.
    """
    code_lower = code.lower()
    
    # C++ indicators (check first due to specific syntax)
    if any(keyword in code for keyword in ['#include', 'std::', 'cout <<', 'int main()', 'using namespace std']):
        return "C++"
    
    # C# indicators (check before Java due to namespace)
    elif ('using System' in code or 'Console.WriteLine' in code or 
          ('namespace ' in code and 'public class' in code)):
        return "C#"
    
    # Java indicators (specific Java patterns)
    elif ('public class' in code and ('System.out' in code or 'public static void main' in code)) or \
         ('import java.' in code):
        return "Java"
    
    # Python indicators (specific Python patterns)
    elif ('def ' in code or 'if __name__' in code or 
          ('import ' in code and ('from ' in code or 'as ' in code)) or
          code.count('print(') > 0):
        return "Python"
    
    # JavaScript/TypeScript indicators
    elif any(keyword in code for keyword in ['function ', 'const ', 'let ', 'var ', 'console.log', '=>', 'async ', 'document.']):
        return "JavaScript/TypeScript"
    
    # HTML indicators
    elif any(keyword in code for keyword in ['<html', '<div', '<p>', '<body', '<head', '<!DOCTYPE']):
        return "HTML"
    
    # CSS indicators
    elif ('{' in code and '}' in code and ':' in code and ';' in code and 
          any(keyword in code for keyword in ['color', 'background', 'margin', 'padding', 'font-'])):
        return "CSS"
    
    # SQL indicators
    elif any(keyword in code_lower for keyword in ['select ', 'from ', 'where ', 'insert into', 'update ', 'delete from']):
        return "SQL"
    
    # Go indicators
    elif any(keyword in code for keyword in ['package main', 'func ', 'fmt.Print', 'import (']):
        return "Go"
    
    # Rust indicators
    elif any(keyword in code for keyword in ['fn main()', 'println!', 'let mut', 'use std::', 'cargo']):
        return "Rust"
    
    else:
        return "Unknown"


def detect_review_type(text: str) -> tuple[str, str]:
    """
    Kullanıcının ne tür bir code review istediğini belirler.
    Returns: (review_type, clean_code)
    """
    text_lower = text.lower().strip()
    
    # Security review
    if any(phrase in text_lower for phrase in [
        "güvenlik", "security", "vulnerability", "zafiyet", 
        "güvenlik açığı", "exploit", "injection"
    ]):
        return "security", text
    
    # Performance review
    elif any(phrase in text_lower for phrase in [
        "performans", "performance", "optimizasyon", "optimization",
        "hız", "speed", "yavaş", "slow", "bellek", "memory"
    ]):
        return "performance", text
    
    # Code quality/best practices
    elif any(phrase in text_lower for phrase in [
        "kalite", "quality", "best practice", "clean code", "refactor",
        "temiz kod", "iyileştir", "improve"
    ]):
        return "quality", text
    
    # Bug finding
    elif any(phrase in text_lower for phrase in [
        "bug", "hata", "error", "sorun", "problem", "çalışmıyor", "doesn't work"
    ]):
        return "bugs", text
    
    # Style/formatting
    elif any(phrase in text_lower for phrase in [
        "stil", "style", "format", "düzen", "convention", "standardize"
    ]):
        return "style", text
    
    # General review (default)
    else:
        return "general", text


def extract_code_from_text(text: str) -> str:
    """
    Metinden kod bloklarını çıkarır.
    """
    # Code block markers (```code```)
    code_pattern = r'```(?:[\w+]*\n)?(.*?)```'
    matches = re.findall(code_pattern, text, re.DOTALL)
    if matches:
        return '\n'.join(matches)
    
    # Indented code (4+ spaces)
    lines = text.split('\n')
    code_lines = []
    in_code_block = False
    
    for line in lines:
        if line.startswith('    ') or line.startswith('\t'):
            code_lines.append(line)
            in_code_block = True
        elif in_code_block and line.strip() == '':
            code_lines.append(line)
        elif in_code_block and not line.startswith('    '):
            break
    
    if code_lines:
        return '\n'.join(code_lines)
    
    # If no specific code format, return original text
    return text


async def perform_code_review(review_type: str, code: str, language: str) -> str:
    """
    Belirtilen review türünü gerçekleştirir.
    """
    
    if review_type == "security":
        prompt = f"""Sen senior bir güvenlik uzmanısın. Aşağıdaki {language} kodunu güvenlik açıları açısından incele.

Güvenlik İnceleme Kriterleri:
- Input validation eksiklikleri
- SQL injection, XSS gibi güvenlik açıkları
- Authentication/authorization sorunları
- Sensitive data handling
- Crypto kullanımı
- Rate limiting eksiklikleri

Kod ({language}):
{code}

Güvenlik İncelemesi (Türkçe):"""

    elif review_type == "performance":
        prompt = f"""Sen senior bir performans uzmanısın. Aşağıdaki {language} kodunu performans açısından incele.

Performans İnceleme Kriterleri:
- Algoritma karmaşıklığı (Big O)
- Bellek kullanımı
- I/O işlemleri optimizasyonu
- Döngü optimizasyonları
- Database query optimizasyonu
- Caching fırsatları

Kod ({language}):
{code}

Performans İncelemesi (Türkçe):"""

    elif review_type == "quality":
        prompt = f"""Sen senior bir yazılım mimarısın. Aşağıdaki {language} kodunu kod kalitesi açısından incele.

Kod Kalitesi Kriterleri:
- SOLID prensipleri
- DRY (Don't Repeat Yourself)
- Naming conventions
- Code readability
- Function/class design
- Error handling
- Documentation

Kod ({language}):
{code}

Kod Kalitesi İncelemesi (Türkçe):"""

    elif review_type == "bugs":
        prompt = f"""Sen experienced bir bug hunter'sın. Aşağıdaki {language} kodunda potansiyel hataları ara.

Bug Arama Kriterleri:
- Logic errors
- Null pointer exceptions
- Array/list index errors
- Type mismatches
- Resource leaks
- Race conditions
- Edge case handling

Kod ({language}):
{code}

Bug Analizi (Türkçe):"""

    elif review_type == "style":
        prompt = f"""Sen kod style uzmanısın. Aşağıdaki {language} kodunu formatting ve style açısından incele.

Style İnceleme Kriterleri:
- Indentation consistency
- Naming conventions
- Comment quality
- Line length
- Whitespace usage
- Import organization
- Code structure

Kod ({language}):
{code}

Style İncelemesi (Türkçe):"""

    else:  # general
        prompt = f"""Sen senior bir software engineer'sın. Aşağıdaki {language} kodunu genel olarak incele.

Genel İnceleme Kapsamı:
- Kod kalitesi ve okunabilirlik
- Potansiyel buglar
- Performans iyileştirmeleri
- Best practice uygunluğu
- Güvenlik dikkat edilmesi gerekenler
- Refactoring önerileri

Kod ({language}):
{code}

Genel Code Review (Türkçe):"""

    return await call_lmstudio(
        prompt=prompt,
        model=MODEL,
        temperature=0.1,  # Code review için düşük temperature
    )


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
    prompt = ""
    for p in parts:
        if p.get("kind") == "text":
            prompt += p.get("text", "") + "\n"

    try:
        # Kodun çıkarılması ve analizi
        code = extract_code_from_text(prompt.strip())
        
        if len(code.strip()) < 10:
            review_result = "💻 Lütfen review edilecek yeterli uzunlukta kod girin (en az 10 karakter)."
        else:
            # Programming language tespiti
            language = detect_programming_language(code)
            
            # Review türünü belirle
            review_type, _ = detect_review_type(prompt)
            
            # Code review gerçekleştir
            lm_response = await perform_code_review(review_type, code, language)
            
            # Review türüne göre emoji ve başlık ekle
            type_icons = {
                "security": "🔒 Güvenlik İncelemesi",
                "performance": "⚡ Performans İncelemesi", 
                "quality": "🏆 Kod Kalitesi İncelemesi",
                "bugs": "🐛 Bug Analizi",
                "style": "🎨 Style İncelemesi",
                "general": "💻 Genel Code Review"
            }
            
            icon = type_icons.get(review_type, "💻 Code Review")
            review_result = f"{icon}\n\n🔧 Dil: {language}\n\n{lm_response.strip()}"
            
    except Exception as e:
        # JSON-RPC hata cevabı
        return JSONResponse(
            jsonrpc_success(
                id_val,
                {
                    "contextId": message.get("contextId"),
                    "message": {
                        "role": "assistant",
                        "parts": [
                            {
                                "kind": "text",
                                "text": f"CodeReviewAgent LLM hatası: {str(e)}. Lütfen LM Studio sunucusunu ve model adını kontrol edin.",
                            }
                        ],
                    },
                    "error": True,
                },
            )
        )

    result = {
        "contextId": message.get("contextId"),
        "message": {
            "role": "assistant",
            "parts": [{"kind": "text", "text": review_result}],
        },
    }
    return JSONResponse(jsonrpc_success(id_val, result))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)