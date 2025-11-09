import json
from typing import Dict, Any
import requests
from datetime import datetime

# 1. Basit Araçlar Tanımlama
def get_weather(city: str) -> Dict[str, Any]:
    """Basit bir hava durumu aracı"""
    # Gerçek bir API yerine örnek veri döndürüyoruz
    return {
        "city": city,
        "temperature": "22°C",
        "condition": "Güneşli",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

def calculate(expression: str) -> Dict[str, Any]:
    """Basit bir hesap makinesi aracı"""
    try:
        result = eval(expression)  # Güvenlik açıklarına dikkat! Sadece örnek amaçlı
        return {
            "expression": expression,
            "result": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

# 2. Kullanılabilir Araçlar Sözlüğü
TOOLS = {
    "get_weather": {
        "function": get_weather,
        "description": "Bir şehrin hava durumunu getirir",
        "parameters": {
            "city": {"type": "string", "description": "Şehir adı"}
        }
    },
    "calculate": {
        "function": calculate,
        "description": "Matematiksel ifade hesaplar",
        "parameters": {
            "expression": {"type": "string", "description": "Hesaplanacak ifade (örn: 2+2*3)"}
        }
    }
}

# 3. Araç Çağırma Fonksiyonu
def call_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Belirtilen aracı çağırır"""
    if tool_name not in TOOLS:
        return {"error": f"Bilinmeyen araç: {tool_name}"}
    
    tool = TOOLS[tool_name]
    try:
        # Aracı çağır ve sonucu döndür
        result = tool["function"](**parameters)
        return {"tool": tool_name, "result": result}
    except Exception as e:
        return {"error": f"Hata oluştu: {str(e)}"}

# 4. Örnek Kullanım
if __name__ == "__main__":
    # Örnek 1: Hava durumu sorgulama
    print("--- Hava Durumu Sorgulama ---")
    weather_result = call_tool("get_weather", {"city": "İstanbul"})
    print(json.dumps(weather_result, indent=2, ensure_ascii=False))
    
    # Örnek 2: Matematik işlemi
    print("\n--- Matematik İşlemi ---")
    calc_result = call_tool("calculate", {"expression": "5 * 8 + 3"})
    print(json.dumps(calc_result, indent=2, ensure_ascii=False))
    
    # Örnek 3: Hatalı araç adı
    print("\n--- Hatalı Araç Adı ---")
    error_result = call_tool("unknown_tool", {})
    print(json.dumps(error_result, indent=2, ensure_ascii=False))