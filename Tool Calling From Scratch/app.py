"""
Tool Calling From Scratch using Google Gemini 2.5 Flash

This implementation demonstrates:
1. Manual tool calling with schemas and system prompts
2. Production-level tool calling using native Gemini API

Based on: https://towardsdatascience.com/implementing-ai-agents-from-scratch-tools-planning-and-memory-68e954207b7a
"""

import os
import json
import inspect
import sys
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pytz
import ast
import operator

# Rich for beautiful console output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich import print as rprint
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
except ImportError:
    print("Installing rich for better output...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich import print as rprint
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()

# Use the modern google-genai package
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Please install: pip install google-genai beautifulsoup4 requests pytz rich")
    exit(1)


# Configure Gemini API
def configure_gemini():
    """Configure the Gemini API with your API key"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


# ============================================================================
# PART 1: Define REAL Working Tool Functions
# ============================================================================

def google_search(query: str, num_results: int = 5) -> dict:
    """
    Tool used to perform real Google web searches using DuckDuckGo API (no API key needed).
    
    Args:
        query: The search query.
        num_results: Number of results to return (default 5).
    
    Returns:
        A dictionary of search results with titles, URLs, and snippets.
    """
    try:
        # Using DuckDuckGo Instant Answer API (free, no key needed)
        try:
            from duckduckgo_search import DDGS
            use_ddgs = True
        except ImportError:
            # Try the new package name
            try:
                from ddgs import DDGS
                use_ddgs = True
            except ImportError:
                use_ddgs = False
        
        if use_ddgs:
            results = []
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress deprecation warning
                with DDGS() as ddgs:
                    search_results = list(ddgs.text(query, max_results=num_results))
                    for result in search_results:
                        results.append({
                            "title": result.get("title", ""),
                            "url": result.get("href", ""),
                            "snippet": result.get("body", "")
                        })
            
            return {
                "query": query,
                "results_count": len(results),
                "results": results
            }
        else:
            # Fallback to a simple requests-based search
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='result')[:num_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem:
                    results.append({
                        "title": title_elem.get_text(strip=True),
                        "url": title_elem.get('href', ''),
                        "snippet": snippet_elem.get_text(strip=True) if snippet_elem else ""
                    })
            
            return {
                "query": query,
                "results_count": len(results),
                "results": results
            }
    except Exception as e:
        return {
            "error": f"Search failed: {str(e)}",
            "query": query,
            "results": []
        }


def scrape_url(url: str, max_length: int = 2000) -> dict:
    """
    Tool used to scrape and clean HTML content from a web URL.
    
    Args:
        url: The URL to scrape.
        max_length: Maximum length of text to return (default 2000 chars).
    
    Returns:
        A dictionary with the cleaned text content and metadata.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
        
        # Get title
        title = soup.title.string if soup.title else "No title"
        
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up extra whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = '\n'.join(lines)
        
        # Truncate if too long
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length] + "...\n[Content truncated]"
        
        return {
            "url": url,
            "title": title,
            "content": clean_text,
            "length": len(clean_text),
            "status": "success"
        }
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "status": "failed"
        }


def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Get real current weather for a location using Open-Meteo API (free, no key needed).
    
    Args:
        location: The city name (e.g., "San Francisco", "London", "Istanbul")
        unit: Temperature unit ("celsius" or "fahrenheit")
    
    Returns:
        Weather information including temperature and conditions
    """
    try:
        # First, geocode the location using Open-Meteo Geocoding API
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(location)}&count=1&language=en&format=json"
        geo_response = requests.get(geocoding_url, timeout=5)
        geo_data = geo_response.json()
        
        if not geo_data.get("results"):
            return {"error": f"Location '{location}' not found"}
        
        result = geo_data["results"][0]
        lat = result["latitude"]
        lon = result["longitude"]
        city_name = result.get("name", location)
        country = result.get("country", "")
        
        # Get weather data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code&temperature_unit={'fahrenheit' if unit == 'fahrenheit' else 'celsius'}"
        weather_response = requests.get(weather_url, timeout=5)
        weather_data = weather_response.json()
        
        current = weather_data.get("current", {})
        
        # Weather code to description mapping
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        
        weather_code = current.get("weather_code", 0)
        conditions = weather_codes.get(weather_code, "Unknown")
        
        return {
            "location": f"{city_name}, {country}",
            "temperature": current.get("temperature_2m"),
            "unit": unit,
            "conditions": conditions,
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "coordinates": {"latitude": lat, "longitude": lon}
        }
    except Exception as e:
        return {"error": f"Weather data fetch failed: {str(e)}", "location": location}


def calculate_math(expression: str) -> Dict[str, Any]:
    """
    Safely calculate a mathematical expression using Python's ast module.
    
    Args:
        expression: The mathematical expression to evaluate (e.g., "2 + 2", "15 * 23 + 47")
    
    Returns:
        The result of the calculation
    """
    try:
        # Safe operators
        safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        def eval_node(node):
            if isinstance(node, ast.Constant):  # Updated for Python 3.8+
                return node.value
            elif isinstance(node, ast.Num):  # Backwards compatibility
                return node.n
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                return safe_operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                return safe_operators[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported operation: {type(node).__name__}")
        
        tree = ast.parse(expression, mode='eval')
        result = eval_node(tree.body)
        
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "success": False
        }


def get_current_time(timezone: str = "UTC") -> Dict[str, Any]:
    """
    Get the real current time for any timezone.
    
    Args:
        timezone: The timezone name (e.g., "UTC", "America/New_York", "Europe/Istanbul", "Asia/Tokyo")
    
    Returns:
        Current time information
    """
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        
        return {
            "timezone": timezone,
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "iso_format": current_time.isoformat(),
            "timestamp": current_time.timestamp(),
            "day_of_week": current_time.strftime("%A"),
            "success": True
        }
    except pytz.exceptions.UnknownTimeZoneError:
        # Return available timezone examples
        return {
            "error": f"Unknown timezone: {timezone}",
            "examples": ["UTC", "America/New_York", "Europe/London", "Europe/Istanbul", "Asia/Tokyo", "Australia/Sydney"],
            "success": False
        }
    except Exception as e:
        return {
            "error": str(e),
            "timezone": timezone,
            "success": False
        }


def wikipedia_search(query: str) -> Dict[str, Any]:
    """
    Search Wikipedia and get article summary.
    
    Args:
        query: The topic to search for
    
    Returns:
        Wikipedia article summary and URL
    """
    try:
        # Wikipedia API
        search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(query)}"
        response = requests.get(search_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("title"),
                "summary": data.get("extract"),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "thumbnail": data.get("thumbnail", {}).get("source", "") if data.get("thumbnail") else None,
                "success": True
            }
        else:
            return {
                "error": f"Article not found: {query}",
                "success": False
            }
    except Exception as e:
        return {
            "error": str(e),
            "query": query,
            "success": False
        }


def get_exchange_rate(from_currency: str = "USD", to_currency: str = "EUR") -> Dict[str, Any]:
    """
    Get real-time exchange rates between currencies.
    
    Args:
        from_currency: Source currency code (e.g., "USD", "EUR", "TRY")
        to_currency: Target currency code (e.g., "EUR", "GBP", "USD")
    
    Returns:
        Exchange rate and conversion information
    """
    try:
        # Using free exchangerate API
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            rates = data.get("rates", {})
            
            to_curr = to_currency.upper()
            if to_curr in rates:
                rate = rates[to_curr]
                return {
                    "from": from_currency.upper(),
                    "to": to_curr,
                    "rate": rate,
                    "example": f"1 {from_currency.upper()} = {rate} {to_curr}",
                    "date": data.get("date"),
                    "success": True
                }
            else:
                return {
                    "error": f"Currency '{to_currency}' not found",
                    "available_currencies": list(rates.keys())[:20],
                    "success": False
                }
        else:
            return {"error": "Exchange rate API failed", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


# ============================================================================
# PART 2: Manual Tool Schema Definition (Educational)
# ============================================================================

# Manual schema definitions
GOOGLE_SEARCH_SCHEMA = {
    "name": "google_search",
    "description": "Tool used to perform real web searches and return ranked results with titles, URLs, and snippets.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default 5).",
            }
        },
        "required": ["query"],
    },
}

SCRAPE_URL_SCHEMA = {
    "name": "scrape_url",
    "description": "Tool used to scrape and clean HTML content from a web URL. Returns cleaned text content.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to scrape.",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum length of text to return (default 2000).",
            }
        },
        "required": ["url"],
    },
}

# Tool registry for manual approach
MANUAL_TOOLS = {
    "google_search": {
        "handler": google_search,
        "declaration": GOOGLE_SEARCH_SCHEMA,
    },
    "scrape_url": {
        "handler": scrape_url,
        "declaration": SCRAPE_URL_SCHEMA,
    },
}

MANUAL_TOOLS_BY_NAME = {tool_name: tool["handler"] for tool_name, tool in MANUAL_TOOLS.items()}
MANUAL_TOOLS_SCHEMA = [tool["declaration"] for tool in MANUAL_TOOLS.values()]

# System prompt for manual tool calling
TOOL_CALLING_SYSTEM_PROMPT = """
You are a helpful AI assistant with access to tools.

## Tool Usage Guidelines
- When you need to perform actions or retrieve information, choose the most appropriate tool.
- Choose different tools based on their descriptions.
- Provide all required parameters with accurate values.

## Tool Call Format
When you need to use a tool, output ONLY the tool call in this exact format:

<tool_call>
{{"name": "tool_name", "args": {{"param1": "value1"}}}}
</tool_call>

## Available Tools
<tool_definitions>
{tools}
</tool_definitions>
"""


# ============================================================================
# PART 3: Helper Functions for Manual Tool Calling
# ============================================================================

def extract_tool_call(response_text: str) -> str:
    """Extracts the tool call JSON from the response text."""
    # Handle both <tool_call> and ```tool_call> formats
    if "<tool_call>" in response_text and "</tool_call>" in response_text:
        try:
            start = response_text.find("<tool_call>") + len("<tool_call>")
            end = response_text.find("</tool_call>")
            return response_text[start:end].strip()
        except:
            pass
    
    # Try markdown code block format
    if "```tool_call>" in response_text or "```json" in response_text:
        try:
            # Find the JSON content in code blocks
            import re
            # Match ```tool_call> or ```json followed by JSON
            pattern = r'```(?:tool_call>|json)\s*(\{[^`]+\})'
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                return match.group(1).strip()
        except:
            pass
    
    return None


def call_tool_manual(response_text: str, tools_by_name: dict) -> Any:
    """Parses the LLM response and executes the requested tool (manual approach)."""
    tool_call_str = extract_tool_call(response_text)
    if not tool_call_str:
        return None
    
    tool_call = json.loads(tool_call_str)
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_handler = tools_by_name[tool_name]
    
    return tool_handler(**tool_args)


# ============================================================================
# PART 4: Production Tool Registry
# ============================================================================
# All available tools for the production approach

PRODUCTION_TOOLS_BY_NAME = {
    "google_search": google_search,
    "scrape_url": scrape_url,
    "get_current_weather": get_current_weather,
    "calculate_math": calculate_math,
    "get_current_time": get_current_time,
    "wikipedia_search": wikipedia_search,
    "get_exchange_rate": get_exchange_rate,
}


def call_tool_production(function_call, tools_by_name: dict = None) -> Any:
    """Execute a tool call from Gemini's native function_call object."""
    if tools_by_name is None:
        tools_by_name = PRODUCTION_TOOLS_BY_NAME
    
    tool_name = function_call.name
    tool_args = {key: value for key, value in function_call.args.items()}
    
    if tool_name not in tools_by_name:
        return {"error": f"Tool {tool_name} not found"}
    
    tool_handler = tools_by_name[tool_name]
    
    try:
        result = tool_handler(**tool_args)
        return result
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# PART 5: Manual Tool Calling Implementation (Educational)
# ============================================================================

def demo_manual_tool_calling():
    """Demonstrates manual tool calling with custom schemas and prompts."""
    console.print("\n")
    console.print(Panel.fit(
        "[bold yellow]📚 MANUAL TOOL CALLING APPROACH[/bold yellow]\n[dim](Educational - See how it works under the hood)[/dim]",
        border_style="yellow"
    ))
    console.print()
    
    client = configure_gemini()
    
    # Test Query: Simple search
    console.print(Panel(
        "[bold]Test: Simple Tool Call[/bold]\n"
        "[cyan]We'll ask the AI to search the web and see how it responds[/cyan]",
        border_style="yellow"
    ))
    console.print()
    
    user_prompt = "Search the web for 'Python programming language' and give me a summary."
    
    messages = [
        TOOL_CALLING_SYSTEM_PROMPT.format(tools=json.dumps(MANUAL_TOOLS_SCHEMA, indent=2)),
        user_prompt
    ]
    
    console.print(f"[bold cyan]User:[/bold cyan] {user_prompt}\n")
    
    with console.status("[bold yellow]Asking Gemini...", spinner="dots"):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="\n\n".join(messages),
        )
    
    # Show LLM response
    console.print(Panel(
        response.text,
        title="[bold yellow]🤖 LLM Response[/bold yellow]",
        border_style="yellow"
    ))
    console.print()
    
    # Execute the tool
    tool_result = call_tool_manual(response.text, MANUAL_TOOLS_BY_NAME)
    if tool_result:
        console.print("[green]✅ Tool executed successfully![/green]\n")
        
        # Pretty print results based on type
        if isinstance(tool_result, dict):
            if 'results' in tool_result and tool_result['results']:
                # Show search results in a table
                table = Table(title=f"Search Results for '{tool_result['query']}'", show_lines=True)
                table.add_column("#", style="cyan", width=3)
                table.add_column("Title", style="green")
                table.add_column("URL", style="blue", overflow="fold")
                
                for i, result in enumerate(tool_result['results'][:3], 1):  # Show top 3
                    table.add_row(
                        str(i),
                        result.get('title', 'N/A')[:60],
                        result.get('url', 'N/A')[:50]
                    )
                
                console.print(table)
                console.print()
            else:
                console.print(Panel(
                    Syntax(json.dumps(tool_result, indent=2, ensure_ascii=False), "json", theme="monokai"),
                    title="[bold]Tool Result[/bold]",
                    border_style="green"
                ))
    else:
        console.print("[yellow]⚠️ No tool call detected in response.[/yellow]\n")
    
    console.print("\n[dim]💡 This shows how manual tool calling works:[/dim]")
    console.print("[dim]   1. We define tool schemas manually[/dim]")
    console.print("[dim]   2. We write a custom system prompt[/dim]")
    console.print("[dim]   3. We parse the LLM's response[/dim]")
    console.print("[dim]   4. We execute the tool ourselves[/dim]")
    console.print("[dim]   5. We send results back to the LLM[/dim]\n")


# ============================================================================
# PART 6: Production Tool Calling Implementation (Recommended)
# ============================================================================

def demo_production_tool_calling():
    """Demonstrates production-level tool calling using native Gemini API."""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]PRODUCTION TOOL CALLING APPROACH[/bold cyan]\n[dim](Recommended)[/dim]",
        border_style="cyan"
    ))
    console.print()
    
    client = configure_gemini()
    
    # Create tools list from our functions
    tools = list(PRODUCTION_TOOLS_BY_NAME.values())
    
    # Configure with tools
    config = types.GenerateContentConfig(
        tools=tools,
        temperature=0.7,
    )
    
    # Test queries - reduced to avoid rate limits
    test_queries = [
        ("🌤️", "Weather", "What's the current weather in Istanbul?"),
        ("🧮", "Math", "Calculate 15 * 23 + 47"),
        ("🕐", "Time", "What time is it in Tokyo right now?"),
    ]
    
    for i, (icon, category, query) in enumerate(test_queries, 1):
        console.print(f"\n[bold blue]{'═' * 70}[/bold blue]")
        console.print(Panel(
            f"[bold]{icon} Test {i}: {category}[/bold]\n[cyan]{query}[/cyan]",
            border_style="blue"
        ))
        
        try:
            # Add small delay between requests to avoid rate limits
            if i > 1:
                import time
                time.sleep(2)
            
            # Send initial request with spinner
            with console.status(f"[bold green]Asking Gemini...", spinner="dots"):
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=query,
                    config=config,
                )
            
            # Handle tool calls iteratively
            messages = [query]
            max_iterations = 5
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                if not response.candidates or not response.candidates[0].content.parts:
                    break
                
                # Check for function calls
                parts = response.candidates[0].content.parts
                function_calls = [p for p in parts if hasattr(p, 'function_call') and p.function_call and hasattr(p.function_call, 'name') and p.function_call.name]
                
                if not function_calls:
                    # No more function calls, print final response
                    if hasattr(response, 'text') and response.text:
                        console.print()
                        console.print(Panel(
                            f"[green]{response.text}[/green]",
                            title="[bold green]✅ Response[/bold green]",
                            border_style="green"
                        ))
                    break
                
                # Process each function call
                for part in function_calls:
                    if hasattr(part, 'function_call'):
                        function_call = part.function_call
                        tool_name = function_call.name
                        tool_args = {k: v for k, v in function_call.args.items()}
                        
                        # Show tool call in a nice table
                        table = Table(show_header=False, box=None, padding=(0, 1))
                        table.add_column(style="bold yellow")
                        table.add_column()
                        table.add_row("🔧 Tool:", f"[cyan]{tool_name}[/cyan]")
                        for k, v in tool_args.items():
                            table.add_row(f"   {k}:", f"[dim]{v}[/dim]")
                        console.print(table)
                        
                        # Execute the function
                        with console.status(f"[bold yellow]Executing {tool_name}...", spinner="dots"):
                            tool_result = call_tool_production(function_call, PRODUCTION_TOOLS_BY_NAME)
                        
                        # Pretty print result
                        if isinstance(tool_result, dict):
                            # Create a nice display based on result type
                            if 'error' in tool_result:
                                console.print(f"   [red]❌ Error: {tool_result['error']}[/red]")
                            elif 'temperature' in tool_result:
                                console.print(f"   [green]✅ {tool_result.get('location')}: {tool_result['temperature']}°{tool_result['unit'][0].upper()}, {tool_result['conditions']}[/green]")
                            elif 'result' in tool_result:
                                console.print(f"   [green]✅ Result: {tool_result['result']}[/green]")
                            elif 'current_time' in tool_result:
                                console.print(f"   [green]✅ {tool_result['timezone']}: {tool_result['current_time']}[/green]")
                            elif 'results' in tool_result:
                                console.print(f"   [green]✅ Found {len(tool_result['results'])} results[/green]")
                            else:
                                console.print(f"   [green]✅ Success[/green]")
                        else:
                            console.print(f"   [green]✅ {str(tool_result)[:100]}[/green]")
                        
                        # Send result back to model
                        messages.append({
                            "role": "model",
                            "parts": [{"function_call": function_call}]
                        })
                        messages.append({
                            "role": "user",
                            "parts": [{
                                "function_response": {
                                    "name": tool_name,
                                    "response": {"result": tool_result}
                                }
                            }]
                        })
                
                # Get next response
                with console.status("[bold green]Generating response...", spinner="dots"):
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-exp",
                        contents=messages,
                        config=config,
                    )
        
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                console.print()
                console.print(Panel(
                    "[yellow]⚠️ Rate limit reached. Please wait a moment.[/yellow]\n\n"
                    "[dim]💡 Tip: Gemini free tier has 10 requests per minute.\n"
                    "   Try interactive mode (option 3) for single queries.[/dim]",
                    title="[bold yellow]Rate Limit[/bold yellow]",
                    border_style="yellow"
                ))
                break
            else:
                console.print(f"\n[red]❌ Error: {error_msg}[/red]\n")


def interactive_mode():
    """Interactive chat mode with production tool calling."""
    console.print("\n")
    console.print(Panel.fit(
        "[bold magenta]🤖 INTERACTIVE MODE[/bold magenta]\n"
        "[dim]Chat with AI using real tools • Type 'quit' to exit[/dim]",
        border_style="magenta"
    ))
    console.print()
    
    client = configure_gemini()
    tools = list(PRODUCTION_TOOLS_BY_NAME.values())
    config = types.GenerateContentConfig(tools=tools, temperature=0.7)
    
    while True:
        try:
            # Get user input with rich prompt
            console.print("[bold cyan]You:[/bold cyan] ", end="")
            user_input = input().strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("\n[bold green]👋 Goodbye![/bold green]\n")
                break
            
            if not user_input:
                continue
            
            console.print()
            
            # Send request with spinner
            with console.status("[bold green]🤔 Thinking...", spinner="dots"):
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=user_input,
                    config=config,
                )
            
            # Handle tool calls
            messages = [user_input]
            max_iterations = 5
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                if not response.candidates or not response.candidates[0].content.parts:
                    break
                
                parts = response.candidates[0].content.parts
                function_calls = [p for p in parts if hasattr(p, 'function_call') and p.function_call and hasattr(p.function_call, 'name') and p.function_call.name]
                
                if not function_calls:
                    if hasattr(response, 'text') and response.text:
                        console.print(Panel(
                            f"[white]{response.text}[/white]",
                            title="[bold green]🤖 Assistant[/bold green]",
                            border_style="green",
                            padding=(1, 2)
                        ))
                        console.print()
                    break
                
                # Execute function calls
                for part in function_calls:
                    if hasattr(part, 'function_call'):
                        function_call = part.function_call
                        tool_name = function_call.name
                        tool_args = {k: v for k, v in function_call.args.items()}
                        
                        # Show compact tool usage
                        args_str = ', '.join(f"{k}={v}" for k, v in tool_args.items())
                        console.print(f"[dim]🔧 Using: [cyan]{tool_name}[/cyan]({args_str})[/dim]")
                        
                        with console.status(f"[dim]Executing {tool_name}...", spinner="dots"):
                            tool_result = call_tool_production(function_call, PRODUCTION_TOOLS_BY_NAME)
                        
                        # Show brief result preview
                        if isinstance(tool_result, dict):
                            if 'error' in tool_result:
                                console.print(f"[dim]   ❌ {tool_result['error']}[/dim]")
                            elif 'content' in tool_result:
                                console.print(f"[dim]   ✅ Scraped: {len(str(tool_result['content']))} chars[/dim]")
                            elif 'results' in tool_result and len(tool_result['results']) > 0:
                                console.print(f"[dim]   ✅ Found {len(tool_result['results'])} results[/dim]")
                            elif 'temperature' in tool_result:
                                console.print(f"[dim]   ✅ {tool_result['temperature']}°{tool_result['unit'][0].upper()}, {tool_result['conditions']}[/dim]")
                            elif 'result' in tool_result:
                                console.print(f"[dim]   ✅ {tool_result['result']}[/dim]")
                            elif 'current_time' in tool_result:
                                console.print(f"[dim]   ✅ {tool_result['current_time']}[/dim]")
                            else:
                                console.print(f"[dim]   ✅ Done[/dim]")
                        else:
                            result_preview = str(tool_result)[:80]
                            console.print(f"[dim]   ✅ {result_preview}[/dim]")
                        
                        messages.append({
                            "role": "model",
                            "parts": [{"function_call": function_call}]
                        })
                        messages.append({
                            "role": "user",
                            "parts": [{
                                "function_response": {
                                    "name": tool_name,
                                    "response": {"result": tool_result}
                                }
                            }]
                        })
                
                # Get next response
                with console.status("[dim]Generating response...", spinner="dots"):
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-exp",
                        contents=messages,
                        config=config,
                    )
        
        except KeyboardInterrupt:
            console.print("\n\n[bold green]👋 Goodbye![/bold green]\n")
            break
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                console.print(Panel(
                    "[yellow]⚠️ Rate limit reached. Waiting 10 seconds...[/yellow]",
                    border_style="yellow"
                ))
                import time
                time.sleep(10)
            else:
                console.print(f"[red]❌ Error: {error_msg}[/red]\n")


def main():
    """Main function to demonstrate both approaches."""
    console.clear()
    
    # Create beautiful header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🤖 Tool Calling From Scratch[/bold cyan]\n"
        "[dim]Using Gemini 2.0 Flash with Real Working Tools[/dim]",
        border_style="bright_cyan",
        padding=(1, 4)
    ))
    console.print()
    
    # Create menu table
    menu_table = Table(show_header=False, box=None, padding=(0, 2))
    menu_table.add_column(style="bold yellow", justify="center")
    menu_table.add_column(style="white")
    
    menu_table.add_row("1️⃣", "[cyan]Manual Tool Calling Demo[/cyan]\n[dim]   Educational - see how it works under the hood[/dim]")
    menu_table.add_row("2️⃣", "[green]Production Tool Calling Demo[/green]\n[dim]   Recommended - using native Gemini API[/dim]")
    menu_table.add_row("3️⃣", "[magenta]Interactive Mode[/magenta]\n[dim]   Best for testing - chat with real tools[/dim]")
    menu_table.add_row("4️⃣", "[blue]Run All Demos[/blue]\n[dim]   See everything in action[/dim]")
    
    console.print(Panel(
        menu_table,
        title="[bold white]Choose Demo[/bold white]",
        border_style="bright_blue",
        padding=(1, 2)
    ))
    console.print()
    
    console.print("[bold]Enter your choice (1-4):[/bold] ", end="")
    choice = input().strip()
    
    if choice == "1":
        demo_manual_tool_calling()
    elif choice == "2":
        demo_production_tool_calling()
    elif choice == "3":
        interactive_mode()
    elif choice == "4":
        demo_manual_tool_calling()
        console.print("\n")
        console.print(Panel("[bold]Press Enter to continue to Production demo...[/bold]", border_style="dim"))
        input()
        demo_production_tool_calling()
        console.print("\n")
        console.print(Panel("[bold]Press Enter to start Interactive Mode...[/bold]", border_style="dim"))
        input()
        interactive_mode()
    else:
        console.print("[yellow]Invalid choice. Running Production demo by default...[/yellow]\n")
        demo_production_tool_calling()
        console.print("\n")
        console.print(Panel("[bold]Press Enter to start Interactive Mode...[/bold]", border_style="dim"))
        input()
        interactive_mode()


if __name__ == "__main__":
    main()
