# Tool Calling From Scratch - Gemini 2.0 Flash

This project demonstrates two approaches to implementing tool calling with Google's Gemini AI:

1. **Manual Approach** (Educational) - Shows how tool calling works under the hood
2. **Production Approach** (Recommended) - Uses native Gemini API for robust tool calling

## Based On

Article: [Implementing AI Agents from Scratch: Tools, Planning, and Memory](https://towardsdatascience.com/implementing-ai-agents-from-scratch-tools-planning-and-memory-68e954207b7a)

## Features

### Available Tools

The demo includes these **REAL WORKING** tools:

- `google_search` - Real web search using DuckDuckGo (no API key needed)
- `scrape_url` - Real web scraping with BeautifulSoup
- `get_current_weather` - Real weather data from Open-Meteo API (free)
- `calculate_math` - Safe mathematical expression evaluation
- `get_current_time` - Real time for any timezone worldwide
- `wikipedia_search` - Real Wikipedia article summaries
- `get_exchange_rate` - Real-time currency exchange rates

**All tools work without API keys!** (except Gemini API)

### Two Implementations

#### 1. Manual Tool Calling (Educational)
Shows the 5-step process:
1. Send LLM your task + tool schemas
2. LLM responds with function_call request
3. Parse and execute the function
4. Send function output back to LLM
5. LLM generates final user-facing response

#### 2. Production Tool Calling (Recommended)
Uses Gemini's native tool calling API:
- Automatic schema generation from Python functions
- Built-in function call parsing
- Cleaner, more robust code
- Less maintenance

## Installation

1. Install dependencies:

```cmd
pip install google-genai beautifulsoup4 requests pytz duckduckgo-search
```

Or use the requirements file:

```cmd
pip install -r requirements.txt
```

2. Set your Gemini API key:

```cmd
set GEMINI_API_KEY=your_api_key_here
```

Get your API key from: <https://aistudio.google.com/app/apikey>

**Note:** Only Gemini API key is needed. All other tools use free APIs!

## Usage

Run the application:
```cmd
python app.py
```

You'll see a menu with options:
- **1** - Manual Tool Calling Demo (Educational)
- **2** - Production Tool Calling Demo (Recommended)
- **3** - Interactive Mode (Chat with the AI)
- **4** - Run All Demos

### Example Queries

Try these in interactive mode:

- "What's the current weather in Istanbul?"
- "Calculate 15 * 23 + 47"
- "What time is it in Tokyo right now?"
- "Search for recent news about artificial intelligence"
- "What is the exchange rate from USD to TRY?"
- "Tell me about Python programming language"
- "Search for Gemini AI, then scrape the first result"

## How It Works

### Manual Approach Flow

```
User Query
    ↓
System Prompt + Tool Schemas
    ↓
LLM generates <tool_call>JSON</tool_call>
    ↓
Parse JSON and execute Python function
    ↓
Send result back to LLM
    ↓
LLM generates final response
```

### Production Approach Flow

```
User Query
    ↓
Gemini API with tools=[] config
    ↓
Gemini returns function_call object
    ↓
Execute function with call_tool_production()
    ↓
Send result via function_response message
    ↓
Gemini generates final response
```

## Code Structure

```
app.py
├── Part 1: Tool Functions (google_search, scrape_url, etc.)
├── Part 2: Manual Schema Definitions (educational)
├── Part 3: Manual Tool Calling Helpers
├── Part 4: Production Tool Registry
├── Part 5: Manual Demo Implementation
├── Part 6: Production Demo Implementation
└── Main Menu & Interactive Mode
```

## Key Concepts

### Tool Schema Format

Each tool needs:
- **name**: Unique identifier
- **description**: What the tool does (crucial for LLM to choose correctly)
- **parameters**: JSON Schema for function arguments

### Why Good Descriptions Matter

The LLM uses the `description` field to decide which tool to use. Compare:

❌ **Bad**: "Tool used to find information"
✅ **Good**: "Tool used to perform Google web searches and return ranked results"

Clear descriptions prevent the LLM from confusing similar tools.

### Multi-Step Tool Calling

The agent can:
1. Call a tool
2. Receive results
3. Decide next action
4. Call another tool
5. Synthesize final answer

Example: "Search for articles, then scrape the top result" → calls `google_search` → calls `scrape_url` → generates summary

## Advantages of Production Approach

| Manual | Production |
|--------|-----------|
| Define schemas manually | Auto-generate from functions |
| Write custom system prompts | Optimized by Gemini |
| Parse string responses | Native function_call objects |
| Maintain sync between code & schema | Single source of truth |

## Adding Your Own Tools

### For Production Approach (Easy)

1. Define your function with type hints and docstring:
```python
def my_tool(arg1: str, arg2: int) -> dict:
    """
    Clear description of what this tool does.
    
    Args:
        arg1: Description of argument 1
        arg2: Description of argument 2
    
    Returns:
        Description of return value
    """
    # Your implementation
    return {"result": "value"}
```

2. Add to registry:
```python
PRODUCTION_TOOLS_BY_NAME["my_tool"] = my_tool
```

That's it! The Gemini API automatically generates the schema.

### For Manual Approach

You'd need to:
1. Define the function
2. Create JSON schema
3. Add to MANUAL_TOOLS registry
4. Update system prompt

## Troubleshooting

### Import Error
If you get `Import "google.genai" could not be resolved`:
```cmd
pip install --upgrade google-genai
```

### API Key Error
Make sure to set the environment variable:
```cmd
set GEMINI_API_KEY=your_key_here
```

### Tool Not Found
Check that the tool is in `PRODUCTION_TOOLS_BY_NAME` dictionary.

## Resources

- [Google AI Studio](https://aistudio.google.com/) - Get API keys & test models
- [Gemini API Docs](https://ai.google.dev/gemini-api/docs) - Official documentation
- [google-genai Package](https://pypi.org/project/google-genai/) - Python SDK

## License

See LICENSE file in the main repository.

## Author

Based on the article by Towards Data Science, implemented for educational purposes.
