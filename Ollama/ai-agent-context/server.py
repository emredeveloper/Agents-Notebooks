"""
6 Types of Context for AI Agents - Ollama Backend
Model: lfm2-8B-A1B:latest

Implements all 6 context types:
1. Instructions (Role, Objective, Requirements)
2. Tool Results
3. Tools (Parameters, Description)
4. Memory (Long-term, Short-term)
5. Knowledge (External Context, Task Context)
6. Examples (Behavior, Responses)
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="AI Agent Context Manager", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "lfm2-8B-A1B:latest"

# ===== Data Storage =====
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Storage files
MEMORY_FILE = DATA_DIR / "long_term_memory.json"
KNOWLEDGE_FILE = DATA_DIR / "knowledge_base.json"
TOOLS_FILE = DATA_DIR / "tools_registry.json"
EXAMPLES_FILE = DATA_DIR / "examples.json"
INSTRUCTIONS_FILE = DATA_DIR / "instructions.json"


def load_json(filepath: Path, default=None):
    if default is None:
        default = {}
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(filepath: Path, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===== Models =====
class InstructionsConfig(BaseModel):
    role: str = "AI Assistant"
    role_description: str = "You are a helpful AI assistant."
    objective: str = "Help users with their questions and tasks."
    objective_motivation: str = "Provide accurate and helpful responses."
    requirements_steps: list[str] = ["Reason step by step", "Provide clear answers"]
    requirements_conventions: list[str] = ["Be polite", "Use clear language"]
    requirements_constraints: list[str] = ["Do not make up information"]
    requirements_response_format: str = "plain text"


class ToolDefinition(BaseModel):
    name: str
    description: str
    what_it_does: str
    how_to_use: str
    return_value: str
    parameters: list[dict] = []


class MemoryEntry(BaseModel):
    type: str  # semantic, episodic, procedural
    content: str
    metadata: dict = {}


class KnowledgeEntry(BaseModel):
    category: str  # domain, system, workflow, documents, structured_data
    title: str
    content: str


class ExampleEntry(BaseModel):
    type: str  # behavior_positive, behavior_negative, response_positive, response_negative
    input_text: str
    output_text: str
    description: str = ""


class ChatMessage(BaseModel):
    message: str
    context_config: dict = {}


# ===== Context Builder =====
class ContextBuilder:
    """Builds the complete context from all 6 types for the AI agent."""

    def __init__(self):
        self.reload()

    def reload(self):
        self.instructions = load_json(INSTRUCTIONS_FILE, {
            "role": "AI Assistant",
            "role_description": "You are a helpful and knowledgeable AI assistant.",
            "objective": "Help users with their questions and tasks effectively.",
            "objective_motivation": "Provide accurate, detailed, and helpful responses.",
            "requirements_steps": ["Reason step by step", "Provide clear and structured answers"],
            "requirements_conventions": ["Be polite and professional", "Use clear language"],
            "requirements_constraints": ["Do not make up information", "Acknowledge uncertainty"],
            "requirements_response_format": "plain text"
        })
        self.tools = load_json(TOOLS_FILE, {"tools": []})
        self.memory = load_json(MEMORY_FILE, {"long_term": [], "short_term": []})
        self.knowledge = load_json(KNOWLEDGE_FILE, {"entries": []})
        self.examples = load_json(EXAMPLES_FILE, {"entries": []})

    def build_system_prompt(self, context_config: dict = None) -> str:
        """Build the complete system prompt from all context types."""
        if context_config is None:
            context_config = {}

        parts = []

        # 1. INSTRUCTIONS
        if context_config.get("instructions", True):
            parts.append(self._build_instructions())

        # 2. TOOLS
        if context_config.get("tools", True):
            tools_section = self._build_tools()
            if tools_section:
                parts.append(tools_section)

        # 3. MEMORY (Long-term)
        if context_config.get("memory", True):
            memory_section = self._build_memory()
            if memory_section:
                parts.append(memory_section)

        # 4. KNOWLEDGE
        if context_config.get("knowledge", True):
            knowledge_section = self._build_knowledge()
            if knowledge_section:
                parts.append(knowledge_section)

        # 5. EXAMPLES
        if context_config.get("examples", True):
            examples_section = self._build_examples()
            if examples_section:
                parts.append(examples_section)

        return "\n\n---\n\n".join(parts)

    def _build_instructions(self) -> str:
        inst = self.instructions
        lines = []
        lines.append("# INSTRUCTIONS")
        lines.append("")
        lines.append(f"## Role: {inst.get('role', 'AI Assistant')}")
        lines.append(inst.get('role_description', ''))
        lines.append("")
        lines.append(f"## Objective")
        lines.append(inst.get('objective', ''))
        lines.append(f"Motivation: {inst.get('objective_motivation', '')}")
        lines.append("")
        lines.append("## Requirements")

        steps = inst.get('requirements_steps', [])
        if steps:
            lines.append("### Steps:")
            for s in steps:
                lines.append(f"- {s}")

        conventions = inst.get('requirements_conventions', [])
        if conventions:
            lines.append("### Conventions:")
            for c in conventions:
                lines.append(f"- {c}")

        constraints = inst.get('requirements_constraints', [])
        if constraints:
            lines.append("### Constraints:")
            for c in constraints:
                lines.append(f"- {c}")

        fmt = inst.get('requirements_response_format', 'plain text')
        lines.append(f"### Response Format: {fmt}")

        return "\n".join(lines)

    def _build_tools(self) -> str:
        tools = self.tools.get("tools", [])
        if not tools:
            return ""

        lines = ["# AVAILABLE TOOLS", ""]
        for tool in tools:
            lines.append(f"## Tool: {tool['name']}")
            lines.append(f"**Description:** {tool.get('description', '')}")
            lines.append(f"**What it does:** {tool.get('what_it_does', '')}")
            lines.append(f"**How to use:** {tool.get('how_to_use', '')}")
            lines.append(f"**Return value:** {tool.get('return_value', '')}")

            params = tool.get('parameters', [])
            if params:
                lines.append("**Parameters:**")
                for p in params:
                    required = "Required" if p.get('is_required', False) else "Optional"
                    lines.append(f"  - `{p.get('name', '')}` ({p.get('type', 'string')}, {required}): {p.get('description', '')}")
            lines.append("")

        return "\n".join(lines)

    def _build_memory(self) -> str:
        long_term = self.memory.get("long_term", [])
        if not long_term:
            return ""

        lines = ["# MEMORY (Long-term Knowledge)", ""]

        semantic = [m for m in long_term if m.get('type') == 'semantic']
        episodic = [m for m in long_term if m.get('type') == 'episodic']
        procedural = [m for m in long_term if m.get('type') == 'procedural']

        if semantic:
            lines.append("## Semantic Memory (Facts, Preferences, User Knowledge)")
            for m in semantic:
                lines.append(f"- {m['content']}")
            lines.append("")

        if episodic:
            lines.append("## Episodic Memory (Past Experiences & Interactions)")
            for m in episodic:
                lines.append(f"- {m['content']}")
            lines.append("")

        if procedural:
            lines.append("## Procedural Memory (Instructions from Previous Interactions)")
            for m in procedural:
                lines.append(f"- {m['content']}")
            lines.append("")

        return "\n".join(lines)

    def _build_knowledge(self) -> str:
        entries = self.knowledge.get("entries", [])
        if not entries:
            return ""

        lines = ["# KNOWLEDGE BASE", ""]

        categories = {}
        for e in entries:
            cat = e.get('category', 'general')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(e)

        category_labels = {
            "domain": "Domain Knowledge (Strategy, Business Model, Market Facts)",
            "system": "System Knowledge (Overall Goals, Other Agents/Services)",
            "workflow": "Workflow (Process Steps, Role Guides, Hands-off)",
            "documents": "Documents (Specs, Procedures, Tickets, Logs)",
            "structured_data": "Structured Data (Variables, Tables, JSON/XML)"
        }

        for cat, items in categories.items():
            label = category_labels.get(cat, cat.title())
            lines.append(f"## {label}")
            for item in items:
                lines.append(f"### {item.get('title', 'Untitled')}")
                lines.append(item.get('content', ''))
                lines.append("")

        return "\n".join(lines)

    def _build_examples(self) -> str:
        entries = self.examples.get("entries", [])
        if not entries:
            return ""

        lines = ["# EXAMPLES", ""]

        behavior_pos = [e for e in entries if e.get('type') == 'behavior_positive']
        behavior_neg = [e for e in entries if e.get('type') == 'behavior_negative']
        response_pos = [e for e in entries if e.get('type') == 'response_positive']
        response_neg = [e for e in entries if e.get('type') == 'response_negative']

        if behavior_pos:
            lines.append("## Positive Behavior Examples")
            for e in behavior_pos:
                lines.append(f"**Input:** {e['input_text']}")
                lines.append(f"**Expected Output:** {e['output_text']}")
                if e.get('description'):
                    lines.append(f"*Note: {e['description']}*")
                lines.append("")

        if behavior_neg:
            lines.append("## Negative Behavior Examples (What NOT to do)")
            for e in behavior_neg:
                lines.append(f"**Input:** {e['input_text']}")
                lines.append(f"**Bad Output:** {e['output_text']}")
                if e.get('description'):
                    lines.append(f"*Note: {e['description']}*")
                lines.append("")

        if response_pos:
            lines.append("## Positive Response Examples")
            for e in response_pos:
                lines.append(f"**Input:** {e['input_text']}")
                lines.append(f"**Good Response:** {e['output_text']}")
                lines.append("")

        if response_neg:
            lines.append("## Negative Response Examples (Avoid these)")
            for e in response_neg:
                lines.append(f"**Input:** {e['input_text']}")
                lines.append(f"**Bad Response:** {e['output_text']}")
                lines.append("")

        return "\n".join(lines)


context_builder = ContextBuilder()

# Short-term memory (session-based conversation history)
conversation_histories: dict[str, list[dict]] = {}


# ===== API Endpoints =====

# --- Health & Status ---
@app.get("/api/health")
async def health_check():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            model_available = MODEL_NAME in model_names
    except Exception:
        model_available = False

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_available": model_available,
        "timestamp": datetime.now().isoformat()
    }


# --- 1. Instructions ---
@app.get("/api/instructions")
async def get_instructions():
    context_builder.reload()
    return context_builder.instructions


@app.post("/api/instructions")
async def update_instructions(config: InstructionsConfig):
    data = config.model_dump()
    save_json(INSTRUCTIONS_FILE, data)
    context_builder.reload()
    return {"status": "saved", "data": data}


# --- 2. Tools ---
@app.get("/api/tools")
async def get_tools():
    context_builder.reload()
    return context_builder.tools


@app.get("/api/tool-results")
async def get_tool_results():
    """Retrieve the log of executed tools"""
    return {"results": tool_results_log}


@app.post("/api/tools")
async def add_tool(tool: ToolDefinition):
    context_builder.reload()
    tools = context_builder.tools
    tool_data = tool.model_dump()
    tool_data["id"] = str(uuid.uuid4())[:8]
    tools.setdefault("tools", []).append(tool_data)
    save_json(TOOLS_FILE, tools)
    context_builder.reload()
    return {"status": "added", "tool": tool_data}


@app.delete("/api/tools/{tool_id}")
async def delete_tool(tool_id: str):
    context_builder.reload()
    tools = context_builder.tools
    tools["tools"] = [t for t in tools.get("tools", []) if t.get("id") != tool_id]
    save_json(TOOLS_FILE, tools)
    context_builder.reload()
    return {"status": "deleted"}


# --- 3. Memory ---
@app.get("/api/memory")
async def get_memory():
    context_builder.reload()
    return context_builder.memory


@app.post("/api/memory")
async def add_memory(entry: MemoryEntry):
    context_builder.reload()
    memory = context_builder.memory
    entry_data = entry.model_dump()
    entry_data["id"] = str(uuid.uuid4())[:8]
    entry_data["timestamp"] = datetime.now().isoformat()
    memory.setdefault("long_term", []).append(entry_data)
    save_json(MEMORY_FILE, memory)
    context_builder.reload()
    return {"status": "added", "entry": entry_data}


@app.delete("/api/memory/{entry_id}")
async def delete_memory(entry_id: str):
    context_builder.reload()
    memory = context_builder.memory
    memory["long_term"] = [m for m in memory.get("long_term", []) if m.get("id") != entry_id]
    save_json(MEMORY_FILE, memory)
    context_builder.reload()
    return {"status": "deleted"}


# --- 4. Knowledge ---
@app.get("/api/knowledge")
async def get_knowledge():
    context_builder.reload()
    return context_builder.knowledge


@app.post("/api/knowledge")
async def add_knowledge(entry: KnowledgeEntry):
    context_builder.reload()
    knowledge = context_builder.knowledge
    entry_data = entry.model_dump()
    entry_data["id"] = str(uuid.uuid4())[:8]
    entry_data["timestamp"] = datetime.now().isoformat()
    knowledge.setdefault("entries", []).append(entry_data)
    save_json(KNOWLEDGE_FILE, knowledge)
    context_builder.reload()
    return {"status": "added", "entry": entry_data}


@app.delete("/api/knowledge/{entry_id}")
async def delete_knowledge(entry_id: str):
    context_builder.reload()
    knowledge = context_builder.knowledge
    knowledge["entries"] = [k for k in knowledge.get("entries", []) if k.get("id") != entry_id]
    save_json(KNOWLEDGE_FILE, knowledge)
    context_builder.reload()
    return {"status": "deleted"}


# --- 5. Examples ---
@app.get("/api/examples")
async def get_examples():
    context_builder.reload()
    return context_builder.examples


@app.post("/api/examples")
async def add_example(entry: ExampleEntry):
    context_builder.reload()
    examples = context_builder.examples
    entry_data = entry.model_dump()
    entry_data["id"] = str(uuid.uuid4())[:8]
    examples.setdefault("entries", []).append(entry_data)
    save_json(EXAMPLES_FILE, examples)
    context_builder.reload()
    return {"status": "added", "entry": entry_data}


@app.delete("/api/examples/{entry_id}")
async def delete_example(entry_id: str):
    context_builder.reload()
    examples = context_builder.examples
    examples["entries"] = [e for e in examples.get("entries", []) if e.get("id") != entry_id]
    save_json(EXAMPLES_FILE, examples)
    context_builder.reload()
    return {"status": "deleted"}


# --- 6. Tool Results (generated during chat) ---
# Tool results are part of the conversation flow

# ===== Real Tool Implementations =====
def tool_web_search(query: str):
    """Simulates a web search with realistic-looking results."""
    print(f"Executing REAL web_search for: {query}")
    # In a real app, this could use DuckDuckGo or Google API
    return [
        {"title": f"The definitive guide to {query}", "url": "https://tech-library.com/guide", "snippet": f"This comprehensive article explores {query} in depth..."},
        {"title": f"Top 10 trends in {query} for 2026", "url": "https://news-now.com/trends", "snippet": "Recent developments have shown that..."},
        {"title": "Wikipedia - " + query, "url": "https://wikipedia.org/wiki/" + query.replace(' ', '_'), "snippet": f"In various contexts, {query} refers to the concept of..."}
    ]

def tool_calculate_revenue(monthly_base: float, growth_rate: float):
    """Calculates compounded yearly revenue."""
    print(f"Executing REAL calculate_revenue for: {monthly_base}, {growth_rate}")
    yearly = []
    current = monthly_base
    for i in range(12):
        yearly.append(current)
        current *= (1 + (growth_rate / 100))
    total = sum(yearly)
    return {
        "monthly_breakdown": [round(m, 2) for m in yearly],
        "total_yearly_revenue": round(total, 2),
        "average_monthly": round(total / 12, 2)
    }

TOOL_REGISTRY = {
    "web_search": tool_web_search,
    "calculate_revenue": tool_calculate_revenue
}

# New global for tool results logging
tool_results_log = []

# --- Context Preview ---
@app.get("/api/tool-results")
async def get_tool_results():
    return {"results": tool_results_log}

@app.post("/api/context/preview")
async def preview_context(config: dict = {}):
    """Preview the full context that will be sent to the model."""
    context_builder.reload()
    system_prompt = context_builder.build_system_prompt(config)
    return {
        "system_prompt": system_prompt,
        "char_count": len(system_prompt),
        "token_estimate": len(system_prompt) // 4
    }


# --- Chat with Orchestration ---
@app.post("/api/chat")
async def chat(msg: ChatMessage):
    context_builder.reload()
    system_prompt = context_builder.build_system_prompt(msg.context_config)

    session_id = msg.context_config.get("session_id", "default")
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    history = conversation_histories[session_id]
    history.append({"role": "user", "content": msg.message})

    # Initial Messages
    messages = [{"role": "system", "content": system_prompt + "\n\nCRITICAL: If you need to use a tool, output ONLY a JSON block like: {\"tool\": \"tool_name\", \"params\": {\"param1\": \"value\"}}. Do not say anything else if you use a tool."}]
    messages.extend(history[-20:])

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Step 1: Send to LLM
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.2} # Lower temperature for tool calls
                }
            )
            result = response.json()
            assistant_message = result.get("message", {}).get("content", "").strip()

            # Step 2: Orchestration Layer - Robust Tool Call Detection
            try:
                # Standard Python regex to find JSON blocks containing "tool"
                import re
                # This finds { ... "tool": "any_name" ... } even across lines
                matches = re.findall(r'(\{\s*"tool":\s*"[^"]+".*?\})', assistant_message, re.DOTALL)
                
                # If the above fails, try a broader catch for any JSON block that looks like a tool call
                if not matches:
                    matches = re.findall(r'(\{.*?\"tool\".*?\})', assistant_message, re.DOTALL)
                
                if matches:
                    tool_json = matches[0]
                    print(f"DEBUG: Found potential tool call: {tool_json}")
                    try:
                        tool_data = json.loads(tool_json)
                        tool_name = tool_data.get("tool")
                        params = tool_data.get("params") or tool_data.get("parameters") or {}
                        
                        if tool_name in TOOL_REGISTRY:
                            print(f"DEBUG: Executing tool '{tool_name}' with params: {params}")
                            # EXECUTE
                            results = TOOL_REGISTRY[tool_name](**params)
                            
                            # LOG FOR UI
                            log_entry = {
                                "id": str(uuid.uuid4())[:8],
                                "tool": tool_name,
                                "params": params,
                                "result": results,
                                "timestamp": datetime.now().isoformat()
                            }
                            tool_results_log.append(log_entry)
                            
                            # Step 3: Feed Tool Results back to LLM
                            messages.append({"role": "assistant", "content": assistant_message})
                            messages.append({"role": "user", "content": f"SYSTEM: Tool '{tool_name}' result: {json.dumps(results)}"})
                            
                            # Final call for narrative
                            final_response = await client.post(
                                f"{OLLAMA_BASE_URL}/api/chat",
                                json={
                                    "model": MODEL_NAME,
                                    "messages": messages,
                                    "stream": False,
                                    "options": {"temperature": 0.3}
                                }
                            )
                            final_result = final_response.json()
                            assistant_message = final_result.get("message", {}).get("content", "Error processing tool results")
                        else:
                            print(f"DEBUG: Tool '{tool_name}' not found in registry.")
                    except json.JSONDecodeError as je:
                        print(f"DEBUG: JSON decode failed: {je}")
            except Exception as e:
                print(f"Orchestration fatal error: {e}")

            # Add final response to history
            history.append({"role": "assistant", "content": assistant_message})

            return {
                "response": assistant_message,
                "model": MODEL_NAME,
                "context_used": {
                    "total_messages": len(messages)
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(msg: ChatMessage):
    """Streaming chat endpoint"""
    from fastapi.responses import StreamingResponse

    context_builder.reload()
    system_prompt = context_builder.build_system_prompt(msg.context_config)

    session_id = msg.context_config.get("session_id", "default")
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    history = conversation_histories[session_id]
    history.append({"role": "user", "content": msg.message})

    messages = [{"role": "system", "content": system_prompt}]
    max_history = 20
    messages.extend(history[-max_history:])

    async def generate():
        full_response = ""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "stream": True,
                        "options": {"temperature": 0.7, "top_p": 0.9}
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                token = data.get("message", {}).get("content", "")
                                full_response += token
                                yield f"data: {json.dumps({'token': token})}\n\n"
                                if data.get("done", False):
                                    yield f"data: {json.dumps({'done': True})}\n\n"
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        history.append({"role": "assistant", "content": full_response})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/chat/history/{session_id}")
async def clear_history(session_id: str):
    if session_id in conversation_histories:
        del conversation_histories[session_id]
    return {"status": "cleared"}


# --- Serve static files ---
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Static files not found. Place index.html in /static/"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
