"""
Gelişmiş Ollama AI Agent
Bu script, Agno framework kullanarak gelişmiş AI agent özellikleri sunar:
- Memory & Session Management
- Interactive Chat Interface
- Structured Output Support
- Tools Integration
- Conversation History
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.db.sqlite import SqliteDb

console = Console()

class ChatMessage(BaseModel):
    """Structured chat message model"""
    timestamp: str = Field(description="Message timestamp")
    user_input: str = Field(description="User's input message")
    agent_response: str = Field(description="Agent's response")
    session_id: str = Field(description="Session identifier")

class AgentConfig(BaseModel):
    """Agent configuration model"""
    model_id: str = Field(default="llama3.2:3b", description="Ollama model ID")
    session_id: str = Field(description="Session identifier")
    db_path: str = Field(description="Database file path")
    max_history: int = Field(default=10, description="Maximum history messages to keep in context")
    enable_markdown: bool = Field(default=True, description="Enable markdown rendering")

class AdvancedOllamaAgent:
    """Advanced Ollama Agent with memory, tools, and interactive features"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.setup_database()
        self.setup_agent()
        
    def setup_database(self):
        """Setup SQLite database for memory storage"""
        db_dir = Path(self.config.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db = SqliteDb(db_file=self.config.db_path)
        
    def setup_agent(self):
        """Initialize the Agno agent with advanced features"""
        self.agent = Agent(
            model=Ollama(id=self.config.model_id),
            session_id=self.config.session_id,
            db=self.db,
            add_history_to_context=True,
            num_history_runs=self.config.max_history,
            markdown=self.config.enable_markdown,
            description="""You are an advanced AI assistant powered by Ollama. 
            You have access to conversation memory and can maintain context across sessions.
            Be helpful, creative, and engaging in your responses."""
        )
    
    def display_welcome(self):
        """Display welcome message and instructions"""
        welcome_text = f"""
# 🤖 Advanced Ollama AI Agent

**Model:** `{self.config.model_id}`
**Session ID:** `{self.config.session_id}`
**Database:** `{self.config.db_path}`

## Available Commands:
- **chat**: Start interactive conversation
- **history**: Show conversation history
- **clear**: Clear current session
- **stats**: Show session statistics
- **export**: Export chat history
- **quit/exit**: Exit the application

Type your message or use a command to get started!
        """
        console.print(Panel(Markdown(welcome_text), title="🚀 Welcome", border_style="blue"))
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        try:
            messages = self.agent.get_messages_for_session()
            return {
                "total_messages": len(messages),
                "session_id": self.config.session_id,
                "last_activity": datetime.now().isoformat(),
                "model": self.config.model_id
            }
        except Exception as e:
            return {"error": str(e)}
    
    def show_history(self, limit: int = 10):
        """Display conversation history - FIXED VERSION"""
        try:
            messages = self.agent.get_messages_for_session()
            if not messages:
                console.print("📝 No conversation history found.", style="yellow")
                return
            
            console.print(f"\n📚 **Last {limit} messages:**\n", style="bold blue")
            
            # Process messages safely - FIX for 'Message' object has no attribute 'get'
            for i, message in enumerate(messages[-limit:]):
                try:
                    # Safe message access - try multiple methods
                    if hasattr(message, 'get'):
                        role = message.get('role', 'unknown')
                        content = message.get('content', '')
                    elif hasattr(message, 'role') and hasattr(message, 'content'):
                        role = message.role
                        content = message.content
                    elif hasattr(message, '__dict__'):
                        msg_dict = message.__dict__
                        role = msg_dict.get('role', 'unknown')
                        content = msg_dict.get('content', str(message))
                    else:
                        role = 'unknown'
                        content = str(message)
                    
                    if role == 'user':
                        console.print(f"👤 **User:** {content}", style="green")
                    elif role == 'assistant':
                        content_short = content[:200] + '...' if len(content) > 200 else content
                        console.print(f"🤖 **Assistant:** {content_short}", style="blue")
                    else:
                        console.print(f"❓ **{role}:** {content}", style="yellow")
                    console.print()
                        
                except Exception as e:
                    console.print(f"❌ Error processing message {i}: {e}", style="red")
                
        except Exception as e:
            console.print(f"❌ Error retrieving history: {e}", style="red")
    
    def clear_session(self):
        """Clear current session history"""
        try:
            # Create new session ID
            self.config.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.setup_agent()
            console.print("✅ Session cleared! New session started.", style="green")
        except Exception as e:
            console.print(f"❌ Error clearing session: {e}", style="red")
    
    def export_history(self, filepath: Optional[str] = None):
        """Export chat history to JSON file - FIXED VERSION"""
        try:
            messages = self.agent.get_messages_for_session()
            if not messages:
                console.print("📝 No history to export.", style="yellow")
                return
            
            if not filepath:
                filepath = f"chat_history_{self.config.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # FIX for 'Object of type Message is not JSON serializable'
            # Safe message serialization
            serialized_messages = []
            for message in messages:
                try:
                    if hasattr(message, 'dict'):
                        serialized_messages.append(message.dict())
                    elif hasattr(message, 'to_dict'):
                        serialized_messages.append(message.to_dict())
                    elif hasattr(message, '__dict__'):
                        msg_dict = {}
                        for key, value in message.__dict__.items():
                            if isinstance(value, (str, int, float, bool, type(None))):
                                msg_dict[key] = value
                            elif isinstance(value, datetime):
                                msg_dict[key] = value.isoformat()
                            else:
                                msg_dict[key] = str(value)
                        serialized_messages.append(msg_dict)
                    else:
                        # Fallback for unknown message types
                        serialized_messages.append({
                            'content': str(message),
                            'type': type(message).__name__,
                            'timestamp': datetime.now().isoformat()
                        })
                except Exception as e:
                    console.print(f"⚠️  Warning: Could not serialize message: {e}", style="yellow")
                    serialized_messages.append({
                        'error': f"Serialization failed: {e}",
                        'type': type(message).__name__,
                        'timestamp': datetime.now().isoformat()
                    })
            
            export_data = {
                "session_id": self.config.session_id,
                "export_date": datetime.now().isoformat(),
                "model": self.config.model_id,
                "messages": serialized_messages,
                "message_count": len(messages),
                "stats": self.get_session_stats()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            console.print(f"✅ History exported to: `{filepath}`", style="green")
            console.print(f"📊 Exported {len(messages)} messages", style="cyan")
            
        except Exception as e:
            console.print(f"❌ Error exporting history: {e}", style="red")
    
    def interactive_chat(self):
        """Start interactive chat session"""
        console.print("\n💬 **Interactive Chat Started**", style="bold green")
        console.print("Type 'quit', 'exit', or press Ctrl+C to end the conversation.\n", style="dim")
        
        try:
            while True:
                # Get user input
                user_input = Prompt.ask("[bold green]You[/bold green]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("👋 Goodbye! Chat session ended.", style="blue")
                    break
                
                if not user_input.strip():
                    continue
                
                # Process commands
                if user_input.startswith('/'):
                    self.handle_command(user_input)
                    continue
                
                # Get agent response
                console.print("\n🤖 **Assistant:**", style="bold blue")
                try:
                    self.agent.print_response(user_input, stream=True)
                except Exception as e:
                    console.print(f"❌ Error: {e}", style="red")
                
                console.print()  # Add spacing
                
        except KeyboardInterrupt:
            console.print("\n👋 Chat interrupted. Goodbye!", style="blue")
    
    def handle_command(self, command: str):
        """Handle special commands during chat"""
        cmd = command.lower().strip()
        
        if cmd in ['/history', '/h']:
            self.show_history()
        elif cmd in ['/stats', '/s']:
            stats = self.get_session_stats()
            console.print(f"📊 **Session Stats:** {json.dumps(stats, indent=2)}", style="cyan")
        elif cmd in ['/clear', '/c']:
            self.clear_session()
        elif cmd in ['/export', '/e']:
            self.export_history()
        elif cmd in ['/help', '/?']:
            self.show_help()
        else:
            console.print(f"❓ Unknown command: {command}", style="yellow")
            console.print("Type `/help` for available commands.", style="dim")
    
    def show_help(self):
        """Show available commands"""
        help_text = """
## 💡 Available Commands:

- `/history` or `/h` - Show conversation history
- `/stats` or `/s` - Show session statistics  
- `/clear` or `/c` - Clear current session
- `/export` or `/e` - Export chat history
- `/help` or `/?` - Show this help message
        """
        console.print(Panel(Markdown(help_text), title="Help", border_style="cyan"))
    
    def run_single_query(self, query: str):
        """Run a single query and return result"""
        console.print(f"\n🤖 **Processing:** {query}\n", style="bold blue")
        try:
            self.agent.print_response(query, stream=True)
            console.print()
        except Exception as e:
            console.print(f"❌ Error: {e}", style="red")
    
    def run(self, mode: str = "interactive"):
        """Run the agent in specified mode"""
        self.display_welcome()
        
        if mode == "interactive":
            self.interactive_chat()
        elif mode == "demo":
            # Run some demo queries
            demo_queries = [
                "Merhaba! Kendini tanıt.",
                "Türkiye'nin başkenti neresi?",
                "Bana kısa bir bilim kurgu hikayesi anlat.",
                "Son sorduğum soru neydi?"
            ]
            
            for query in demo_queries:
                self.run_single_query(query)
                input("\n⏸️  Press Enter to continue...")
        else:
            console.print("❓ Unknown mode. Use 'interactive' or 'demo'.", style="red")

def create_default_config() -> AgentConfig:
    """Create default agent configuration"""
    session_id = f"ollama_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db_path = "tmp/ollama_agent.db"
    
    return AgentConfig(
        session_id=session_id,
        db_path=db_path,
        model_id="llama3.2:3b",
        max_history=15,
        enable_markdown=True
    )

def main():
    """Main function to run the advanced agent"""
    try:
        # Create configuration
        config = create_default_config()
        
        # Initialize agent
        agent = AdvancedOllamaAgent(config)
        
        # Check if we should run in demo mode
        if len(os.sys.argv) > 1 and os.sys.argv[1] == "--demo":
            agent.run(mode="demo")
        else:
            agent.run(mode="interactive")
            
    except KeyboardInterrupt:
        console.print("\n👋 Application terminated by user.", style="blue")
    except Exception as e:
        console.print(f"❌ Fatal error: {e}", style="red")

if __name__ == "__main__":
    main()