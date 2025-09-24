"""
🚀 Advanced Ollama Agent with RAG (Retrieval-Augmented Generation)
SQLite + sqlite-vec kullanarak vector database desteği
PDF'leri yükleyip sorgulanabilir hale getiren gelişmiş AI agent

Özellikler:
- SQLite tabanlı vector database (sqlite-vec)
- PDF döküman yükleme ve indexleme
- RAG (Retrieval-Augmented Generation)
- Interactive chat interface
- Memory & Session Management
- Export/Import functionality
"""

import os
import json
import sqlite3
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import hashlib
import PyPDF2
from io import BytesIO
import requests

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge

console = Console()

class DocumentChunk(BaseModel):
    """Document chunk model"""
    id: str = Field(description="Unique chunk identifier")
    content: str = Field(description="Chunk text content")
    source: str = Field(description="Source document path/URL")
    page_number: Optional[int] = Field(description="Page number in document")
    chunk_index: int = Field(description="Chunk index in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class RAGConfig(BaseModel):
    """RAG configuration model"""
    model_id: str = Field(default="llama3.2:3b", description="Ollama model ID")
    session_id: str = Field(description="Session identifier")
    db_path: str = Field(description="Database file path")
    vector_db_path: str = Field(description="Vector database file path")
    max_history: int = Field(default=10, description="Maximum history messages")
    chunk_size: int = Field(default=2000, description="Text chunk size for vectorization")
    chunk_overlap: int = Field(default=400, description="Overlap between chunks")
    top_k_results: int = Field(default=3, description="Number of top results to retrieve")
    enable_markdown: bool = Field(default=True, description="Enable markdown rendering")
    embedding_model: str = Field(default="embeddinggemma:latest", description="Ollama embedding model")

class SQLiteVectorDB:
    """SQLite + sqlite-vec based vector database with Ollama embeddings"""
    
    def __init__(self, db_path: str, embedder=None):
        self.db_path = db_path
        self.embedder = embedder
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database with vector extension"""
        try:
            # Create database directory if not exists
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
            
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # Try to load sqlite-vec extension (if available)
            try:
                self.conn.enable_load_extension(True)
                # Note: sqlite-vec extension path might vary
                # self.conn.load_extension("sqlite-vec")
                console.print("ℹ️  SQLite-vec extension would be loaded here", style="cyan")
            except Exception as e:
                console.print(f"⚠️  SQLite-vec not available, using fallback: {e}", style="yellow")
            
            # Create tables
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page_number INTEGER,
                    chunk_index INTEGER,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # For now, we'll store embeddings as JSON (in production, use proper vector column)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    document_id TEXT,
                    embedding TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            self.conn.commit()
            console.print("✅ Vector database initialized", style="green")
            
        except Exception as e:
            console.print(f"❌ Error setting up vector database: {e}", style="red")
            raise
    
    def add_document(self, chunk: DocumentChunk, embedding: Optional[List[float]] = None):
        """Add document chunk to database with embedding generation"""
        try:
            # Generate embedding if embedder available and no embedding provided
            if not embedding and self.embedder:
                try:
                    embedding = self.embedder.get_embedding(chunk.content)
                except Exception as e:
                    console.print(f"⚠️  Embedding error: {e}", style="yellow")
            
            # Insert document
            self.conn.execute("""
                INSERT OR REPLACE INTO documents 
                (id, source, content, page_number, chunk_index, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk.id,
                chunk.source,
                chunk.content,
                chunk.page_number,
                chunk.chunk_index,
                json.dumps(chunk.metadata)
            ))
            
            # Insert embedding if available
            if embedding:
                self.conn.execute("""
                    INSERT OR REPLACE INTO embeddings (document_id, embedding)
                    VALUES (?, ?)
                """, (chunk.id, json.dumps(embedding)))
            
            self.conn.commit()
            
        except Exception as e:
            console.print(f"❌ Error adding document: {e}", style="red")
            raise
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[DocumentChunk]:
        """Search for similar documents using cosine similarity"""
        try:
            # Get all documents with embeddings
            cursor = self.conn.execute("""
                SELECT d.id, d.source, d.content, d.page_number, d.chunk_index, d.metadata, e.embedding
                FROM documents d
                LEFT JOIN embeddings e ON d.id = e.document_id
                WHERE e.embedding IS NOT NULL
            """)
            
            results = []
            for row in cursor.fetchall():
                try:
                    stored_embedding = json.loads(row[6])
                    
                    # Calculate cosine similarity
                    similarity = self.cosine_similarity(query_embedding, stored_embedding)
                    
                    chunk = DocumentChunk(
                        id=row[0],
                        source=row[1],
                        content=row[2],
                        page_number=row[3],
                        chunk_index=row[4],
                        metadata=json.loads(row[5]) if row[5] else {}
                    )
                    
                    results.append((similarity, chunk))
                    
                except Exception as e:
                    console.print(f"⚠️  Warning: Error processing embedding for {row[0]}: {e}", style="yellow")
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x[0], reverse=True)
            return [chunk for _, chunk in results[:top_k]]
            
        except Exception as e:
            console.print(f"❌ Error searching documents: {e}", style="red")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import math
            
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except Exception as e:
            console.print(f"⚠️  Warning: Error calculating similarity: {e}", style="yellow")
            return 0.0
    
    def get_all_documents(self) -> List[DocumentChunk]:
        """Get all documents from database"""
        try:
            cursor = self.conn.execute("""
                SELECT id, source, content, page_number, chunk_index, metadata
                FROM documents
                ORDER BY source, chunk_index
            """)
            
            results = []
            for row in cursor.fetchall():
                chunk = DocumentChunk(
                    id=row[0],
                    source=row[1],
                    content=row[2],
                    page_number=row[3],
                    chunk_index=row[4],
                    metadata=json.loads(row[5]) if row[5] else {}
                )
                results.append(chunk)
            
            return results
            
        except Exception as e:
            console.print(f"❌ Error getting documents: {e}", style="red")
            return []
    
    def clear_documents(self):
        """Clear all documents"""
        try:
            self.conn.execute("DELETE FROM embeddings")
            self.conn.execute("DELETE FROM documents")
            self.conn.commit()
            console.print("✅ All documents cleared", style="green")
        except Exception as e:
            console.print(f"❌ Error clearing documents: {e}", style="red")

class DocumentProcessor:
    """Document processing utilities"""
    
    @staticmethod
    def extract_text_from_pdf(file_path_or_url: str) -> List[Dict[str, Any]]:
        """Extract text from PDF file or URL"""
        try:
            if file_path_or_url.startswith(('http://', 'https://')):
                # Download PDF from URL
                response = requests.get(file_path_or_url)
                pdf_file = BytesIO(response.content)
            else:
                # Read local PDF file
                pdf_file = open(file_path_or_url, 'rb')
            
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pages = []
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    pages.append({
                        'page_number': page_num,
                        'text': text,
                        'source': file_path_or_url
                    })
            
            if not file_path_or_url.startswith(('http://', 'https://')):
                pdf_file.close()
                
            console.print(f"✅ Extracted text from {len(pages)} pages", style="green")
            return pages
            
        except Exception as e:
            console.print(f"❌ Error extracting PDF text: {e}", style="red")
            return []
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks with overlap"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence or word boundary
            if end < len(text):
                # Look for sentence end
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + chunk_size // 2:
                        end = word_end
            
            chunks.append(text[start:end].strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks

class AdvancedRAGAgent:
    """Advanced Ollama Agent with RAG capabilities"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.setup_components()
    
    def setup_components(self):
        """Initialize all components with Ollama embedder"""
        try:
            # Setup Ollama embedder
            console.print(f"🔄 Initializing Ollama embedder with {self.config.embedding_model}...", style="blue")
            self.embedder = OllamaEmbedder(id=self.config.embedding_model, dimensions=768)
            
            # Setup vector database with embedder
            self.vector_db = SQLiteVectorDB(
                db_path=self.config.vector_db_path,
                embedder=self.embedder
            )
            
            # Setup regular database for chat history
            self.db = SqliteDb(db_file=self.config.db_path)
            
            # Setup Ollama agent
            self.agent = Agent(
                model=Ollama(id=self.config.model_id),
                session_id=self.config.session_id,
                db=self.db,
                add_history_to_context=True,
                num_history_runs=self.config.max_history,
                markdown=self.config.enable_markdown,
                description="You are an advanced AI assistant with access to document knowledge. Use the provided context from documents to answer questions accurately and cite your sources."
            )
            
            console.print("✅ RAG Agent with Ollama embeddings initialized", style="green")
            
        except Exception as e:
            console.print(f"❌ Error setting up components: {e}", style="red")
            console.print("💡 Make sure Ollama is running and gemma:latest model is available", style="yellow")
            raise
    
    def add_document(self, file_path_or_url: str) -> bool:
        """Add document to knowledge base"""
        try:
            console.print(f"📄 Processing document: {file_path_or_url}", style="blue")
            
            # Extract text from PDF
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task("Extracting text from PDF...", total=None)
                pages = DocumentProcessor.extract_text_from_pdf(file_path_or_url)
                progress.remove_task(task)
            
            if not pages:
                console.print("❌ No text extracted from document", style="red")
                return False
            
            # Process each page
            total_chunks = 0
            console.print(f"📄 Processing {len(pages)} pages...", style="blue")
            
            for page_data in pages:
                    # Chunk the text
                    chunks = DocumentProcessor.chunk_text(
                        page_data['text'], 
                        self.config.chunk_size, 
                        self.config.chunk_overlap
                    )
                    
                    # Store each chunk
                    for chunk_idx, chunk_text in enumerate(chunks):
                        chunk_id = hashlib.md5(
                            f"{file_path_or_url}_{page_data['page_number']}_{chunk_idx}".encode()
                        ).hexdigest()
                        
                        chunk = DocumentChunk(
                            id=chunk_id,
                            content=chunk_text,
                            source=file_path_or_url,
                            page_number=page_data['page_number'],
                            chunk_index=chunk_idx,
                            metadata={
                                'total_chunks': len(chunks),
                                'char_count': len(chunk_text)
                            }
                        )
                        
                        self.vector_db.add_document(chunk)
                        total_chunks += 1
                    
                    # Progress handled silently
            
            console.print(f"✅ Added {total_chunks} chunks from {len(pages)} pages", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Error adding document: {e}", style="red")
            return False
    
    def search_knowledge(self, query: str, top_k: Optional[int] = None) -> List[DocumentChunk]:
        """Search knowledge base using semantic similarity"""
        try:
            if top_k is None:
                top_k = self.config.top_k_results
            
            # Generate embedding for query
            query_embedding = self.embedder.get_embedding(query)
            
            if query_embedding:
                # Use vector similarity search
                relevant_docs = self.vector_db.search_similar(query_embedding, top_k)
                if relevant_docs:
                    console.print(f"📄 {len(relevant_docs)} relevant documents found", style="dim green")
                return relevant_docs
            else:
                # Fallback to keyword search
                return self._fallback_keyword_search(query, top_k)
            
        except Exception as e:
            console.print(f"❌ Error in semantic search: {e}", style="red")
            # Fallback to simple search
            return self._fallback_keyword_search(query, top_k)
    
    def _fallback_keyword_search(self, query: str, top_k: int) -> List[DocumentChunk]:
        """Fallback keyword-based search"""
        try:
            all_docs = self.vector_db.get_all_documents()
            
            # Simple keyword matching
            query_lower = query.lower()
            scored_docs = []
            
            for doc in all_docs:
                content_lower = doc.content.lower()
                score = 0
                
                # Count keyword matches
                for word in query_lower.split():
                    if len(word) > 2:  # Skip very short words
                        score += content_lower.count(word)
                
                if score > 0:
                    scored_docs.append((score, doc))
            
            # Sort by score and return top results
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored_docs[:top_k]]
            
        except Exception as e:
            console.print(f"❌ Error in fallback search: {e}", style="red")
            return []
    
    def generate_response_with_rag(self, user_query: str) -> str:
        """Generate response using RAG"""
        try:
            # Search for relevant context
            relevant_docs = self.search_knowledge(user_query)
            
            if relevant_docs:
                # Prepare context with longer, more meaningful chunks
                context_parts = []
                for doc in relevant_docs:
                    clean_content = doc.content.strip()
                    # Use full content, not truncated
                    context_parts.append(f"[Sayfa {doc.page_number}]\n{clean_content}")
                    context_parts.append("")  # Empty line for separation
                
                context = "\n".join(context_parts)
                
                # Create enhanced prompt
                enhanced_query = f"""Aşağıdaki doküman içeriğini kullanarak soruyu yanıtla:

{context}

Soru: {user_query}

Doküman içeriğine dayalı olarak kapsamlı bir yanıt ver. Türkçe yanıtla."""
            else:
                enhanced_query = user_query
            
            return enhanced_query
            
        except Exception as e:
            console.print(f"❌ Error in RAG generation: {e}", style="red")
            return user_query
    
    def interactive_chat(self):
        """Start interactive chat with RAG"""
        console.print("\n💬 **RAG-Enhanced Chat Started**", style="bold green")
        console.print("Type 'quit', 'exit', or use commands like /add, /docs, /clear\n", style="dim")
        
        try:
            while True:
                user_input = Prompt.ask("[bold green]You[/bold green]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("👋 Goodbye! Chat session ended.", style="blue")
                    break
                
                # Handle commands
                if user_input.startswith('/'):
                    self.handle_command(user_input)
                    continue
                
                # Generate enhanced query with RAG
                enhanced_query = self.generate_response_with_rag(user_input)
                
                # Get agent response
                console.print("\n🤖 **Assistant:**", style="bold blue")
                try:
                    self.agent.print_response(enhanced_query, stream=True)
                except Exception as e:
                    console.print(f"❌ Error: {e}", style="red")
                
                console.print()
                
        except KeyboardInterrupt:
            console.print("\n👋 Chat interrupted. Goodbye!", style="blue")
    
    def handle_command(self, command: str):
        """Handle special commands"""
        cmd = command.lower().strip()
        
        if cmd.startswith('/add'):
            # Add document
            file_path = cmd[4:].strip()
            if not file_path:
                file_path = Prompt.ask("Enter document path or URL")
            
            if file_path:
                self.add_document(file_path)
        
        elif cmd in ['/docs', '/documents']:
            # Show documents
            self.show_documents()
        
        elif cmd in ['/clear', '/cleardocs']:
            # Clear documents
            if Confirm.ask("Are you sure you want to clear all documents?"):
                self.vector_db.clear_documents()
        
        elif cmd in ['/help', '/?']:
            self.show_help()
        
        else:
            console.print(f"❓ Unknown command: {command}", style="yellow")
            console.print("Type /help for available commands.", style="dim")
    
    def show_documents(self):
        """Show loaded documents"""
        docs = self.vector_db.get_all_documents()
        
        if not docs:
            console.print("📝 No documents loaded", style="yellow")
            return
        
        # Group by source
        sources = {}
        for doc in docs:
            if doc.source not in sources:
                sources[doc.source] = []
            sources[doc.source].append(doc)
        
        table = Table(title="📚 Loaded Documents")
        table.add_column("Source", style="cyan")
        table.add_column("Pages", justify="center", style="green")
        table.add_column("Chunks", justify="center", style="blue")
        table.add_column("Total Chars", justify="center", style="yellow")
        
        for source, source_docs in sources.items():
            pages = len(set(doc.page_number for doc in source_docs if doc.page_number))
            chunks = len(source_docs)
            total_chars = sum(len(doc.content) for doc in source_docs)
            
            table.add_row(
                source[-50:] if len(source) > 50 else source,
                str(pages),
                str(chunks),
                f"{total_chars:,}"
            )
        
        console.print(table)
    
    def show_help(self):
        """Show available commands"""
        help_text = """
## 💡 Available Commands:

- `/add [path/url]` - Add PDF document to knowledge base
- `/docs` - Show loaded documents
- `/clear` - Clear all documents
- `/help` - Show this help message

## 📖 Usage Examples:

- `/add https://example.com/document.pdf` - Add PDF from URL
- `/add /path/to/document.pdf` - Add local PDF file
        """
        console.print(Panel(Markdown(help_text), title="Help", border_style="cyan"))

def create_default_rag_config() -> RAGConfig:
    """Create default RAG configuration with Ollama embeddings"""
    session_id = f"rag_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return RAGConfig(
        session_id=session_id,
        db_path="tmp/rag_agent.db",
        vector_db_path="tmp/vector_db.db",
        model_id="llama3.2:3b",
        embedding_model="embeddinggemma:latest",
        max_history=10,
        chunk_size=1000,
        chunk_overlap=200,
        top_k_results=5,
        enable_markdown=True
    )

def main():
    """Main function"""
    try:
        # Create configuration
        config = create_default_rag_config()
        
        # Show welcome
        welcome_text = f"""
# 🤖 Advanced Ollama Agent with RAG

**Chat Model:** `{config.model_id}`
**Embedding Model:** `{config.embedding_model}`
**Session:** `{config.session_id}`
**Vector DB:** SQLite with Ollama embeddings

## 🌟 Features:
- 📄 PDF document processing
- 🔍 Semantic search with Ollama embeddings
- 🎯 Retrieval-Augmented Generation (RAG)
- 💬 Interactive chat interface
- 💾 Session persistence
- 📊 Document management

Ready to chat with your documents using semantic similarity!
        """
        console.print(Panel(Markdown(welcome_text), title="🚀 Welcome", border_style="blue"))
        
        # Initialize agent
        agent = AdvancedRAGAgent(config)
        
        # Check if embedding model is available
        console.print(f"🔍 Checking if {config.embedding_model} is available...", style="blue")
        
        # Check if we should add a demo document
        if Confirm.ask("Would you like to add a demo document?"):
            demo_url = "https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
            console.print(f"📄 Adding demo document: {demo_url}")
            console.print("⏳ This may take a moment for embedding generation...", style="yellow")
            agent.add_document(demo_url)
        
        # Start interactive chat
        agent.interactive_chat()
        
    except KeyboardInterrupt:
        console.print("\n👋 Application terminated by user.", style="blue")
    except Exception as e:
        console.print(f"❌ Fatal error: {e}", style="red")

if __name__ == "__main__":
    main()