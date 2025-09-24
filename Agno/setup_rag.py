#!/usr/bin/env python3
"""
🛠️ Ollama RAG Agent Setup Script

Bu script, RAG agent'ı kullanmak için gerekli Ollama modellerini kontrol eder ve indirir.
"""

import subprocess
import sys
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def run_command(command: str, description: str = "") -> bool:
    """Run a shell command and return success status"""
    try:
        if description:
            console.print(f"🔄 {description}", style="blue")
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            console.print(f"✅ Success: {description or command}", style="green")
            return True
        else:
            console.print(f"❌ Failed: {description or command}", style="red")
            if result.stderr:
                console.print(f"   Error: {result.stderr.strip()}", style="red")
            return False
            
    except Exception as e:
        console.print(f"❌ Error running command: {e}", style="red")
        return False

def check_ollama():
    """Check if Ollama is installed and running"""
    console.print("\n🔍 Checking Ollama installation...", style="bold blue")
    
    # Check if ollama command exists
    if not run_command("ollama --version", "Checking Ollama version"):
        console.print("\n❌ Ollama is not installed!", style="red")
        console.print("📥 Please install Ollama from: https://ollama.com/download", style="yellow")
        return False
    
    # Check if Ollama is running
    if not run_command("ollama list", "Checking Ollama service"):
        console.print("\n⚠️  Ollama might not be running.", style="yellow")
        console.print("🚀 Try running: ollama serve", style="cyan")
        return False
    
    return True

def pull_required_models():
    """Pull required Ollama models"""
    console.print("\n📦 Pulling required models...", style="bold blue")
    
    models = [
        ("llama3.2:3b", "Chat model for conversations"),
        ("embeddinggemma:latest", "Embedding model for semantic search")
    ]
    
    success = True
    
    for model, description in models:
        console.print(f"\n📥 Pulling {model} ({description})...", style="blue")
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task(f"Downloading {model}...", total=None)
            
            result = subprocess.run(f"ollama pull {model}", shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            progress.remove_task(task)
        
        if result.returncode == 0:
            console.print(f"✅ {model} downloaded successfully", style="green")
        else:
            console.print(f"❌ Failed to download {model}", style="red")
            if result.stderr:
                console.print(f"   Error: {result.stderr.strip()}", style="red")
            success = False
    
    return success

def check_python_packages():
    """Check Python package requirements"""
    console.print("\n🐍 Checking Python packages...", style="bold blue")
    
    packages = ["agno", "rich", "pydantic", "PyPDF2", "requests"]
    
    missing = []
    
    for package in packages:
        try:
            __import__(package.lower().replace("pdf2", "PDF2"))
            console.print(f"✅ {package} is installed", style="green")
        except ImportError:
            console.print(f"❌ {package} is missing", style="red")
            missing.append(package)
    
    if missing:
        console.print(f"\n📦 Install missing packages:", style="yellow")
        console.print(f"   pip install {' '.join(missing)}", style="cyan")
        return False
    
    return True

def test_embedding_generation():
    """Test embedding generation"""
    console.print("\n🧪 Testing embedding generation...", style="bold blue")
    
    try:
        from agno.knowledge.embedder.ollama import OllamaEmbedder
        
        embedder = OllamaEmbedder(id="embeddinggemma:latest")
        test_text = "This is a test sentence for embedding generation."
        
        console.print("🔄 Generating test embedding...", style="blue")
        embedding = embedder.get_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            console.print(f"✅ Embedding generated successfully! Dimension: {len(embedding)}", style="green")
            return True
        else:
            console.print("❌ Embedding generation failed", style="red")
            return False
            
    except Exception as e:
        console.print(f"❌ Error testing embeddings: {e}", style="red")
        return False

def main():
    """Main setup function"""
    
    # Welcome message
    welcome_text = """
# 🛠️ Ollama RAG Agent Setup

This script will check and setup everything needed for the RAG agent:

1. ✅ Ollama installation
2. 📦 Required models (llama3.2:3b, gemma:latest)  
3. 🐍 Python packages
4. 🧪 Embedding functionality test

Let's get started!
    """
    
    console.print(Panel(welcome_text, title="Setup", border_style="blue"))
    
    # Setup steps
    steps = [
        ("Ollama Check", check_ollama),
        ("Model Download", pull_required_models),
        ("Package Check", check_python_packages),
        ("Embedding Test", test_embedding_generation)
    ]
    
    results = []
    
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append((step_name, result))
            
            if not result:
                console.print(f"\n⚠️  {step_name} failed. Please fix the issues above before continuing.", style="yellow")
                # Don't break, continue with other checks
                
        except Exception as e:
            console.print(f"\n❌ Unexpected error in {step_name}: {e}", style="red")
            results.append((step_name, False))
    
    # Summary
    console.print("\n" + "="*60, style="green")
    console.print("🎯 SETUP SUMMARY", style="bold green")
    console.print("="*60, style="green")
    
    all_passed = True
    for step_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        style = "green" if result else "red"
        console.print(f"  {step_name}: {status}", style=style)
        if not result:
            all_passed = False
    
    if all_passed:
        console.print("\n🎉 Setup completed successfully!", style="bold green")
        console.print("\n🚀 You can now run:", style="cyan")
        console.print("   python ollama-rag-agent.py", style="white")
    else:
        console.print("\n⚠️  Some setup steps failed. Please fix the issues above.", style="yellow")
        
        # Provide help
        console.print("\n💡 Common solutions:", style="cyan")
        console.print("   • Install Ollama: https://ollama.com/download", style="white")
        console.print("   • Start Ollama: ollama serve", style="white")
        console.print("   • Install packages: pip install -r requirements_rag.txt", style="white")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n👋 Setup cancelled by user.", style="blue")
    except Exception as e:
        console.print(f"\n❌ Fatal setup error: {e}", style="red")