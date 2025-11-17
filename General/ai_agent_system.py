"""
REAL AI AGENT SYSTEM - With LangChain & CrewAI
Works using real agent frameworks
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

# For PDF reading
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# For DOCX reading
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Unnecessary imports removed - We are only using Gemini

# Gemini API
import google.generativeai as genai

st.set_page_config(
    page_title="AI Agent Control Panel",
    page_icon="🤖",
    layout="wide"
)

# ============================================================================
# AGENT TYPES
# ============================================================================

AGENT_TYPES = {
    "Research Agent": {
        "icon": "🔍",
        "description": "Real web research with Google Search grounding",
        "color": "blue",
        "tools": ["google_search"],
        "model": "gemini-1.5-flash"
    },
    "Data Analysis Agent": {
        "icon": "📊",
        "description": "Data analysis and computation with code execution",
        "color": "green",
        "tools": ["code_execution"],
        "model": "gemini-1.5-flash"
    },
    "Content Writer Agent": {
        "icon": "✍️",
        "description": "SEO-friendly content with structured output",
        "color": "purple",
        "tools": ["structured_output"],
        "model": "gemini-1.5-flash"
    },
    "Document Analysis Agent": {
        "icon": "📄",
        "description": "PDF/DOCX analysis with document understanding",
        "color": "cyan",
        "tools": ["document_understanding"],
        "model": "gemini-1.5-flash"
    },
    "Code Assistant Agent": {
        "icon": "💻",
        "description": "Code writing and testing with code execution",
        "color": "red",
        "tools": ["code_execution"],
        "model": "gemini-1.5-flash"
    }
}

# ============================================================================ 
# HELPER FUNCTIONS - File and URL processing
# ============================================================================ 

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        if PDF_AVAILABLE:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            return "PDF reading library is not installed."
    except Exception as e:
        return f"PDF reading error: {str(e)}"

def extract_text_from_docx(docx_file) -> str:
    """Extract text from DOCX file"""
    try:
        if DOCX_AVAILABLE:
            doc = docx.Document(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        else:
            return "DOCX reading library is not installed."
    except Exception as e:
        return f"DOCX reading error: {str(e)}"

def extract_text_from_txt(txt_file) -> str:
    """Extract text from TXT file"""
    try:
        return txt_file.read().decode('utf-8')
    except Exception as e:
        return f"TXT reading error: {str(e)}"

def scrape_website(url: str) -> str:
    """Scrape content from website"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]  # First 5000 characters
    except Exception as e:
        return f"Web scraping error: {str(e)}"

def read_csv_file(csv_file) -> str:
    """Read CSV file and extract summary"""
    try:
        df = pd.read_csv(csv_file)
        summary = f"""
CSV File Summary:
- Row count: {len(df)}
- Column count: {len(df.columns)}
- Columns: {', '.join(df.columns.tolist())}

First 5 rows:
{df.head().to_string()}

Statistics:
{df.describe().to_string()}
"""
        return summary
    except Exception as e:
        return f"CSV reading error: {str(e)}"

def read_excel_file(excel_file) -> str:
    """Read Excel file and extract summary"""
    try:
        df = pd.read_excel(excel_file)
        summary = f"""
Excel File Summary:
- Row count: {len(df)}
- Column count: {len(df.columns)}
- Columns: {', '.join(df.columns.tolist())}

First 5 rows:
{df.head().to_string()}

Statistics:
{df.describe().to_string()}
"""
        return summary
    except Exception as e:
        return f"Excel reading error: {str(e)}"

# Unnecessary tool functions removed - Gemini uses them directly

# ============================================================================ 
# RESEARCH AGENT - Real web research with LangChain
# ============================================================================ 

def research_agent_task(query: str, gemini_key: str, additional_context: str = "") -> Dict:
    """Research Agent - Gemini 1.5 Flash Lite + Google Search (New SDK)"""
    try:
        # Use new SDK: google.genai
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
        
        ref_text = f"\n\nAdditional Context:\n{additional_context[:2000]}" if additional_context else ""
        
        prompt = f"""You are a professional research expert. Conduct in-depth research using Google Search with CURRENT and VERIFIED information.

🔍 Research Topic: {query}
{ref_text}

RESEARCH REQUIREMENTS:
✅ Use current sources (2024-2025)
✅ Check multiple reliable sources
✅ Add figures and statistics
✅ Evaluate different perspectives
✅ Specify source links

OUTPUT FORMAT (JSON):
{{
    "summary": "Comprehensive summary (3-4 sentences, including main findings)",
    "key_findings": ["Finding 1 (detailed)", "Finding 2 (detailed)", "Finding 3 (detailed)", "Finding 4 (detailed)", "Finding 5 (detailed)"],
    "detailed_analysis": "In-depth analysis (4-5 paragraphs, supported by figures and examples)",
    "sources": ["Source 1 (title and link)", "Source 2", "Source 3", "Source 4"],
    "recommendations": ["Recommendation 1 (action-oriented)", "Recommendation 2", "Recommendation 3"],
    "statistics": ["Statistic 1", "Statistic 2"],
    "trends": ["Trend 1", "Trend 2"],
    "expert_opinions": ["Expert opinion 1", "Expert opinion 2"]
}}
"""
        
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=config
        )
        
        text = response.text.strip()
        
        # Get sources from grounding metadata
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

            # Add sources from grounding
            if sources:
                parsed_result['sources'] = sources[:3] + parsed_result.get('sources', [])[:2]

        except json.JSONDecodeError as e:
            # JSON parsing failed - return actual error
            return {
                "status": "error",
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": text[:1000],  # For debugging
                "query": query,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            # Other unexpected errors
            return {
                "status": "error",
                "error": f"Unexpected error during parsing: {str(e)}",
                "raw_response": text[:1000],
                "query": query,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return {
            "status": "success",
            "agent_type": "Research Agent",
            "framework": "Gemini 1.5 Flash Lite + Google Search",
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
# DATA ANALYSIS AGENT - Data analysis with LangChain
# ============================================================================ 

def data_analysis_agent_task(data_description: str, gemini_key: str, data_content: str = "") -> Dict:
    """Data Analysis Agent - Gemini 1.5 Flash Lite + Code Execution"""
    try:
        genai.configure(api_key=gemini_key)
        
        # Create model with code execution tool
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            tools='code_execution'
        )
        
        data_text = f"\n\nData:\n{data_content[:3000]}" if data_content else ""
        
        prompt = f"""You are a professional data analyst. Perform REAL calculations and analyses by running Python code.

📊 Analysis Task: {data_description}
{data_text}

ANALYSIS REQUIREMENTS:
✅ Perform real statistical calculations with Python (use numpy, pandas)
✅ Check data quality (missing values, outliers)
✅ Perform distribution analysis (normal distribution, skewness)
✅ Perform correlation analysis (relationships between variables)
✅ Perform trend analysis (if there is a time series)
✅ Suggest visualizations (matplotlib, seaborn)

OUTPUT FORMAT (JSON):
{{
    "summary": "Dataset summary (row/column count, data type, general status)",
    "statistics": {{
        "mean": 0,
        "median": 0,
        "std": 0,
        "min": 0,
        "max": 0,
        "q1": 0,
        "q3": 0,
        "missing_values": 0,
        "outliers": 0
    }},
    "insights": [
        "Insight 1 (supported by numbers)",
        "Insight 2 (statistical finding)",
        "Insight 3 (pattern/trend)"
    ],
    "correlations": [
        "Variable1 - Variable2: correlation coefficient",
        "Variable3 - Variable4: correlation coefficient"
    ],
    "visualizations": [
        "Histogram (for distribution analysis)",
        "Box plot (for outlier detection)",
        "Scatter plot (for correlation)",
        "Time series plot (for trend)"
    ],
    "recommendations": [
        "Recommendation 1 (action-oriented)",
        "Recommendation 2 (data quality)",
        "Recommendation 3 (further analysis)"
    ],
    "data_quality": "Good/Medium/Low (with explanation)"
}}

NOTE: Run real Python code and report the results!"""
        
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
        except json.JSONDecodeError as e:
            # JSON parsing failed - return actual error
            return {
                "status": "error",
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": text[:1000],  # For debugging
                "data_description": data_description,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            # Other unexpected errors
            return {
                "status": "error",
                "error": f"Unexpected error during parsing: {str(e)}",
                "raw_response": text[:1000],
                "data_description": data_description,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return {
            "status": "success",
            "agent_type": "Data Analysis Agent",
            "framework": "Gemini 1.5 Flash Lite + Code Execution",
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
# CONTENT WRITER AGENT - Content generation with CrewAI
# ============================================================================ 

def content_writer_agent_task(topic: str, content_type: str, gemini_key: str, reference_content: str = "") -> Dict:
    """Content Writer Agent - Gemini 1.5 Flash + Structured Output (Content-Type Specific)"""
    try:
        genai.configure(api_key=gemini_key)
        
        # Custom settings based on content type
        content_specs = {
            "Article": {
                "word_count": "800-1200 words",
                "structure": "Introduction, 3-4 main sections, conclusion",
                "tone": "Professional and informative",
                "extras": "Subheadings, bulleted lists",
                "visual": "Suggest infographic or explanatory visuals"
            },
            "Blog": {
                "word_count": "600-900 words",
                "structure": "Catchy intro, 2-3 main points, conclusion with CTA",
                "tone": "Friendly and engaging",
                "extras": "Personal anecdotes, emoji usage",
                "visual": "Suggest blog banner and in-content visuals"
            },
            "Social Media": {
                "word_count": "50-150 characters (Twitter) or 200-300 words (LinkedIn)",
                "structure": "Hook + Main message + CTA + Hashtag",
                "tone": "Short, concise, and impactful",
                "extras": "Emoji, hashtag (#), mention (@)",
                "visual": "Suggest eye-catching visual or carousel design"
            },
            "E-mail": {
                "word_count": "200-400 words",
                "structure": "Subject line + Personalized intro + Main message + CTA",
                "tone": "Professional yet personal",
                "extras": "Subject line alternatives, PS note",
                "visual": "Suggest e-mail header visual and CTA button design"
            },
            "Newsletter": {
                "word_count": "400-600 words",
                "structure": "Title + Summary + Details + Related links",
                "tone": "Objective and informative",
                "extras": "Source links, date information",
                "visual": "Suggest news visual and thumbnail"
            },
            "Product Description": {
                "word_count": "150-300 words",
                "structure": "Features + Benefits + Technical details + CTA",
                "tone": "Persuasive and clear",
                "extras": "Bullet points, feature list",
                "visual": "Suggest product images and usage scenarios"
            }
        }
        
        spec = content_specs.get(content_type, content_specs["Article"])
        
        # Generation config for structured output
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
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
        
        ref_text = f"\n\nReference:\n{reference_content[:2000]}" if reference_content else ""
        
        prompt = f"""You are a professional {content_type} writer. Create content on the following topic.

Topic: {topic}
Content Type: {content_type}
{ref_text}

SPECIAL REQUIREMENTS:
- Length: {spec['word_count']}
- Structure: {spec['structure']}
- Tone: {spec['tone']}
- Extras: {spec['extras']}
- Visual: {spec['visual']}

OUTPUT REQUIREMENTS:
- title: Catchy and SEO-friendly title
- content: Content of {spec['word_count']} length, in a {spec['tone']} tone
- keywords: 5-7 keywords
- summary: 30-50 word summary
- seo_score: score between 1-10
- target_audience: Target audience definition
- visual_suggestions: 2-3 visual suggestions (detailed description)
- hashtags: 5-10 relevant hashtags (for social media)
- cta: Call-to-action text

NOTE: Use a {spec['tone']} language suitable for the {content_type} format!"""
        
        response = model.generate_content(prompt)
        parsed_result = json.loads(response.text)
        
        # Add content type
        parsed_result['content_type'] = content_type
        parsed_result['specifications'] = spec
        
        return {
            "status": "success",
            "agent_type": "Content Writer Agent",
            "framework": "Gemini 1.5 Flash + Structured Output",
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
    """Document Analysis Agent - Gemini 1.5 Flash Lite + Document Understanding"""
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if not document_content:
            return {
                "status": "error",
                "error": "Document content not provided",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Determine document type
        doc_type = "General"
        if any(word in document_description.lower() for word in ['report']):
            doc_type = "Report"
        elif any(word in document_description.lower() for word in ['article', 'paper']):
            doc_type = "Academic"
        elif any(word in document_description.lower() for word in ['contract', 'agreement']):
            doc_type = "Legal"
        elif any(word in document_description.lower() for word in ['cv', 'resume']):
            doc_type = "Resume"
        
        prompt = f"""You are a professional document analysis expert. Perform analysis on a {doc_type} type document.

📄 Analysis Request: {document_description}
📋 Document Type: {doc_type}

Document Content:
{document_content[:10000]}

ANALYSIS REQUIREMENTS:
✅ Understand the document's purpose and context
✅ Extract key information (names, dates, numbers)
✅ Perform structural analysis (sections, headings)
✅ Perform sentiment and tone analysis
✅ Identify missing or unclear points

OUTPUT FORMAT (JSON):
{{
    "summary": "Comprehensive summary (3-4 sentences, including the main message)",
    "document_type": "{doc_type}",
    "key_points": [
        "Key point 1 (detailed)",
        "Key point 2 (detailed)",
        "Key point 3 (detailed)",
        "Key point 4 (detailed)"
    ],
    "entities": {{
        "people": ["Name 1", "Name 2"],
        "organizations": ["Organization 1", "Organization 2"],
        "dates": ["Date 1", "Date 2"],
        "locations": ["Location 1", "Location 2"],
        "numbers": ["Number 1: description", "Number 2: description"]
    }},
    "structure": {{
        "sections": ["Section 1", "Section 2"],
        "word_count": 0,
        "page_count": 0
    }},
    "sentiment": "Positive/Negative/Neutral (with explanation)",
    "tone": "Formal/Informal/Technical/Academic",
    "key_insights": [
        "Key Insight 1",
        "Key Insight 2"
    ],
    "missing_info": ["Missing info 1", "Missing info 2"],
    "recommendations": [
        "Recommendation 1 (action-oriented)",
        "Recommendation 2"
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
        except json.JSONDecodeError as e:
            # JSON parsing failed - return actual error
            return {
                "status": "error",
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": text[:1000],  # For debugging
                "document_description": document_description,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            # Other unexpected errors
            return {
                "status": "error",
                "error": f"Unexpected error during parsing: {str(e)}",
                "raw_response": text[:1000],
                "document_description": document_description,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return {
            "status": "success",
            "agent_type": "Document Analysis Agent",
            "framework": "Gemini 1.5 Flash Lite + Document Understanding",
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
# CUSTOMER SUPPORT AGENT - Q&A with LangChain
# ============================================================================ 

# Customer Support Agent removed - replaced with Document Analysis

# ============================================================================ 
# CODE ASSISTANT AGENT - Code writing with CrewAI
# ============================================================================ 

def code_assistant_agent_task(task_description: str, language: str, gemini_key: str, existing_code: str = "") -> Dict:
    """Code Assistant Agent - Gemini 1.5 Flash Lite + Code Execution"""
    try:
        genai.configure(api_key=gemini_key)
        
        # Model with code execution tool
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            tools='code_execution'
        )
        
        existing_code_text = f"\n\nExisting Code:\n```{language}\n{existing_code[:2000]}\n```" if existing_code else ""
        
        # Language-specific settings
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
        
        prompt = f"""You are an expert {language} developer. WRITE, TEST, and RUN CODE with code execution.

💻 Task: {task_description}
🔤 Language: {language}
{existing_code_text}

CODE REQUIREMENTS:
✅ Adhere to {spec['best_practices']} standards
✅ Apply clean code principles
✅ Consider edge cases
✅ Perform performance optimization
✅ Check for security vulnerabilities
✅ Run and test real code

OUTPUT FORMAT (JSON):
{{
    "code": "Fully working code (with comments)",
    "explanation": "What the code does (detailed, step-by-step)",
    "complexity": {{
        "time": "O(n) - time complexity",
        "space": "O(1) - space complexity"
    }},
    "test_cases": [
        "Test 1: Input -> Output (successful)",
        "Test 2: Edge case (successful)",
        "Test 3: Error handling (successful)"
    ],
    "best_practices": [
        "Applied best practice 1",
        "Applied best practice 2"
    ],
    "improvements": [
        "Improvement suggestion 1",
        "Improvement suggestion 2"
    ],
    "dependencies": ["Required library 1", "Required library 2"],
    "security_notes": ["Security note 1", "Security note 2"]
}}

NOTE: Write and run real {language} code! Test framework: {spec['testing']} """
        
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
        except json.JSONDecodeError as e:
            # JSON parsing failed - return actual error
            return {
                "status": "error",
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": text[:1000],  # For debugging
                "task_description": task_description,
                "language": language,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            # Other unexpected errors
            return {
                "status": "error",
                "error": f"Unexpected error during parsing: {str(e)}",
                "raw_response": text[:1000],
                "task_description": task_description,
                "language": language,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return {
            "status": "success",
            "agent_type": "Code Assistant Agent",
            "framework": "Gemini 1.5 Flash Lite + Code Execution",
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
    """Execute Agent Task - Gemini 1.5 Flash Lite + Tools"""
    
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
            task_params.get("content_type", "Article"),
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
            "error": f"Unknown agent type: {agent_type}"
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

st.title("🤖 AI Agent Control Panel")
st.markdown("**Real Agent Frameworks:** LangChain ReAct + CrewAI Multi-Agent")

# ============================================================================ 
# SIDEBAR - Agent Management
# ============================================================================ 

with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key,
        help="Get it from Google AI Studio: https://aistudio.google.com/app/apikey"
    )
    
    if api_key:
        st.session_state.gemini_api_key = api_key
        st.success("✅ API Key saved")
    
    st.divider()
    
    # Create Agent
    st.header("➕ Create New Agent")
    
    agent_type = st.selectbox(
        "Agent Type",
        list(AGENT_TYPES.keys()),
        format_func=lambda x: f"{AGENT_TYPES[x]['icon']} {x}"
    )
    
    agent_name = st.text_input("Agent Name", f"{agent_type.split()[0]}Agent-{len(st.session_state.agents)+1:02d}")
    
    if st.button("🚀 Create Agent", use_container_width=True):
        if not st.session_state.gemini_api_key:
            st.error("❌ First, enter the API Key!")
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
                "action": "Agent Created",
                "details": f"{agent_type} ({AGENT_TYPES[agent_type]['model']})"
            })
            st.success(f"✅ {agent_name} created!")
            st.rerun()
    
    st.divider()
    
    # Active Agents
    st.header("🤖 Active Agents")
    if st.session_state.agents:
        for agent in st.session_state.agents:
            with st.expander(f"{agent['icon']} {agent['name']}", expanded=False):
                st.write(f"**Type:** {agent['type']}")
                st.write(f"**Model:** {agent['model']}")
                st.write(f"**Tools:** {', '.join(agent['tools'])}")
                st.write(f"**Status:** {agent['status']}")
                st.write(f"**Tasks:** {agent['tasks_completed']}")
                if st.button(f"🗑️ Delete", key=f"delete_{agent['id']}"):
                    st.session_state.agents.remove(agent)
                    st.rerun()
    else:
        st.info("No agents yet")

# ============================================================================ 
# MAIN - Tabs
# ============================================================================ 

tab1, tab2, tab3 = st.tabs(["🎯 Task Management", "📊 Results", "📋 Logs"])

# TAB 1: Task Management
with tab1:
    st.header("🎯 Create New Task")
    
    if not st.session_state.agents:
        st.warning("⚠️ First, create an agent from the sidebar!")
    else:
        # Select agent
        agent_options = [f"{a['icon']} {a['name']} ({a['type']})" for a in st.session_state.agents]
        selected_agent_display = st.selectbox("Select Agent", agent_options)
        selected_agent_idx = agent_options.index(selected_agent_display)
        selected_agent = st.session_state.agents[selected_agent_idx]
        
        agent_type = selected_agent['type']
        
        st.info(f"**Model:** {selected_agent['model']} | **Tools:** {', '.join(selected_agent['tools'])} | **Description:** {AGENT_TYPES[agent_type]['description']}")
        
        # Input fields based on agent type
        task_params = {}
        
        if agent_type == "Research Agent":
            st.subheader("🔍 Research Parameters")
            task_params["query"] = st.text_area("Research Topic", "What are the AI trends for 2025?")
            
            # Additional source
            st.write("**Additional Source (Optional):**")
            col1, col2, col3 = st.columns(3)
            with col1:
                url_input = st.text_input("URL")
                if url_input:
                    task_params["additional_context"] = scrape_website(url_input)
            with col2:
                file_upload = st.file_uploader("Upload File", type=['pdf', 'docx', 'txt'])
                if file_upload:
                    if file_upload.name.endswith('.pdf'):
                        task_params["additional_context"] = extract_text_from_pdf(file_upload)
                    elif file_upload.name.endswith('.docx'):
                        task_params["additional_context"] = extract_text_from_docx(file_upload)
                    else:
                        task_params["additional_context"] = extract_text_from_txt(file_upload)
            with col3:
                manual_text = st.text_area("Manual Text")
                if manual_text:
                    task_params["additional_context"] = manual_text
        
        elif agent_type == "Data Analysis Agent":
            st.subheader("📊 Data Analysis Parameters")
            task_params["data_description"] = st.text_area("Data Description", "Analyze sales data")
            
            st.write("**Data Source:**")
            col1, col2 = st.columns(2)
            with col1:
                data_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
                if data_file:
                    if data_file.name.endswith('.csv'):
                        task_params["data_content"] = read_csv_file(data_file)
                    else:
                        task_params["data_content"] = read_excel_file(data_file)
            with col2:
                manual_data = st.text_area("Manual Data")
                if manual_data:
                    task_params["data_content"] = manual_data
        
        elif agent_type == "Content Writer Agent":
            st.subheader("✍️ Content Generation Parameters")
            task_params["topic"] = st.text_input("Topic", "Artificial Intelligence and the Future")
            task_params["content_type"] = st.selectbox("Content Type", ["Article", "Blog", "Social Media", "E-mail", "Newsletter", "Product Description"])
            
            st.write("**Reference Content (Optional):**")
            ref_file = st.file_uploader("Reference File", type=['pdf', 'docx', 'txt'])
            if ref_file:
                if ref_file.name.endswith('.pdf'):
                    task_params["reference_content"] = extract_text_from_pdf(ref_file)
                elif ref_file.name.endswith('.docx'):
                    task_params["reference_content"] = extract_text_from_docx(ref_file)
                else:
                    task_params["reference_content"] = extract_text_from_txt(ref_file)
        
        elif agent_type == "Document Analysis Agent":
            st.subheader("📄 Document Analysis Parameters")
            task_params["document_description"] = st.text_area("Analysis Request", "Summarize this document and extract the main points")
            
            st.write("**Upload Document:**")
            doc_file = st.file_uploader("PDF/DOCX/TXT File", type=['pdf', 'docx', 'txt'], key="doc_analysis")
            if doc_file:
                if doc_file.name.endswith('.pdf'):
                    task_params["document_content"] = extract_text_from_pdf(doc_file)
                elif doc_file.name.endswith('.docx'):
                    task_params["document_content"] = extract_text_from_docx(doc_file)
                else:
                    task_params["document_content"] = extract_text_from_txt(doc_file)
                st.success(f"✅ {doc_file.name} uploaded ({len(task_params.get('document_content', ''))} characters)")
        
        elif agent_type == "Code Assistant Agent":
            st.subheader("💻 Code Assistant Parameters")
            task_params["task_description"] = st.text_area("Task Description", "Calculate Fibonacci sequence")
            task_params["language"] = st.selectbox("Programming Language", ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go"])
            task_params["existing_code"] = st.text_area("Existing Code (Optional)")
        
        # Run task
        if st.button("▶️ Run Task", type="primary", use_container_width=True):
            if not st.session_state.gemini_api_key:
                st.error("❌ API Key required!")
            else:
                with st.spinner(f"🤖 {selected_agent['name']} is running..."):
                    progress = st.progress(0)
                    status = st.empty()

                    try:
                        status.text("Processing task...")
                        progress.progress(30)

                        # Run the agent
                        result = execute_agent_task(
                            agent_type,
                            task_params,
                            st.session_state.gemini_api_key
                        )

                        # Update session state first
                        st.session_state.task_results.append(result)
                        st.session_state.execution_logs.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "agent": selected_agent['name'],
                            "action": "Task Completed" if result['status'] == 'success' else "Task Failed",
                            "details": result.get('framework', 'N/A')
                        })
                        selected_agent['tasks_completed'] += 1

                        progress.progress(100)
                        status.text("✅ Completed!")
                        time.sleep(0.5)

                        if result['status'] == 'success':
                            st.success("✅ Task completed successfully!")
                        else:
                            st.error(f"❌ Agent Error: {result.get('error', 'Unknown error')}")
                            # Show debug info on error as well
                            with st.expander("🔍 Error Details"):
                                st.json({
                                    "error": result.get('error'),
                                    "agent_type": result.get('agent_type'),
                                    "framework": result.get('framework'),
                                    "raw_response": result.get('raw_response', 'N/A')[:500] + "..."
                                })

                        st.rerun()

                    except Exception as e:
                        progress.progress(100)
                        status.text("❌ Error!")
                        st.error(f"❌ System Error: {str(e)}")
                        # Save on error as well
                        error_result = {
                            "status": "error",
                            "error": f"System error: {str(e)}",
                            "agent_type": agent_type,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.task_results.append(error_result)

# TAB 2: Results
with tab2:
    st.header("📊 Task Results")

    if not st.session_state.task_results:
        st.warning("⚠️ No task results yet. Run the first task!")
    
    if st.session_state.task_results:
        for idx, result in enumerate(reversed(st.session_state.task_results)):
            agent_type = result.get('agent_type', 'Unknown Agent')
            icon = AGENT_TYPES.get(agent_type, {}).get('icon', '🤖')
            
            with st.expander(
                f"{icon} {agent_type} - {result.get('timestamp', 'N/A')}",
                expanded=(idx == 0)
            ):
                if result.get('status') == 'success':
                    st.success(f"✅ Success | Framework: **{result.get('framework', 'N/A')}**")
                    
                    # Show results in a readable format
                    res = result.get('result', {})
                    
                    # Custom view based on agent type
                    if agent_type == "Research Agent":
                        st.subheader("📝 Summary")
                        st.info(res.get('summary', 'N/A'))
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Finding Count", len(res.get('key_findings', [])))
                        with col2:
                            st.metric("Source Count", len(res.get('sources', [])))
                        with col3:
                            st.metric("Recommendation Count", len(res.get('recommendations', [])))
                        
                        st.subheader("🔍 Key Findings")
                        for i, finding in enumerate(res.get('key_findings', []), 1):
                            st.markdown(f"**{i}.** {finding}")
                        
                        # Statistics and Trends
                        if res.get('statistics') or res.get('trends'):
                            col1, col2 = st.columns(2)
                            with col1:
                                if res.get('statistics'):
                                    st.markdown("**📈 Statistics:**")
                                    for stat in res.get('statistics', []):
                                        st.markdown(f"- {stat}")
                            with col2:
                                if res.get('trends'):
                                    st.markdown("**📊 Trends:**")
                                    for trend in res.get('trends', []):
                                        st.markdown(f"- {trend}")
                        
                        # Expert Opinions
                        if res.get('expert_opinions'):
                            st.markdown("**👨‍🏫 Expert Opinions:**")
                            for opinion in res.get('expert_opinions', []):
                                st.markdown(f"> {opinion}")
                        
                        st.subheader("📊 Detailed Analysis")
                        st.markdown(f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>{res.get('detailed_analysis', 'N/A')}</div>", unsafe_allow_html=True)
                        
                        st.subheader("📚 Sources")
                        for i, source in enumerate(res.get('sources', []), 1):
                            st.markdown(f"**[{i}]** {source}")
                        
                        st.subheader("💡 Recommendations")
                        for i, rec in enumerate(res.get('recommendations', []), 1):
                            st.success(f"**{i}.** {rec}")
                
                elif agent_type == "Data Analysis Agent":
                    st.subheader("📝 Dataset Summary")
                    st.info(res.get('summary', 'N/A'))
                    
                    # Data Quality
                    if res.get('data_quality'):
                        quality = res.get('data_quality', '')
                        if 'Good' in quality:
                            st.success(f"✅ Data Quality: {quality}")
                        elif 'Medium' in quality:
                            st.warning(f"⚠️ Data Quality: {quality}")
                        else:
                            st.error(f"❌ Data Quality: {quality}")
                    
                    st.subheader("📊 Statistical Analysis")
                    stats = res.get('statistics', {})
                    
                    # Main metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{stats.get('mean', 'N/A')}")
                        st.metric("Q1", f"{stats.get('q1', 'N/A')}")
                    with col2:
                        st.metric("Median", f"{stats.get('median', 'N/A')}")
                        st.metric("Q3", f"{stats.get('q3', 'N/A')}")
                    with col3:
                        st.metric("Min", f"{stats.get('min', 'N/A')}")
                        st.metric("Max", f"{stats.get('max', 'N/A')}")
                    with col4:
                        st.metric("Std Deviation", f"{stats.get('std', 'N/A')}")
                        st.metric("Outliers", f"{stats.get('outliers', 'N/A')}")
                    
                    # Missing values warning
                    if stats.get('missing_values', 0) > 0:
                        st.warning(f"⚠️ Missing Data: {stats.get('missing_values')} values")
                    
                    # Correlations
                    if res.get('correlations'):
                        st.subheader("🔗 Correlation Analysis")
                        for corr in res.get('correlations', []):
                            st.markdown(f"- {corr}")
                    
                    st.subheader("💡 Key Insights")
                    for i, insight in enumerate(res.get('insights', []), 1):
                        st.markdown(f"**{i}.** {insight}")
                    
                    st.subheader("📈 Visualization Suggestions")
                    viz_cols = st.columns(2)
                    for i, viz in enumerate(res.get('visualizations', [])):
                        with viz_cols[i % 2]:
                            st.markdown(f"📊 {viz}")
                    
                    st.subheader("🎯 Recommendations")
                    for i, rec in enumerate(res.get('recommendations', []), 1):
                        st.success(f"**{i}.** {rec}")
                
                elif agent_type == "Content Writer Agent":
                    content_type = res.get('content_type', 'Article')
                    specs = res.get('specifications', {})
                    
                    # Content type badge
                    st.markdown(f"**📋 Content Type:** `{content_type}` | **📏 Length:** `{specs.get('word_count', 'N/A')}`")
                    
                    st.subheader(f"📰 {res.get('title', 'Title')}")
                    
                    # Content
                    st.markdown("**✍️ Content:**")
                    st.markdown(f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50;'>{res.get('content', 'N/A')}</div>", unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("SEO Score", f"{res.get('seo_score', 'N/A')}/10")
                    with col2:
                        st.metric("Word Count", f"~{len(res.get('content', '').split())}")
                    with col3:
                        st.metric("Characters", f"{len(res.get('content', ''))}")
                    
                    # Summary
                    st.markdown("**📝 Summary:**")
                    st.info(res.get('summary', 'N/A'))
                    
                    # Keywords and Hashtags
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🏷️ Keywords:**")
                        for kw in res.get('keywords', []):
                            st.markdown(f"- `{kw}`")
                    with col2:
                        st.markdown("**#️⃣ Hashtags:**")
                        hashtags = res.get('hashtags', [])
                        if hashtags:
                            st.write(" ".join([f"`{tag}`" for tag in hashtags]))
                    
                    # CTA
                    if res.get('cta'):
                        st.markdown("**📢 Call-to-Action:**")
                        st.success(res.get('cta'))
                    
                    # Visual Suggestions
                    st.markdown("**🎨 Visual Suggestions:**")
                    for i, visual in enumerate(res.get('visual_suggestions', []), 1):
                        st.markdown(f"{i}. 🖼️ {visual}")
                    
                    # Target Audience
                    st.markdown("**🎯 Target Audience:**")
                    st.write(res.get('target_audience', 'N/A'))
                    
                    # Technical Details (Expander)
                    with st.expander("⚙️ Technical Details"):
                        st.json({
                            "Structure": specs.get('structure', 'N/A'),
                            "Tone": specs.get('tone', 'N/A'),
                            "Extras": specs.get('extras', 'N/A')
                        })
                
                elif agent_type == "Document Analysis Agent":
                    doc_type = res.get('document_type', 'General')
                    st.markdown(f"**📋 Document Type:** `{doc_type}`")
                    
                    st.subheader("📝 Summary")
                    st.info(res.get('summary', 'N/A'))
                    
                    # Structure and Metrics
                    structure = res.get('structure', {})
                    if structure:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Word Count", structure.get('word_count', 'N/A'))
                        with col2:
                            st.metric("Page Count", structure.get('page_count', 'N/A'))
                        with col3:
                            st.metric("Section Count", len(structure.get('sections', [])))
                    
                    # Sentiment and Tone
                    col1, col2 = st.columns(2)
                    with col1:
                        sentiment = res.get('sentiment', 'N/A')
                        if 'Positive' in sentiment:
                            st.success(f"😊 Sentiment: {sentiment}")
                        elif 'Negative' in sentiment:
                            st.error(f"😞 Sentiment: {sentiment}")
                        else:
                            st.info(f"😐 Sentiment: {sentiment}")
                    with col2:
                        st.metric("🎭 Tone", res.get('tone', 'N/A'))
                    
                    st.subheader("🔑 Key Points")
                    for i, point in enumerate(res.get('key_points', []), 1):
                        st.markdown(f"**{i}.** {point}")
                    
                    # Entities
                    entities = res.get('entities', {})
                    if isinstance(entities, dict) and entities:
                        st.subheader("🏷️ Extracted Entities")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if entities.get('people'):
                                st.markdown("**👥 People:**")
                                for person in entities.get('people', []):
                                    st.markdown(f"- {person}")
                            
                            if entities.get('organizations'):
                                st.markdown("**🏢 Organizations:**")
                                for org in entities.get('organizations', []):
                                    st.markdown(f"- {org}")
                        
                        with col2:
                            if entities.get('dates'):
                                st.markdown("**📅 Dates:**")
                                for date in entities.get('dates', []):
                                    st.markdown(f"- {date}")
                            
                            if entities.get('locations'):
                                st.markdown("**📍 Locations:**")
                                for loc in entities.get('locations', []):
                                    st.markdown(f"- {loc}")
                        
                        if entities.get('numbers'):
                            st.markdown("**🔢 Important Numbers:**")
                            for num in entities.get('numbers', []):
                                st.markdown(f"- {num}")
                    
                    # Key Insights
                    if res.get('key_insights'):
                        st.subheader("💡 Key Insights")
                        for insight in res.get('key_insights', []):
                            st.markdown(f"✨ {insight}")
                    
                    # Missing Info
                    if res.get('missing_info'):
                        st.subheader("⚠️ Missing/Unclear Information")
                        for missing in res.get('missing_info', []):
                            st.warning(missing)
                    
                    # Document Structure
                    if structure.get('sections'):
                        with st.expander("📑 Document Structure"):
                            for i, section in enumerate(structure.get('sections', []), 1):
                                st.markdown(f"{i}. {section}")
                    
                    st.subheader("🎯 Recommendations")
                    for i, rec in enumerate(res.get('recommendations', []), 1):
                        st.success(f"**{i}.** {rec}")
                
                elif agent_type == "Code Assistant Agent":
                    lang = result.get('language', 'python').lower()
                    
                    st.subheader(f"💻 {result.get('language', 'Python')} Code")
                    st.code(res.get('code', 'N/A'), language=lang)
                    
                    # Complexity Metrics
                    complexity = res.get('complexity', {})
                    if isinstance(complexity, dict):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("⏱️ Time Complexity", complexity.get('time', 'N/A'))
                        with col2:
                            st.metric("💾 Space Complexity", complexity.get('space', 'N/A'))
                    else:
                        st.metric("Complexity", complexity)
                    
                    st.subheader("📝 Explanation")
                    st.markdown(f"<div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px;'>{res.get('explanation', 'N/A')}</div>", unsafe_allow_html=True)
                    
                    # Test Cases
                    st.subheader("🧪 Test Cases")
                    for i, test in enumerate(res.get('test_cases', []), 1):
                        if '✅' in test or 'successful' in test.lower():
                            st.success(f"**Test {i}:** {test}")
                        else:
                            st.info(f"**Test {i}:** {test}")
                    
                    # Best Practices
                    if res.get('best_practices'):
                        st.subheader("✨ Applied Best Practices")
                        for practice in res.get('best_practices', []):
                            st.markdown(f"✓ {practice}")
                    
                    # Dependencies
                    if res.get('dependencies'):
                        st.subheader("📦 Required Dependencies")
                        deps = res.get('dependencies', [])
                        st.code(", ".join(deps))
                    
                    # Security Notes
                    if res.get('security_notes'):
                        st.subheader("🔒 Security Notes")
                        for note in res.get('security_notes', []):
                            st.warning(note)
                    
                    # Improvement Suggestions
                    st.subheader("⚡ Improvement Suggestions")
                    for i, imp in enumerate(res.get('improvements', []), 1):
                        st.markdown(f"**{i}.** {imp}")
                    
                    else:
                        # Fallback: show JSON
                        st.json(res)
                    
                    # For those who want to see the raw JSON
                    with st.expander("🔍 Raw JSON Data"):
                        st.json(res)
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown')}")
    else:
        st.info("📭 No results yet")

# TAB 3: Logs
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
        
        if st.button("🗑️ Clear Logs"):
            st.session_state.execution_logs = []
            st.rerun()
    else:
        st.info("📭 No logs yet")

# Footer
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Agents", len(st.session_state.agents))
with col2:
    st.metric("Completed Tasks", len(st.session_state.task_results))
with col3:
    success_count = len([r for r in st.session_state.task_results if r.get('status') == 'success'])
    if st.session_state.task_results:
        success_rate = int(success_count / len(st.session_state.task_results) * 100)
        st.metric("Success Rate", f"{success_rate}%")
    else:
        st.metric("Success Rate", "N/A")

st.markdown("""
<div style='text-align: center; color: gray; margin-top: 20px;'>
    <p>🤖 AI Agent Control Panel v5.0 | Gemini 1.5 Flash + Advanced Tools</p>
    <p style='font-size: 0.8em;'>✨ Google Search | 🐍 Code Execution | 📋 Structured Output | 📄 Document Understanding</p>
</div>
""", unsafe_allow_html=True)