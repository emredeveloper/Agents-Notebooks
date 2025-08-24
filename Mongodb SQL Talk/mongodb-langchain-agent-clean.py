#!/usr/bin/env python3
"""
MongoDB LangChain Agent with LM Studio Integration - Clean Version
Enhanced with comprehensive fallback system and improved error handling
"""

import os
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Import required packages
from pymongo import MongoClient
from langchain.tools import BaseTool
from langchain.llms.base import LLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult
from pydantic import Field
import requests
from flask import Flask, request, jsonify, render_template_string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LM Studio LLM Integration
class LMStudioLLM(LLM):
    """Custom LLM class for LM Studio integration"""
    base_url: str = Field(default="http://localhost:1234/v1")
    model_name: str = Field(default="qwen/qwen3-4b-2507")
    
    @property
    def _llm_type(self) -> str:
        return "lmstudio"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"LM Studio API error: {response.status_code}")
                return "Error: Unable to get response from LM Studio"
                
        except Exception as e:
            logger.error(f"LM Studio connection error: {str(e)}")
            return f"Error connecting to LM Studio: {str(e)}"

# MongoDB Tools
class MongoDBConnectionTool(BaseTool):
    """Tool for connecting to MongoDB database"""
    name: str = "mongodb_connection"
    description: str = "Connect to MongoDB database and set active database"
    mongo_uri: str = Field(default="mongodb://localhost:27017/")
    client: Any = Field(default=None)
    db: Any = Field(default=None)
    
    def _run(self, database_name: str = "agent") -> str:
        try:
            if self.client is None:
                self.client = MongoClient(self.mongo_uri)
                logger.info(f"Connected to MongoDB at {self.mongo_uri}")
            
            self.db = self.client[database_name]
            
            # Test connection (use admin database, avoid treating a collection as callable)
            try:
                # 'ping' is preferred over deprecated 'ismaster'
                self.client.admin.command('ping')
            except Exception as ping_err:
                logger.warning(f"Ping command failed: {ping_err}")
            
            return json.dumps({
                "success": True,
                "message": f"Connected to database: {database_name}",
                "database": database_name
            }, indent=2)
            
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"success": False, "error": error_msg}, indent=2)

class MongoDBQueryTool(BaseTool):
    """Enhanced MongoDB query tool with comprehensive natural language processing"""
    name: str = "mongodb_query"
    description: str = "Execute MongoDB queries with advanced filtering, sorting, and aggregation"
    connection_tool: Any = Field(default=None)
    
    def __init__(self, connection_tool: MongoDBConnectionTool, **kwargs):
        super().__init__(connection_tool=connection_tool, **kwargs)
        
    def _run(self, query_params: str) -> str:
        if self.connection_tool.db is None:
            return json.dumps({"success": False, "error": "No database connection"}, indent=2)
            
        try:
            params = json.loads(query_params)
            collection = self.connection_tool.db[params.get("collection", "users")]
            
            # Check if collection is empty and add sample data if needed
            if collection.count_documents({}) == 0:
                self._add_sample_data(collection, params.get("collection", "users"))
            
            # Execute query based on type
            if params.get("operation") == "count":
                return self._execute_count(collection, params)
            elif params.get("operation") == "distinct":
                return self._execute_distinct(collection, params)
            elif params.get("operation") == "aggregate":
                return self._execute_aggregation(collection, params)
            else:
                return self._execute_find(collection, params)
                
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            return json.dumps({"success": False, "error": str(e)}, indent=2)
    
    def _execute_find(self, collection, params: Dict) -> str:
        """Execute find operation with advanced filtering"""
        try:
            filter_obj = params.get("filter", {})
            limit = params.get("limit", 10)
            sort_criteria = params.get("sort", [])
            projection = params.get("projection", {})
            
            cursor = collection.find(filter_obj, projection)
            
            if sort_criteria:
                cursor = cursor.sort(sort_criteria)
            
            results = list(cursor.limit(limit))
            
            # Convert ObjectId to string
            for result in results:
                if "_id" in result:
                    result["_id"] = str(result["_id"])
                # Format datetime objects
                for key, value in result.items():
                    if isinstance(value, datetime):
                        result[key] = value.strftime("%Y-%m-%d %H:%M:%S.%f")
            
            return json.dumps({
                "success": True,
                "results": results,
                "count": len(results),
                "total": collection.count_documents(filter_obj),
                "collection": collection.name
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
    
    def _execute_count(self, collection, params: Dict) -> str:
        """Execute count operation"""
        try:
            filter_obj = params.get("filter", {})
            count = collection.count_documents(filter_obj)
            
            return json.dumps({
                "success": True,
                "count": count,
                "collection": collection.name,
                "filter": filter_obj
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
    
    def _execute_distinct(self, collection, params: Dict) -> str:
        """Execute distinct operation"""
        try:
            field = params.get("field", "name")
            filter_obj = params.get("filter", {})
            
            distinct_values = collection.distinct(field, filter_obj)
            
            return json.dumps({
                "success": True,
                "values": distinct_values,
                "count": len(distinct_values),
                "field": field,
                "collection": collection.name
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
    
    def _execute_aggregation(self, collection, params: Dict) -> str:
        """Execute aggregation pipeline"""
        try:
            pipeline = params.get("pipeline", [])
            
            results = list(collection.aggregate(pipeline))
            
            # Convert ObjectId to string
            for result in results:
                if "_id" in result:
                    result["_id"] = str(result["_id"]) if result["_id"] else None
            
            return json.dumps({
                "success": True,
                "results": results,
                "count": len(results),
                "collection": collection.name
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
    
    def _add_sample_data(self, collection, collection_name: str):
        """Add intelligent sample data when collection is empty"""
        if collection_name == "users":
            sample_users = [
                {"name": "Ahmet", "surname": "Yılmaz", "age": 25, "email": "ahmet@example.com", "city": "İstanbul", "created_at": datetime.now()},
                {"name": "Ayşe", "surname": "Kaya", "age": 30, "email": "ayse@example.com", "city": "Ankara", "created_at": datetime.now()},
                {"name": "Mehmet", "surname": "Demir", "age": 35, "email": "mehmet@example.com", "city": "İzmir", "created_at": datetime.now()},
                {"name": "Fatma", "surname": "Şahin", "age": 28, "email": "fatma@example.com", "city": "Bursa", "created_at": datetime.now()},
                {"name": "Ali", "surname": "Öz", "age": 32, "email": "ali@example.com", "city": "Antalya", "created_at": datetime.now()}
            ]
            collection.insert_many(sample_users)
            logger.info(f"Added {len(sample_users)} sample users to {collection_name}")

class MongoDBInsertTool(BaseTool):
    """Tool for inserting data into MongoDB"""
    name: str = "mongodb_insert"
    description: str = "Insert new documents into MongoDB collections"
    connection_tool: Any = Field(default=None)
    
    def __init__(self, connection_tool: MongoDBConnectionTool, **kwargs):
        super().__init__(connection_tool=connection_tool, **kwargs)
        
    def _run(self, insert_params: str) -> str:
        if self.connection_tool.db is None:
            return json.dumps({"success": False, "error": "No database connection"}, indent=2)
            
        try:
            params = json.loads(insert_params)
            collection_name = params.get("collection", "users")
            data = params.get("data", [])
            count = params.get("count", 5)
            
            collection = self.connection_tool.db[collection_name]
            
            if not data:
                data = self._generate_sample_data(collection_name, count)
            
            result = collection.insert_many(data)
            
            return json.dumps({
                "success": True,
                "inserted_count": len(result.inserted_ids),
                "collection": collection_name,
                "message": f"Successfully inserted {len(result.inserted_ids)} documents"
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
    
    def _generate_sample_data(self, collection_name: str, count: int) -> List[Dict]:
        """Generate sample data for insertion"""
        turkish_names = ["Ahmet", "Mehmet", "Ali", "Mustafa", "Emre", "Ayşe", "Fatma", "Elif", "Zeynep", "Selin"]
        turkish_surnames = ["Yılmaz", "Kaya", "Demir", "Şahin", "Öz", "Arslan", "Doğan", "Kara", "Çelik", "Yıldız"]
        cities = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya", "Gaziantep"]
        
        data = []
        for i in range(count):
            if collection_name == "users":
                data.append({
                    "name": random.choice(turkish_names),
                    "surname": random.choice(turkish_surnames),
                    "age": random.randint(18, 65),
                    "email": f"user{i+1}@example.com",
                    "city": random.choice(cities),
                    "created_at": datetime.now()
                })
        
        return data

class MongoDBSchemaAnalyzer(BaseTool):
    """Tool for analyzing MongoDB schemas and collections"""
    name: str = "mongodb_schema"
    description: str = "Analyze MongoDB collection schemas and list available collections"
    connection_tool: Any = Field(default=None)
    
    def __init__(self, connection_tool: MongoDBConnectionTool, **kwargs):
        super().__init__(connection_tool=connection_tool, **kwargs)
        
    def _run(self, collection_name: Optional[str] = None) -> str:
        if self.connection_tool.db is None:
            return json.dumps({"success": False, "error": "No database connection"}, indent=2)
            
        try:
            if collection_name:
                # Analyze specific collection
                collection = self.connection_tool.db[collection_name]
                total_docs = collection.count_documents({})
                
                if total_docs == 0:
                    return json.dumps({
                        "collection": collection_name,
                        "status": "empty",
                        "total_documents": 0,
                        "message": "Collection is empty"
                    }, indent=2)
                
                # Sample documents to understand schema
                sample = list(collection.find().limit(3))
                fields = set()
                for doc in sample:
                    fields.update(doc.keys())
                
                return json.dumps({
                    "collection": collection_name,
                    "total_documents": total_docs,
                    "fields": list(fields),
                    "sample_data": sample[0] if sample else None
                }, indent=2, default=str)
            else:
                # List all collections
                collections = self.connection_tool.db.list_collection_names()
                
                collection_info = []
                for coll_name in collections:
                    coll = self.connection_tool.db[coll_name]
                    doc_count = coll.count_documents({})
                    collection_info.append({
                        "name": coll_name,
                        "document_count": doc_count
                    })
                
                return json.dumps({
                    "success": True,
                    "database": self.connection_tool.db.name,
                    "collections": collection_info,
                    "total_collections": len(collections)
                }, indent=2)
                
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)

# Main Agent Class
class MongoDBLangChainAgent:
    """Main MongoDB Agent with LangChain and LM Studio integration"""
    
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", 
                 lm_studio_url: str = "http://localhost:1234/v1",
                 model_name: str = "qwen/qwen3-4b-2507"):
        
        # Initialize LLM
        self.llm = LMStudioLLM(base_url=lm_studio_url, model_name=model_name)
        
        # Initialize tools
        self.connection_tool = MongoDBConnectionTool(mongo_uri=mongo_uri)
        self.query_tool = MongoDBQueryTool(connection_tool=self.connection_tool)
        self.insert_tool = MongoDBInsertTool(connection_tool=self.connection_tool)
        self.schema_tool = MongoDBSchemaAnalyzer(connection_tool=self.connection_tool)
        
        self.tools = [
            self.connection_tool,
            self.query_tool,
            self.insert_tool,
            self.schema_tool
        ]
        
        # Create agent prompt
        self.prompt = PromptTemplate.from_template(
            """Sen MongoDB veritabanı uzmanı bir AI asistanısın. Kullanıcının sorularını analiz et ve uygun MongoDB araçlarını kullan.

Mevcut araçlar:
- mongodb_connection: Veritabanına bağlan
- mongodb_query: Verileri sorgula, filtrele, sırala
- mongodb_insert: Yeni veri ekle
- mongodb_schema: Koleksiyonları listele ve şemayı analiz et

Kullanıcı soruları için:
1. Önce veritabanına bağlan (mongodb_connection)
2. Sorguya göre uygun aracı seç
3. Sonuçları Türkçe olarak açıkla

Örnekler:
- "users ilk 5 veriyi göster" → mongodb_query ile users koleksiyonundan 5 kayıt getir
- "adı Ayşe olanları listele" → mongodb_query ile name filter kullan
- "koleksiyonları listele" → mongodb_schema ile tüm koleksiyonları göster
- "5 kullanıcı ekle" → mongodb_insert ile 5 örnek kullanıcı oluştur

{tools}

Kurallar:
1. Asla aynı yanıtta hem Action hem de Final Answer verme. Eğer daha fazla araca ihtiyaç varsa Final Answer yazma.
2. Final Answer yalnızca gerekli tüm Action/Observation adımları tamamlandıktan sonra gelir ve içinde başka Action/Observation içermez.
3. Final Answer bölümünde sadece kullanıcıya nihai Türkçe açıklama ve gerekiyorsa özet tablo anlatımı olsun.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: kullanıcıya final cevap

Question: {input}
Thought: {agent_scratchpad}"""
        )
        
        # Create agent
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=60,
            return_intermediate_steps=True,
            early_stopping_method="generate"
        )
        
    def process_query(self, user_input: str, db_name: str = "agent") -> Dict[str, Any]:
        """Process user query with enhanced error handling and fallback system"""
        try:
            # Ensure connection
            if self.connection_tool.db is None or self.connection_tool.db.name != db_name:
                connect_result = self.connection_tool._run(db_name)
                if "Error" in connect_result:
                    return {"error": connect_result}
            
            # Try agent first
            try:
                result = self.agent_executor.invoke({"input": user_input})
                
                agent_output = result.get("output", "")
                intermediate_steps = result.get("intermediate_steps", [])
                
                # Extract structured data from intermediate steps
                structured_data = None
                for step in intermediate_steps:
                    action, observation = step
                    if action.tool == "mongodb_query":
                        try:
                            parsed_obs = json.loads(observation)
                            if parsed_obs.get("success") and parsed_obs.get("results"):
                                structured_data = parsed_obs["results"]
                                break
                        except:
                            pass
                
                if structured_data:
                    return {
                        "success": True,
                        "type": "query",
                        "response": f"📋 Sorgu Sonucu:",
                        "data": structured_data,
                        "agent_response": agent_output
                    }
                
                return {
                    "success": True,
                    "type": "agent",
                    "response": agent_output,
                    "data": None
                }
                
            except Exception as agent_error:
                logger.error(f"Agent execution error: {str(agent_error)}")
                
                # Enhanced fallback system
                user_input_lower = user_input.lower()
                return self._handle_fallback_query(user_input, user_input_lower, str(agent_error))
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"error": f"❌ Sistem hatası: {str(e)}"}
    
    def _handle_fallback_query(self, user_input: str, user_input_lower: str, agent_error: str) -> Dict[str, Any]:
        """Comprehensive fallback system for failed agent queries"""
        import re
        
        # 1. Basic user listing queries
        if any(word in user_input_lower for word in ["users", "kullanıcı"]) and any(word in user_input_lower for word in ["göster", "listele", "show", "list"]):
            limit_match = re.search(r'(\d+)', user_input_lower)
            limit = int(limit_match.group(1)) if limit_match else 10
            
            query_params = json.dumps({
                "collection": "users",
                "filter": {},
                "limit": limit,
                "sort": [["created_at", -1]]
            })
            
            result = self.query_tool._run(query_params)
            try:
                result_data = json.loads(result)
                if result_data.get("success") and result_data.get("results"):
                    return {
                        "success": True,
                        "type": "query",
                        "response": f"📋 Users Koleksiyonu - İlk {limit} kayıt:",
                        "data": result_data["results"][:limit]
                    }
            except:
                pass
        
        # 2. Name-based search
        elif any(word in user_input_lower for word in ["adı", "adi", "ismi", "name"]):
            name_patterns = [
                r"adı\s+([^\s]+)",
                r"adi\s+([^\s]+)", 
                r"ismi\s+([^\s]+)",
                r"name\s+([^\s]+)"
            ]
            
            name_value = None
            for pattern in name_patterns:
                match = re.search(pattern, user_input_lower)
                if match:
                    name_value = match.group(1)
                    break
            
            if name_value:
                query_params = json.dumps({
                    "collection": "users",
                    "filter": {"name": {"$regex": f"^{name_value}$", "$options": "i"}},
                    "limit": 20
                })
                
                result = self.query_tool._run(query_params)
                try:
                    result_data = json.loads(result)
                    if result_data.get("success") and result_data.get("results"):
                        return {
                            "success": True,
                            "type": "query",
                            "response": f"📋 '{name_value}' isimli kullanıcılar:",
                            "data": result_data["results"]
                        }
                    else:
                        return {
                            "success": True,
                            "type": "no_data",
                            "response": f"❌ '{name_value}' isimli kullanıcı bulunamadı"
                        }
                except:
                    pass
        
        # 3. Count queries
        elif any(word in user_input_lower for word in ["kaç", "count", "sayı", "how many"]):
            try:
                result = self.connection_tool.db.users.count_documents({})
                return {
                    "success": True,
                    "type": "count",
                    "response": f"📊 Toplam kullanıcı sayısı: {result}"
                }
            except:
                pass
        
        # 4. Collections listing
        elif any(word in user_input_lower for word in ["koleksiyon", "collection", "tablo"]):
            result = self.schema_tool._run()
            try:
                result_data = json.loads(result)
                return {
                    "success": True,
                    "type": "collections",
                    "response": f"📁 Mevcut Koleksiyonlar:",
                    "data": result_data.get("collections", [])
                }
            except:
                pass
        
        # 5. Empty collection check
        elif "boş" in user_input_lower or "empty" in user_input_lower:
            collection_name = "users"
            if "agent-sql" in user_input_lower:
                collection_name = "agent-sql"
            
            try:
                count = self.connection_tool.db[collection_name].count_documents({})
                return {
                    "success": True,
                    "type": "check",
                    "response": f"📊 {collection_name} koleksiyonu {'boş' if count == 0 else f'{count} kayıt içeriyor'}"
                }
            except:
                pass
        
        # Default error response
        return {
            "success": False,
            "type": "error",
            "response": f"❌ Agent hatası: {agent_error}. Lütfen sorgunuzu daha açık yazın.",
            "suggestion": "Örnek: '5 users verisi göster' veya 'koleksiyonları listele'"
        }
    
    def get_collections(self, db_name: str = "agent") -> Dict[str, Any]:
        """Get list of collections in database"""
        try:
            if self.connection_tool.db is None or self.connection_tool.db.name != db_name:
                connect_result = self.connection_tool._run(db_name)
                if "Error" in connect_result:
                    return {"error": connect_result}
            
            result = self.schema_tool._run()
            return {"success": True, "data": json.loads(result)}
            
        except Exception as e:
            return {"error": f"Error getting collections: {str(e)}"}

# Flask Web Interface
app = Flask(__name__)
agent = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MongoDB LangChain Agent - Enhanced</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); overflow: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
        .chat-container { height: 500px; overflow-y: auto; padding: 20px; border-bottom: 1px solid #eee; }
        .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
        .user-message { background: #e3f2fd; margin-left: 50px; }
        .bot-message { background: #f1f8e9; margin-right: 50px; }
        .data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        .data-table th, .data-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .data-table th { background-color: #f2f2f2; }
        .input-container { padding: 20px; background: #fafafa; }
        .input-group { display: flex; gap: 10px; }
        .input-field { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 16px; }
        .send-button { padding: 12px 24px; background: #667eea; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; }
        .send-button:hover { background: #5a67d8; }
        .stats { display: flex; justify-content: space-around; padding: 15px; background: #f8f9fa; font-size: 14px; }
        .loading { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 MongoDB LangChain Agent</h1>
            <p>Enhanced with Smart Fallback System & LM Studio Integration</p>
        </div>
        
        <div class="stats">
            <div>📊 Model: qwen/qwen3-4b-2507</div>
            <div>🔗 Database: agent</div>
            <div>⚡ Status: Ready</div>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message bot-message">
                <strong>🤖 MongoDB Agent:</strong> Merhaba! MongoDB sorularınızı yanıtlamaya hazırım. 
                <br><br>Örnek komutlar:
                <ul>
                    <li>"users ilk 5 veriyi göster"</li>
                    <li>"adı Ayşe olanları listele"</li>
                    <li>"kaç kullanıcı var"</li>
                    <li>"koleksiyonları listele"</li>
                    <li>"5 kullanıcı ekle"</li>
                </ul>
            </div>
        </div>
        
        <div class="input-container">
            <div class="input-group">
                <input type="text" id="queryInput" class="input-field" placeholder="MongoDB sorgunuzu yazın..." onkeypress="handleKeyPress(event)">
                <button class="send-button" onclick="sendQuery()">Gönder</button>
            </div>
        </div>
    </div>

    <script>
        function addMessage(content, isUser = false) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            messageDiv.innerHTML = `<strong>${isUser ? '👤 Sen' : '🤖 MongoDB Agent'}:</strong> ${content}`;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function formatData(data) {
            if (!data || data.length === 0) return 'Veri bulunamadı';
            
            let html = '<table class="data-table"><thead><tr>';
            const keys = Object.keys(data[0]);
            keys.forEach(key => {
                if (key !== '_id') html += `<th>${key}</th>`;
            });
            html += '</tr></thead><tbody>';
            
            data.slice(0, 10).forEach(item => {
                html += '<tr>';
                keys.forEach(key => {
                    if (key !== '_id') html += `<td>${item[key] || ''}</td>`;
                });
                html += '</tr>';
            });
            html += '</tbody></table>';
            
            if (data.length > 10) {
                html += `<p><em>... ve ${data.length - 10} kayıt daha</em></p>`;
            }
            
            return html;
        }

        async function sendQuery() {
            const input = document.getElementById('queryInput');
            const query = input.value.trim();
            
            if (!query) return;
            
            addMessage(query, true);
            input.value = '';
            
            addMessage('<span class="loading">Sorgu işleniyor...</span>');
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const result = await response.json();
                
                // Remove loading message
                const messages = document.querySelectorAll('.message');
                const lastMessage = messages[messages.length - 1];
                if (lastMessage.innerHTML.includes('loading')) {
                    lastMessage.remove();
                }
                
                let responseContent = result.response || 'Sonuç alınamadı';
                
                if (result.data && result.data.length > 0) {
                    responseContent += '<br><br>' + formatData(result.data);
                }
                
                if (result.error) {
                    responseContent = `❌ Hata: ${result.error}`;
                }
                
                addMessage(responseContent);
                
            } catch (error) {
                const messages = document.querySelectorAll('.message');
                const lastMessage = messages[messages.length - 1];
                if (lastMessage.innerHTML.includes('loading')) {
                    lastMessage.remove();
                }
                addMessage(`❌ Bağlantı hatası: ${error.message}`);
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendQuery();
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.get_json()
        query = data.get("query", "")
        
        if not query:
            return jsonify({"error": "Query is required"})
        
        result = agent.process_query(query)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/collections")
def get_collections():
    try:
        result = agent.get_collections()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

def main():
    global agent
    
    print("🚀 MongoDB LangChain Agent with LM Studio - Enhanced Version")
    print("=" * 60)
    
    # Initialize agent
    agent = MongoDBLangChainAgent(
        mongo_uri="mongodb://localhost:27017/",
        lm_studio_url="http://localhost:1234/v1",
        model_name="qwen/qwen3-4b-2507"
    )
    
    print("✅ Agent initialized successfully")
    print("🌐 Starting Flask web server...")
    print("📱 Web interface: http://localhost:5000")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()
