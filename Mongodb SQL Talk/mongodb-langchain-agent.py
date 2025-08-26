import os
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S.%f")
        return super().default(obj)

# Import required packages
from pymongo import MongoClient
from langchain.tools import BaseTool
from langchain.llms.base import LLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
# Removed unused import: LLMResult
from pydantic import Field
import requests
from flask import Flask, request, jsonify, render_template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LM Studio LLM Integration
class LMStudioLLM(LLM):
    """Custom LLM class for LM Studio integration"""
    base_url: str = Field(default="http://localhost:1234/v1")
    model_name: str = Field(default="qwen/qwen3-4b-2507")  # Updated to preferred model
    
    @property
    def _llm_type(self) -> str:
        return "lmstudio"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.01,
                "max_tokens": 500,
                "stream": False,
                "stop": ["Final Answer:", "Observation:"]
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
            # Handle both string and dict inputs
            if isinstance(query_params, str):
                params = json.loads(query_params)
            else:
                params = query_params
            
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
            }, indent=2, cls=DateTimeEncoder)
            
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
            }, indent=2, cls=DateTimeEncoder)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
    
    def _add_sample_data(self, collection, collection_name: str):
        """Add intelligent sample data when collection is empty based on collection name"""
        from datetime import datetime
        
        sample_data = []
        
        # Generate sample data based on collection name patterns
        if any(pattern in collection_name.lower() for pattern in ['user', 'kullanıcı', 'person', 'kişi']):
            sample_data = [
                {"name": "Ahmet", "surname": "Yılmaz", "age": 25, "email": "ahmet@example.com", "city": "İstanbul", "created_at": datetime.now()},
                {"name": "Ayşe", "surname": "Kaya", "age": 30, "email": "ayse@example.com", "city": "Ankara", "created_at": datetime.now()},
                {"name": "Mehmet", "surname": "Demir", "age": 35, "email": "mehmet@example.com", "city": "İzmir", "created_at": datetime.now()},
            ]
        elif any(pattern in collection_name.lower() for pattern in ['product', 'ürün', 'item']):
            sample_data = [
                {"name": "Laptop", "price": 15000, "category": "Electronics", "stock": 50, "created_at": datetime.now()},
                {"name": "Phone", "price": 8000, "category": "Electronics", "stock": 100, "created_at": datetime.now()},
                {"name": "Book", "price": 25, "category": "Education", "stock": 200, "created_at": datetime.now()},
            ]
        elif any(pattern in collection_name.lower() for pattern in ['order', 'sipariş']):
            sample_data = [
                {"order_id": "ORD001", "customer": "Ahmet Y.", "amount": 150, "status": "completed", "created_at": datetime.now()},
                {"order_id": "ORD002", "customer": "Ayşe K.", "amount": 250, "status": "pending", "created_at": datetime.now()},
            ]
        else:
            # Generic sample data
            sample_data = [
                {"title": "Sample Item 1", "description": "This is a sample", "value": 100, "created_at": datetime.now()},
                {"title": "Sample Item 2", "description": "Another sample", "value": 200, "created_at": datetime.now()},
            ]
        
        if sample_data:
            collection.insert_many(sample_data)
            logger.info(f"Added {len(sample_data)} sample documents to {collection_name}")

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
            }, indent=2, cls=DateTimeEncoder)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
    
    def _generate_sample_data(self, collection_name: str, count: int) -> List[Dict]:
        """Generate sample data using the centralized method"""
        # Reuse the existing sample data generation logic
        temp_collection = type('MockCollection', (), {})()
        self.connection_tool.db[collection_name]._add_sample_data = lambda: None
        
        # Use the centralized bulk generation method from the agent
        from datetime import datetime
        import random
        
        if 'user' in collection_name.lower():
            names = ["Ahmet", "Ayşe", "Mehmet", "Fatma", "Ali", "Zeynep"]
            surnames = ["Yılmaz", "Kaya", "Demir", "Şahin", "Öz", "Arslan"]
            cities = ["İstanbul", "Ankara", "İzmir", "Bursa"]
            
            data = []
            for i in range(count):
                data.append({
                    "name": random.choice(names),
                    "surname": random.choice(surnames),
                    "age": random.randint(18, 65),
                    "email": f"user{i+1}@example.com",
                    "city": random.choice(cities),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            return data
        
        return []

class MongoDBUpdateTool(BaseTool):
    """Tool for updating documents in MongoDB"""
    name: str = "mongodb_update"
    description: str = "Update existing documents in MongoDB collections"
    connection_tool: Any = Field(default=None)
    
    def __init__(self, connection_tool: MongoDBConnectionTool, **kwargs):
        super().__init__(connection_tool=connection_tool, **kwargs)
        
    def _run(self, update_params: str) -> str:
        if self.connection_tool.db is None:
            return json.dumps({"success": False, "error": "No database connection"}, indent=2)
            
        try:
            params = json.loads(update_params)
            collection_name = params.get("collection", "users")
            filter_obj = params.get("filter", {})
            update_obj = params.get("update", {})
            upsert = params.get("upsert", False)
            
            collection = self.connection_tool.db[collection_name]
            
            # Use update_many for bulk updates
            result = collection.update_many(filter_obj, update_obj, upsert=upsert)
            
            return json.dumps({
                "success": True,
                "matched_count": result.matched_count,
                "modified_count": result.modified_count,
                "upserted_id": str(result.upserted_id) if result.upserted_id else None,
                "collection": collection_name,
                "message": f"Updated {result.modified_count} documents"
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)

class MongoDBDeleteTool(BaseTool):
    """Tool for deleting documents from MongoDB"""
    name: str = "mongodb_delete"
    description: str = "Delete documents from MongoDB collections"
    connection_tool: Any = Field(default=None)
    
    def __init__(self, connection_tool: MongoDBConnectionTool, **kwargs):
        super().__init__(connection_tool=connection_tool, **kwargs)
        
    def _run(self, delete_params: str) -> str:
        if self.connection_tool.db is None:
            return json.dumps({"success": False, "error": "No database connection"}, indent=2)
            
        try:
            params = json.loads(delete_params)
            collection_name = params.get("collection", "users")
            filter_obj = params.get("filter", {})
            
            if not filter_obj:
                return json.dumps({"success": False, "error": "Filter is required for delete operations"}, indent=2)
            
            collection = self.connection_tool.db[collection_name]
            
            # Use delete_many for bulk deletes
            result = collection.delete_many(filter_obj)
            
            return json.dumps({
                "success": True,
                "deleted_count": result.deleted_count,
                "collection": collection_name,
                "message": f"Deleted {result.deleted_count} documents"
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)

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
                 model_name: str = "google/gemma-3n-e4b"):  # Updated to preferred model
        
        # Initialize collection context
        self.selected_collection = None
        
        # Initialize LLM
        self.llm = LMStudioLLM(base_url=lm_studio_url, model_name=model_name)
        
        # Initialize tools
        self.connection_tool = MongoDBConnectionTool(mongo_uri=mongo_uri)
        self.query_tool = MongoDBQueryTool(connection_tool=self.connection_tool)
        self.insert_tool = MongoDBInsertTool(connection_tool=self.connection_tool)
        self.update_tool = MongoDBUpdateTool(connection_tool=self.connection_tool)
        self.delete_tool = MongoDBDeleteTool(connection_tool=self.connection_tool)
        self.schema_tool = MongoDBSchemaAnalyzer(connection_tool=self.connection_tool)
        
        self.tools = [
            self.connection_tool,
            self.query_tool,
            self.insert_tool,
            self.update_tool,
            self.delete_tool,
            self.schema_tool
        ]
        
        # Create agent prompt - Strict parsing to prevent errors
        self.prompt = PromptTemplate.from_template(
            """Sen MongoDB veritabanı uzmanısın. Kullanıcı sorularını adım adım çöz.

Mevcut araçlar:
- mongodb_connection: Veritabanına bağlan
- mongodb_query: Veri sorgula (find, count, distinct, aggregate)
- mongodb_insert: Yeni veri ekle
- mongodb_update: Mevcut veriyi güncelle
- mongodb_delete: Veri sil
- mongodb_schema: Koleksiyon şemalarını analiz et

{tools}

KRITIK KURALLAR:
- Bir seferde SADECE bir şey yap
- Action yazdıysan Final Answer YAZMA
- Final Answer yazdıysan Action YAZMA

Format (kesinlikle uy):
Question: {input}
Thought: Ne yapmam gerek?
Action: [{tool_names}] birini seç
Action Input: JSON formatında input
Observation: aracın sonucu
Thought: Sonucu değerlendir
Final Answer: Kullanıcıya türkçe cevap

{agent_scratchpad}"""
        )
        
        # Create agent
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors="STOP! Format hatası: Bir seferde sadece Action VEYA Final Answer yaz. İkisini birlikte yazma!",
            max_iterations=3,
            max_execution_time=15,
            return_intermediate_steps=True,
            early_stopping_method="force"
        )
        
    def process_query(self, user_input: str, db_name: str = "agent") -> Dict[str, Any]:
        """Process user query with enhanced error handling, fallback system and collection context"""
        try:
            # Ensure connection
            if self.connection_tool.db is None or self.connection_tool.db.name != db_name:
                connect_result = self.connection_tool._run(db_name)
                if "Error" in connect_result:
                    return {"error": connect_result}
            
            # Log collection context
            if self.selected_collection:
                logger.info(f"Processing query with collection context: {self.selected_collection}")
            else:
                logger.info("Processing query with database context")
            
            # Enhanced fallback system - use directly for better reliability
            user_input_lower = user_input.lower()
            
            # Check for common patterns and use fallback directly
            if any(word in user_input_lower for word in ["göster", "listele", "show", "list", "ekle", "add", "insert", "oluştur", "create", "güncelle", "update", "sil", "delete", "kaç", "count", "koleksiyon", "collection"]):
                logger.info("Using direct fallback for user query")
                
                # Special handling for database-level queries when no collection is selected
                if not self.selected_collection and any(word in user_input_lower for word in ["kaç", "count"]) and any(word in user_input_lower for word in ["collection", "koleksiyon", "tablo"]):
                    logger.info("Database-level collection count query detected")
                
                return self._handle_fallback_query(user_input, user_input_lower, "Using direct fallback for reliability")
            
            # Try agent for complex queries only
            try:
                result = self.agent_executor.invoke({"input": user_input})
                
                agent_output = result.get("output", "")
                
                # Check if agent output is meaningful
                if "Agent stopped" in agent_output or "iteration limit" in agent_output or len(agent_output.strip()) < 10:
                    logger.warning("Agent output insufficient, using fallback")
                    return self._handle_fallback_query(user_input, user_input_lower, "Agent output insufficient")
                
                return {
                    "success": True,
                    "type": "agent",
                    "response": agent_output,
                    "data": None
                }
                
            except Exception as agent_error:
                logger.error(f"Agent execution error: {str(agent_error)}")
                return self._handle_fallback_query(user_input, user_input_lower, str(agent_error))
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"error": f"❌ Sistem hatası: {str(e)}"}
    
    def _detect_collection_from_query(self, user_query: str) -> str:
        """Detect which collection the user is referring to"""
        try:
            # Get available collections
            collections = self.connection_tool.db.list_collection_names()
            
            if not collections:
                return None
            
            # If a collection is selected in UI, prioritize it
            if self.selected_collection and self.selected_collection in collections:
                return self.selected_collection
            
            user_query_lower = user_query.lower()
            
            # Check for explicit collection mentions
            for collection in collections:
                if collection.lower() in user_query_lower:
                    return collection
            
            # Check for common collection patterns
            collection_patterns = {
                'users': ['user', 'kullanıcı', 'kişi', 'person'],
                'products': ['product', 'ürün', 'item'],
                'orders': ['order', 'sipariş', 'purchase'],
                'customers': ['customer', 'müşteri', 'client']
            }
            
            for collection in collections:
                for pattern_collection, patterns in collection_patterns.items():
                    if collection.lower().startswith(pattern_collection[:4]):  # Match first 4 chars
                        for pattern in patterns:
                            if pattern in user_query_lower:
                                return collection
            
            # If selected collection exists, use it as fallback
            if self.selected_collection and self.selected_collection in collections:
                return self.selected_collection
            
            # Default to first collection
            return collections[0]
            
        except Exception as e:
            logger.error(f"Collection detection error: {str(e)}")
            return None
    
    def _extract_query_parameters_with_llm(self, collection_name: str, user_query: str) -> Dict[str, Any]:
        """Use LLM to interpret the user's query and extract MongoDB query parameters"""
        try:
            # Get sample document to understand schema
            collection = self.connection_tool.db[collection_name]
            sample = list(collection.find().limit(1))
            sample_keys = []
            if sample:
                sample_keys = list(sample[0].keys())
            
            system_prompt = f"""
            MongoDB query generator için:
            Koleksiyon: {collection_name}
            Mevcut alanlar: {sample_keys}
            
            Kullanıcı sorgusu: "{user_query}"
            
            Bu koleksiyondaki alanlara göre SADECE geçerli JSON object ver:
            {{
                "filter": {{MongoDB filter koşulları - sadece mevcut alanları kullan}},
                "limit": sayı (varsayılan 10),
                "sort": [["alan", 1 veya -1]] (mevcut alanlardan),
                "projection": {{dahil edilecek alanlar - mevcut alanlardan}}
            }}
            
            Açıklama ekleme, sadece JSON.
            """
            
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.llm.model_name,
                "messages": [{"role": "user", "content": system_prompt}],
                "temperature": 0.1,
                "max_tokens": 500,
                "stream": False
            }
            
            response = requests.post(
                f"{self.llm.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                llm_response = response.json()["choices"][0]["message"]["content"]
                
                # Extract JSON from response
                import re
                json_match = re.search(r'({.*})', llm_response.replace('\n', ' '), re.DOTALL)
                if json_match:
                    try:
                        query_params = json.loads(json_match.group(1))
                        return query_params
                    except json.JSONDecodeError:
                        logger.error("Failed to parse LLM JSON response")
                        return None
            
            return None
            
        except Exception as e:
            logger.error(f"LLM query extraction error: {str(e)}")
            return None
    
    def _handle_insert_request(self, user_input: str, user_input_lower: str) -> Dict[str, Any]:
        """Handle insert requests using LLM to extract data dynamically"""
        import re
        from datetime import datetime
        
        # Detect target collection
        target_collection = self._detect_collection_from_query(user_input)
        
        if not target_collection:
            return {
                "success": False,
                "type": "error",
                "response": "❌ Hedef koleksiyon belirlenemedi. Veritabanında koleksiyon bulunamadı."
            }
        
        try:
            # Get sample document to understand schema
            collection = self.connection_tool.db[target_collection]
            sample = list(collection.find().limit(1))
            sample_fields = []
            if sample:
                sample_fields = list(sample[0].keys())
                # Remove _id from sample fields
                sample_fields = [f for f in sample_fields if f != '_id']
            
            # Use LLM to extract data from natural language
            system_prompt = f"""
            Kullanıcı "{target_collection}" koleksiyonuna yeni veri eklemek istiyor: "{user_input}"
            
            Mevcut koleksiyon alanları: {sample_fields}
            
            Bu metinden veri çıkar ve SADECE JSON formatında ver:
            {{
                {', '.join([f'"{field}": "değer"' for field in sample_fields[:5]])}
            }}
            
            - Sadece mevcut alanları kullan
            - Eksik bilgiler için mantıklı varsayılan değerler kullan
            - Sayısal alanlar için sayı, metin alanlar için string kullan
            Sadece JSON, açıklama yok.
            """
            
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.llm.model_name,
                "messages": [{"role": "user", "content": system_prompt}],
                "temperature": 0.1,
                "max_tokens": 300,
                "stream": False
            }
            
            response = requests.post(
                f"{self.llm.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                llm_response = response.json()["choices"][0]["message"]["content"]
                
                # Extract JSON from response
                json_match = re.search(r'({.*})', llm_response.replace('\n', ' '), re.DOTALL)
                if json_match:
                    try:
                        user_data = json.loads(json_match.group(1))
                        
                        # Add timestamp
                        user_data["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Insert the data
                        insert_params = {
                            "collection": target_collection,
                            "data": [user_data],
                            "count": 1
                        }
                        
                        result = self.insert_tool._run(json.dumps(insert_params))
                        result_data = json.loads(result)
                        
                        if result_data.get("success"):
                            return {
                                "success": True,
                                "type": "insert",
                                "response": f"✅ {target_collection} koleksiyonuna veri başarıyla eklendi",
                                "data": [user_data]
                            }
                        else:
                            return {
                                "success": False,
                                "type": "error",
                                "response": f"❌ {target_collection} koleksiyonuna veri eklenirken hata: {result_data.get('error', 'Bilinmeyen hata')}"
                            }
                            
                    except json.JSONDecodeError:
                        logger.error("Failed to parse LLM insert response")
            
            # Fallback to manual parsing
            return self._manual_insert_parsing(user_input, user_input_lower)
            
        except Exception as e:
            logger.error(f"Insert request error: {str(e)}")
            return self._manual_insert_parsing(user_input, user_input_lower)
    
    def _manual_insert_parsing(self, user_input: str, user_input_lower: str) -> Dict[str, Any]:
        """Manual parsing for insert requests as fallback - supports bulk inserts"""
        import re
        from datetime import datetime
        
        # Detect target collection
        target_collection = self._detect_collection_from_query(user_input)
        
        if not target_collection:
            return {
                "success": False,
                "type": "error",
                "response": "❌ Hedef koleksiyon belirlenemedi."
            }
        
        try:
            # Check for bulk insert requests (e.g., "5 satırlı", "10 veri")
            bulk_match = re.search(r'(\d+)\s*(satır|veri|kayıt|adet)', user_input_lower)
            count = int(bulk_match.group(1)) if bulk_match else 1
            
            # Check for column specification
            column_match = re.search(r'(\d+)\s*sütun', user_input_lower)
            column_count = int(column_match.group(1)) if column_match else 5
            
            # Generate sample data based on collection name and requirements
            sample_data = self._generate_bulk_sample_data(target_collection, count, column_count)
            
            if sample_data:
                insert_params = {
                    "collection": target_collection,
                    "data": sample_data,
                    "count": len(sample_data)
                }
                
                result = self.insert_tool._run(json.dumps(insert_params))
                result_data = json.loads(result)
                
                if result_data.get("success"):
                    return {
                        "success": True,
                        "type": "insert",
                        "response": f"✅ {target_collection} koleksiyonuna {len(sample_data)} veri başarıyla eklendi",
                        "data": sample_data[:3]  # Show first 3 for preview
                    }
                else:
                    return {
                        "success": False,
                        "type": "error",
                        "response": f"❌ Veri eklenirken hata: {result_data.get('error', 'Bilinmeyen hata')}"
                    }
            else:
                return {
                    "success": False,
                    "type": "error",
                    "response": "❌ Örnek veri oluşturulamadı."
                }
                
        except Exception as e:
            logger.error(f"Manual insert parsing error: {str(e)}")
            return {
                "success": False,
                "type": "error",
                "response": f"❌ Veri ekleme hatası: {str(e)}"
            }
    
    def _generate_bulk_sample_data(self, collection_name: str, count: int, column_count: int = 5) -> List[Dict]:
        """Generate bulk sample data for any collection"""
        from datetime import datetime
        import random
        
        sample_data = []
        
        # Common field generators
        def generate_fields(collection_type: str, num_fields: int) -> List[Dict]:
            data = []
            
            if 'user' in collection_type or 'person' in collection_type:
                names = ["Ahmet", "Ayşe", "Mehmet", "Fatma", "Ali", "Zeynep", "Emre", "Selin"]
                surnames = ["Yılmaz", "Kaya", "Demir", "Şahin", "Öz", "Arslan", "Doğan", "Çelik"]
                cities = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana"]
                
                for i in range(count):
                    record = {
                        "name": random.choice(names),
                        "surname": random.choice(surnames),
                        "age": random.randint(18, 65),
                        "email": f"user{i+1}@example.com",
                        "city": random.choice(cities),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    # Add extra fields if needed
                    if num_fields > 6:
                        record["phone"] = f"555-{random.randint(1000, 9999)}"
                        record["status"] = random.choice(["active", "inactive"])
                    data.append(record)
                    
            elif 'product' in collection_type or 'item' in collection_type:
                products = ["Laptop", "Phone", "Tablet", "Book", "Chair", "Desk", "Monitor"]
                categories = ["Electronics", "Furniture", "Books", "Accessories"]
                
                for i in range(count):
                    record = {
                        "name": f"{random.choice(products)} {i+1}",
                        "price": random.randint(50, 5000),
                        "category": random.choice(categories),
                        "stock": random.randint(0, 100),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    # Add extra fields
                    if num_fields > 5:
                        record["brand"] = random.choice(["Brand A", "Brand B", "Brand C"])
                        record["rating"] = round(random.uniform(1.0, 5.0), 1)
                    data.append(record)
                    
            else:
                # Generic data for unknown collection types
                for i in range(count):
                    record = {
                        "id": f"ID_{i+1:03d}",
                        "title": f"Sample Item {i+1}",
                        "description": f"This is sample description {i+1}",
                        "value": random.randint(1, 1000),
                        "status": random.choice(["active", "pending", "completed"]),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    # Add extra fields
                    if num_fields > 6:
                        record["category"] = random.choice(["Type A", "Type B", "Type C"])
                        record["priority"] = random.choice(["low", "medium", "high"])
                    data.append(record)
            
            return data
        
        return generate_fields(collection_name.lower(), column_count)
    
    def _handle_update_request(self, user_input: str, user_input_lower: str) -> Dict[str, Any]:
        """Handle update requests using LLM to extract update parameters"""
        import re
        
        # Detect target collection
        target_collection = self._detect_collection_from_query(user_input)
        
        if not target_collection:
            return {
                "success": False,
                "type": "error",
                "response": "❌ Hedef koleksiyon belirlenemedi."
            }
        
        try:
            # Get sample document to understand schema
            collection = self.connection_tool.db[target_collection]
            sample = list(collection.find().limit(1))
            sample_fields = []
            if sample:
                sample_fields = list(sample[0].keys())
                # Remove _id from sample fields
                sample_fields = [f for f in sample_fields if f != '_id']
            
            # Use LLM to extract update parameters from natural language
            system_prompt = f"""
            Kullanıcı "{target_collection}" koleksiyonunda güncelleme yapmak istiyor: "{user_input}"
            
            Mevcut koleksiyon alanları: {sample_fields}
            
            Bu metinden güncelleme parametrelerini çıkar ve SADECE JSON formatında ver:
            {{
                "filter": {{"hangi kayıtlar güncellenecek"}},
                "update": {{"$set": {{"hangi alanlar nasıl güncellenecek"}}}}
            }}
            
            Örnekler:
            - "sütunları ingilizce update et" → tüm Türkçe alan adlarını İngilizce karşılıklarına çevir
            - "yaşı 25 olanların şehrini İstanbul yap" → {{"filter": {{"age": 25}}, "update": {{"$set": {{"city": "İstanbul"}}}}}}
            
            Sadece JSON, açıklama yok.
            """
            
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.llm.model_name,
                "messages": [{"role": "user", "content": system_prompt}],
                "temperature": 0.1,
                "max_tokens": 500,
                "stream": False
            }
            
            response = requests.post(
                f"{self.llm.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                llm_response = response.json()["choices"][0]["message"]["content"]
                
                # Extract JSON from response
                json_match = re.search(r'({.*})', llm_response.replace('\n', ' '), re.DOTALL)
                if json_match:
                    try:
                        update_params = json.loads(json_match.group(1))
                        
                        # Handle special case: column name translation
                        if any(word in user_input_lower for word in ["sütun", "column", "ingilizce", "english"]):
                            return self._handle_column_translation(target_collection, sample_fields)
                        
                        # Execute the update
                        update_data = {
                            "collection": target_collection,
                            "filter": update_params.get("filter", {}),
                            "update": update_params.get("update", {}),
                            "upsert": False
                        }
                        
                        result = self.update_tool._run(json.dumps(update_data))
                        result_data = json.loads(result)
                        
                        if result_data.get("success"):
                            return {
                                "success": True,
                                "type": "update",
                                "response": f"✅ {target_collection} koleksiyonunda {result_data.get('modified_count', 0)} kayıt güncellendi",
                                "data": result_data
                            }
                        else:
                            return {
                                "success": False,
                                "type": "error",
                                "response": f"❌ Güncelleme hatası: {result_data.get('error', 'Bilinmeyen hata')}"
                            }
                            
                    except json.JSONDecodeError:
                        logger.error("Failed to parse LLM update response")
            
            # Fallback to manual parsing
            return self._manual_update_parsing(user_input, user_input_lower, target_collection)
            
        except Exception as e:
            logger.error(f"Update request error: {str(e)}")
            return {
                "success": False,
                "type": "error",
                "response": f"❌ Güncelleme hatası: {str(e)}"
            }
    
    def _handle_column_translation(self, collection_name: str, fields: List[str]) -> Dict[str, Any]:
        """Handle column name translation from Turkish to English"""
        try:
            # Turkish to English field mappings
            field_mappings = {
                "isim": "name",
                "ad": "name", 
                "soyisim": "surname",
                "soyad": "surname",
                "yaş": "age",
                "yas": "age",
                "eposta": "email",
                "mail": "email",
                "şehir": "city",
                "sehir": "city",
                "telefon": "phone",
                "adres": "address",
                "tarih": "date",
                "durum": "status",
                "fiyat": "price",
                "kategori": "category",
                "açıklama": "description",
                "aciklama": "description",
                "başlık": "title",
                "baslik": "title"
            }
            
            # Find fields that need translation
            updates_needed = []
            for field in fields:
                if field.lower() in field_mappings:
                    english_name = field_mappings[field.lower()]
                    updates_needed.append((field, english_name))
            
            if not updates_needed:
                return {
                    "success": True,
                    "type": "info",
                    "response": f"📋 {collection_name} koleksiyonunda çevrilecek Türkçe alan bulunamadı"
                }
            
            # Perform field renaming (this requires using MongoDB's $rename operator)
            collection = self.connection_tool.db[collection_name]
            total_updated = 0
            
            for old_field, new_field in updates_needed:
                # Use $rename to change field names
                result = collection.update_many(
                    {old_field: {"$exists": True}},  # Only update documents that have this field
                    {"$rename": {old_field: new_field}}
                )
                total_updated += result.modified_count
            
            return {
                "success": True,
                "type": "update",
                "response": f"✅ {collection_name} koleksiyonunda {len(updates_needed)} alan İngilizce'ye çevrildi. {total_updated} kayıt güncellendi",
                "data": {"translated_fields": dict(updates_needed), "updated_count": total_updated}
            }
            
        except Exception as e:
            logger.error(f"Column translation error: {str(e)}")
            return {
                "success": False,
                "type": "error", 
                "response": f"❌ Alan çevirme hatası: {str(e)}"
            }
    
    def _manual_update_parsing(self, user_input: str, user_input_lower: str, collection_name: str) -> Dict[str, Any]:
        """Manual parsing for update requests as fallback"""
        import re
        
        # Simple pattern matching for common update scenarios
        if "hepsini" in user_input_lower or "tümünü" in user_input_lower:
            # Update all records
            filter_obj = {}
        else:
            # Try to extract simple conditions
            filter_obj = {}
        
        return {
            "success": True,
            "type": "info",
            "response": f"🔄 {collection_name} için karmaşık güncelleme talebi. Lütfen daha spesifik olun veya basit JSON formatında belirtin."
        }
    
    def _handle_delete_request(self, user_input: str, user_input_lower: str) -> Dict[str, Any]:
        """Handle delete requests"""
        target_collection = self._detect_collection_from_query(user_input)
        
        if not target_collection:
            return {
                "success": False,
                "type": "error",
                "response": "❌ Hedef koleksiyon belirlenemedi."
            }
        
        # Simple delete example - can be enhanced with LLM parsing
        return {
            "success": True,
            "type": "info",
            "response": f"🗑️ {target_collection} koleksiyonu için silme özelliği geliştirilmekte. Güvenlik için manuel JSON ile silme yapabilirsiniz."
        }
    
    def _handle_fallback_query(self, user_input: str, user_input_lower: str, agent_error: str) -> Dict[str, Any]:
        """Enhanced fallback system with LLM-powered query understanding"""
        import re
        
        # PRIORITY 1: Database-level collection count queries (must be handled first)
        if any(word in user_input_lower for word in ["kaç", "count", "sayı", "how many"]) and any(word in user_input_lower for word in ["collection", "koleksiyon", "tablo"]):
            collections = self.connection_tool.db.list_collection_names()
            collection_counts = []
            for coll in collections:
                count = self.connection_tool.db[coll].count_documents({})
                collection_counts.append({"name": coll, "count": count})
            
            # Format collection details for better display
            collection_details = []
            for c in collection_counts:
                collection_details.append(f"{c['name']} ({c['count']} kayıt)")
            
            return {
                "success": True,
                "type": "collections_count",
                "response": f"📊 Veritabanında toplam {len(collections)} koleksiyon var:\n\n" + 
                          "\n".join([f"• {detail}" for detail in collection_details]),
                "data": collection_counts
            }
        
        # Check for different operations
        if any(word in user_input_lower for word in ["ekle", "add", "insert", "oluştur", "create"]):
            return self._handle_insert_request(user_input, user_input_lower)
        elif any(word in user_input_lower for word in ["güncelle", "update", "değiştir", "modify"]):
            return self._handle_update_request(user_input, user_input_lower)
        elif any(word in user_input_lower for word in ["sil", "delete", "remove", "kaldır"]):
            return self._handle_delete_request(user_input, user_input_lower)
        
        # Try LLM-powered query understanding for search operations
        else:
            # Detect target collection dynamically
            target_collection = self._detect_collection_from_query(user_input)
            
            if target_collection:
                llm_query_params = self._extract_query_parameters_with_llm(target_collection, user_input)
                
                if llm_query_params:
                    logger.info(f"Using LLM-generated query for {target_collection}: {llm_query_params}")
                    
                    # Execute the LLM-generated query
                    query_params = {
                        "collection": target_collection,
                        "filter": llm_query_params.get("filter", {}),
                        "limit": llm_query_params.get("limit", 10),
                        "sort": llm_query_params.get("sort", []),
                        "projection": llm_query_params.get("projection", {})
                    }
                    
                    result = self.query_tool._run(query_params)
                    try:
                        result_data = json.loads(result)
                        if result_data.get("success") and result_data.get("results"):
                            return {
                                "success": True,
                                "type": "query",
                                "response": f"📋 {target_collection.title()} - Sorgu Sonucu ({len(result_data['results'])} kayıt):",
                                "data": result_data["results"]
                            }
                        else:
                            return {
                                "success": True,
                                "type": "no_data",
                                "response": f"❌ {target_collection} koleksiyonunda sorgu kriterlerine uygun veri bulunamadı"
                            }
                    except Exception as e:
                        logger.error(f"Error processing LLM query result: {str(e)}")
        
        # Handle other operations
        
        # Document count queries (collection-level only, database-level handled in priority section)
        if any(word in user_input_lower for word in ["kaç", "count", "sayı", "how many"]) and not any(word in user_input_lower for word in ["collection", "koleksiyon", "tablo"]):
            try:
                # Document count queries (collection-level)
                target_collection = self._detect_collection_from_query(user_input)
                if target_collection:
                    result = self.connection_tool.db[target_collection].count_documents({})
                    return {
                        "success": True,
                        "type": "count",
                        "response": f"📊 {target_collection} koleksiyonunda toplam {result} kayıt var"
                    }
                else:
                    # Count all documents across all collections
                    collections = self.connection_tool.db.list_collection_names()
                    total_docs = 0
                    collection_counts = []
                    for coll in collections:
                        count = self.connection_tool.db[coll].count_documents({})
                        total_docs += count
                        collection_counts.append({"name": coll, "count": count})
                    
                    return {
                        "success": True,
                        "type": "count",
                        "response": f"📊 Veritabanında toplam {total_docs} kayıt var",
                        "data": collection_counts
                    }
            except Exception as e:
                logger.error(f"Count error: {str(e)}")
                pass
        
        # Collections listing
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

# HTML template moved to templates/index.html

@app.route("/")
def index():
    return render_template('index.html', 
                         model_name="google/gemma-3n-e4b",
                         database_name="agent")

@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.get_json()
        query = data.get("query", "")
        selected_collection = data.get("selected_collection", None)
        
        if not query:
            return jsonify({"error": "Query is required"})
        
        # Set collection context in agent
        agent.selected_collection = selected_collection
        
        # Process query with collection context
        result = agent.process_query(query)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/collections")
def get_collections():
    try:
        # Ensure database connection
        if agent.connection_tool.db is None:
            connect_result = agent.connection_tool._run("agent")
            if "Error" in connect_result:
                logger.error(f"Database connection failed: {connect_result}")
                return jsonify({
                    "success": False,
                    "collections": [],
                    "error": "Database connection failed"
                })
        
        collections_list = agent.connection_tool.db.list_collection_names()
        collections_data = []
        
        for coll_name in collections_list:
            try:
                count = agent.connection_tool.db[coll_name].count_documents({})
                collections_data.append({
                    "name": coll_name,
                    "count": count
                })
            except Exception as e:
                logger.warning(f"Error counting documents in {coll_name}: {str(e)}")
                collections_data.append({
                    "name": coll_name,
                    "count": 0
                })
        
        logger.info(f"Found {len(collections_data)} collections")
        return jsonify({
            "success": True,
            "collections": collections_data
        })
        
    except Exception as e:
        logger.error(f"Collections endpoint error: {str(e)}")
        return jsonify({
            "success": False,
            "collections": [],
            "error": str(e)
        })

def main():
    global agent
    
    print("🚀 MongoDB LangChain Agent with LM Studio - Enhanced Version")
    print("=" * 60)
    
    # Initialize agent
    agent = MongoDBLangChainAgent(
        mongo_uri="mongodb://localhost:27017/",
        lm_studio_url="http://localhost:1234/v1",
        model_name="google/gemma-3n-e4b"
    )
    
    # Initialize database connection
    try:
        connect_result = agent.connection_tool._run("agent")
        if "Error" not in connect_result:
            print("✅ Agent initialized successfully")
            print(f"📊 Database connected: {agent.connection_tool.db.name}")
        else:
            print(f"⚠️ Agent initialized but database connection failed: {connect_result}")
    except Exception as e:
        print(f"⚠️ Agent initialized but database connection failed: {str(e)}")
    print("🌐 Starting Flask web server...")
    print("📱 Web interface: http://localhost:5000")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()
