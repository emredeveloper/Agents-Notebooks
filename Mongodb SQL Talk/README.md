# MongoDB Natural Language Agent

This project is an intelligent agent that lets you interact with MongoDB databases in natural language. With LM Studio integration, it can work with any MongoDB collection and fields.

## 🚀 Features

- **Natural Language Understanding**: Queries like "find users whose name is Ahmet"
- **Dynamic Collection Detection**: Works with any collection name
- **Smart Data Insertion**: Insert data with commands like "add a new user"
- **Automatic Schema Analysis**: Automatically detects existing fields
- **Web Interface**: User-friendly modern web UI
- **LM Studio Integration**: Local LLM support

## 📋 Requirements

- Python 3.8+
- MongoDB (local or remote)
- LM Studio (for local LLM)
- Required Python packages (in requirements.txt)

## 🛠 Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd mongodb-langchain-agent
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start LM Studio:**
   - Download and run LM Studio
   - Load a model (e.g., Qwen, Gemma)
   - Start the server on port 1234

5. **Start MongoDB:**
   - Ensure MongoDB is running
   - Default: `mongodb://localhost:27017/`

## 🎯 Usage

### With the Web UI

```bash
python mongodb-langchain-agent-clean.py
```

Open `http://localhost:5000` in your browser.

### Example Queries

#### Query Data:
- "list collections"
- "show the first 5 records in the users table"
- "find users whose name is Ahmet"
- "list those older than 25"
- "products with price between 100 and 500"

#### Insert Data:
- "add a user: name Mehmet, surname Kaya, age 30"
- "add a new product: laptop, price 15000"
- "add an order: customer Ayşe, amount 250"

#### Statistics:
- "how many users are there?"
- "how many records in the products collection?"
- "what is the total record count?"

## ⚙️ Configuration

### Database Connection

```python
# in mongodb-langchain-agent-clean.py
agent = MongoDBLangChainAgent(
    mongo_uri="mongodb://localhost:27017/",  # MongoDB URI
    lm_studio_url="http://localhost:1234/v1", # LM Studio URL
    model_name="your-model-name"              # LLM Model name
)
```

### LM Studio Settings

```python
# in LMStudioLLM class
base_url: str = "http://localhost:1234/v1"  # LM Studio URL
model_name: str = "qwen/qwen3-4b-2507"      # Model name
```

## 📁 Project Structure

```
mongodb-langchain-agent/
├── mongodb-langchain-agent-clean.py  # Main application
├── templates/
│   └── index.html                    # Web interface
├── static/
│   ├── css/
│   │   └── style.css                # Stylesheet
│   └── js/
│       └── app.js                   # JavaScript
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## 🔧 Customization

### Add New Collection Types

```python
# in _detect_collection_from_query method
collection_patterns = {
    'users': ['user', 'kullanıcı', 'kişi', 'person'],
    'products': ['product', 'ürün', 'item'],
    'orders': ['order', 'sipariş', 'purchase'],
    'customers': ['customer', 'müşteri', 'client'],
    # Add new types here
}
```

### Sample Data Templates

```python
# in _add_sample_data method you can add new data types
elif 'custom_collection' in collection_name.lower():
    sample_data = [
        {"field1": "value1", "field2": 123, "created_at": datetime.now()},
        # your custom data
    ]
```

## 🐛 Troubleshooting

### LM Studio Connection Error
- Ensure LM Studio is running
- Check port 1234 is open
- Verify the model is loaded

### MongoDB Connection Error
- Ensure MongoDB service is running
- Check the connection string
- Check firewall settings

### Parsing Errors
- Try a different LLM model
- Lower the temperature value
- Simplify the prompts

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) - Agent framework
- [LM Studio](https://lmstudio.ai/) - Local LLM support
- [MongoDB](https://www.mongodb.com/) - Database
- [Flask](https://flask.palletsprojects.com/) - Web framework
