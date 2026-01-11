# 🎙️ Turkish Voice Assistant

**Whisper Small TR** + **LM Studio (Qwen3-VL-4B)** + **Tools Support**

An intelligent Turkish assistant that understands your voice or text questions, can search the web, perform calculations, and more.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 **Speech Recognition** | Turkish-optimized Whisper model |
| 🧠 **Local LLM** | Qwen3-VL-4B via LM Studio |
| 🔍 **Web Search** | Current information access via DuckDuckGo |
| 🔢 **Calculations** | Mathematical operations |
| 🕐 **Date/Time** | Real-time time information |
| 💬 **Chat** | Context-aware conversations |

## 📋 Requirements

### 1. LM Studio Installation

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download the `qwen3-vl-4b` model
3. Load the model and start the local server
4. Server runs by default at `http://localhost:1234`

### 2. Python Requirements

```bash
# Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## 🚀 Running

```bash
# For Jupyter Notebook
jupyter notebook whisper_lmstudio_colab.ipynb

# Or open in Google Colab
```

Open `http://localhost:7860` in your browser after launching.

## ⚙️ Configuration

Before running, configure the LM Studio connection:

1. Start LM Studio and load your model
2. Start the local server (usually port 1234)
3. If using ngrok, set up the tunnel:
   ```bash
   ngrok http 1234
   ```
4. Update the `ngrok_url` in the notebook configuration cell

**Note:** For security, use environment variables for sensitive URLs. See `.env.example` for reference.

## 🎯 Usage

### Voice Question

1. Click the microphone button or upload an audio file
2. Click "Process Audio" button
3. View transcription and response

### Text Question

1. Type your question in the text box
2. Click "Send" button or press Enter

### Example Questions

| Question Type | Example |
|--------------|---------|
| **Information** | "What is Python programming language?" |
| **Current Events** | "What are today's news?" |
| **Calculation** | "What is 125 times 48?" |
| **Date/Time** | "What day is today?" |
| **Weather** | "What's the weather in Istanbul?" |

## 🛠️ Tools

The assistant can automatically use the following tools:

| Tool | Description | Example Usage |
|------|-------------|---------------|
| `web_search` | Searches the web | "Latest Tesla news" |
| `calculate` | Performs math calculations | "567 * 234 = ?" |
| `get_current_time` | Returns date/time | "What time is it?" |
| `get_weather` | Returns weather information | "Weather in Istanbul" |
| `ask_clarification` | Asks for clarification | Used for unclear questions |

## 📁 File Structure

```
Audio/
├── whisper_lmstudio_colab.ipynb  # Main application (Jupyter Notebook)
├── create_test_audio.py          # Test audio file generator
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
└── test_audios/                  # Test audio files (gitignored)
```

## ⚠️ Troubleshooting

### LM Studio Connection Error

- Make sure LM Studio is running
- Ensure the local server is started
- Check the port number (default: 1234)
- If using ngrok, verify the URL is correct

### Whisper Running Slowly

- CUDA-enabled GPU will be used automatically if available
- First load may take some time on CPU

### Speech Recognition Errors

- 16kHz sampling rate recommended
- Record clear and noise-free audio

### ngrok URL Issues

- ngrok URLs are temporary - update the configuration if expired
- Consider using environment variables for better security
- For production, use a static ngrok domain or direct connection

## 🔒 Security Notes

- **API Keys**: The project uses `api_key="not-needed"` for LM Studio local connections, which is safe
- **ngrok URLs**: These are temporary. Update the configuration when they expire
- **Environment Variables**: For production, store sensitive URLs in environment variables (see `.env.example`)
- **Localhost**: Default connections use localhost, which is safe for local development

## 📚 Technologies Used

- **[Whisper Small TR](https://huggingface.co/emredeveloper/whisper-small-tr)** - Turkish ASR
- **[LM Studio](https://lmstudio.ai/)** - Local LLM runtime
- **[Qwen3-VL-4B](https://huggingface.co/Qwen)** - Multimodal LLM
- **[DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search)** - Web search
- **[Gradio](https://gradio.app/)** - Web interface

## 📄 License

MIT License

---

**Developer:** Emre Developer  
**Model:** [emredeveloper/whisper-small-tr](https://huggingface.co/emredeveloper/whisper-small-tr)
