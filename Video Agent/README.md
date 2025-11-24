# 🎥 YouTube Video Understanding System - Nemotron Nano 12B 2 VL

A system that analyzes and understands YouTube videos using NVIDIA Nemotron Nano 12B 2 VL model.

## 🌟 Features

- ✅ Directly analyzes YouTube video URLs
- ✅ Understands and summarizes video content in detail
- ✅ Ask custom questions
- ✅ Different analysis types (comprehensive, summary, topics, key points, visual description)
- ✅ Compare multiple videos
- ✅ Free model usage via OpenRouter API

## 📋 Requirements

### Basic Libraries
```bash
pip install httpx python-dotenv
```

### Video Processing (Recommended)
For video visual analysis:
```bash
pip install yt-dlp opencv-python
```

### For Fast Download (Optional)
ffmpeg is recommended to download only first N minutes (faster):
- **Windows**: [Download ffmpeg](https://www.gyan.dev/ffmpeg/builds/) or `choco install ffmpeg`
- **Linux**: `sudo apt install ffmpeg` or `sudo yum install ffmpeg`
- **Mac**: `brew install ffmpeg`

**Note**: If ffmpeg is not available, the system will download the full video and then trim it (slightly slower but works).

### Transcript Support (Optional)
To use video transcripts:
```bash
pip install youtube-transcript-api
```

**Note**: If you only want to use transcripts, video download libraries are not required.

## 🔑 API Key Setup

1. Go to [OpenRouter](https://openrouter.ai/) and create an account
2. Get your API key
3. Set as environment variable:

```bash
# Windows
set OPENROUTER_API_KEY=sk-or-v1-...

# Linux/Mac
export OPENROUTER_API_KEY=sk-or-v1-...
```

Or create a `.env` file:

```env
OPENROUTER_API_KEY=sk-or-v1-...
```

## 🚀 Usage

### Basic Usage

```python
# File name: nano-video-understand.py
# To import as module:
import sys
sys.path.append('Video Agent')
from nano_video_understand import VideoUnderstandingAgent

# Or run directly:
# python nano-video-understand.py

# Create agent
agent = VideoUnderstandingAgent()

# Analyze video
result = agent.understand_video(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    question="What is discussed in this video?"
)

if result["success"]:
    print(result["answer"])
else:
    print(f"Error: {result['error']}")
```

### Detailed Analysis

```python
# Comprehensive analysis
result = agent.analyze_video_detailed(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    analysis_type="comprehensive"
)

# Summary
result = agent.analyze_video_detailed(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    analysis_type="summary"
)

# Topics
result = agent.analyze_video_detailed(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    analysis_type="topics"
)

# Key points
result = agent.analyze_video_detailed(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    analysis_type="key_points"
)

# Visual description
result = agent.analyze_video_detailed(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    analysis_type="visual_description"
)
```

### Video Comparison

```python
# Compare two or more videos
result = agent.compare_videos(
    video_urls=[
        "https://www.youtube.com/watch?v=VIDEO_ID_1",
        "https://www.youtube.com/watch?v=VIDEO_ID_2"
    ],
    comparison_aspect="content quality"
)
```

### CLI Usage

```bash
python nano-video-understand.py
```

The program offers you these options:
1. Single video analysis
2. Detailed analysis (comprehensive)
3. Video comparison
4. Ask custom question

## 📊 Analysis Types

- **comprehensive**: Comprehensive analysis (main topic, key points, visual content, purpose, details)
- **summary**: Brief and concise summary
- **topics**: List of topics covered in the video
- **key_points**: List of key points as items
- **visual_description**: Detailed description of visual content

## ⚙️ Parameters

### `understand_video()` Parameters

- `video_url` (str): YouTube video URL (required)
- `question` (str, optional): Question to ask about the video
- `fps` (float, default: 2.0): Frames per second to extract from video (model recommendation: 2.0)
- `max_tokens` (int, default: 4096): Maximum output token count (model limit: 128K total)
- `min_frames` (int, default: 8): Minimum frame count (model requirement)
- `max_frames` (int, default: 10): Maximum frame count (**OpenRouter API limit: 10 images**)
- `use_transcript` (bool, default: True): Use transcript (True) or only images (False)
- `analyze_first_minutes` (float, optional): Analyze only first N minutes and predict about remaining parts (None = full video)

**Important Notes**:
- **OpenRouter API limit**: Maximum 10 images can be sent at once
- Frames are selected at equal intervals throughout the video (representative frames from beginning, middle, end)
- Optimized according to model specifications. FPS is set to 2.0 (model's recommended value)

### `analyze_video_detailed()` Parameters

- `video_url` (str): YouTube video URL (required)
- `analysis_type` (str): Analysis type (comprehensive, summary, topics, key_points, visual_description)

## 📝 Example Output

```
✅ Video Analysis Successful!
------------------------------------------------------------
📹 Video URL: https://www.youtube.com/watch?v=VIDEO_ID
📹 Video ID: VIDEO_ID
❓ Question: What is discussed in this video?
------------------------------------------------------------
📊 Token Usage:
   - Prompt: 1234
   - Completion: 567
   - Total: 1801
------------------------------------------------------------

💬 Analysis Result:
============================================================
This video covers the uses of artificial intelligence 
technologies in daily life. The following topics are covered:
...
============================================================
```

## 🔧 Troubleshooting

### API Key Error
```
ValueError: OpenRouter API key required!
```
**Solution**: Set `OPENROUTER_API_KEY` environment variable or create `.env` file.

### Timeout
```
httpx.TimeoutException: Request timed out
```
**Solution**: 
- If video is too long, reduce `fps` parameter (e.g., 0.5)
- Reduce `max_tokens` parameter
- Check your internet connection

### Invalid URL
```
❌ Error: Invalid YouTube URL
```
**Solution**: Make sure the YouTube video URL is in the correct format.

## 📚 About the Model

**NVIDIA Nemotron Nano 12B 2 VL** ([Model Card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)):
- **12.6 billion parameter** multimodal model
- **Text, image and video understanding** capability
- **Video support**: MP4, MKV, FLV, 3GP formats
- **Frame parameters**: 
  - Recommended FPS: 2.0
  - Minimum frames: 8
  - Maximum frames: 128 (model limit)
  - **OpenRouter API limit: 10 images** (maximum that can be sent at once)
- **Multi-image support**: Up to 4 images
- **Token limit**: 128K (input + output)
- **Resolution**: 12-tile layout (each tile 512×512 pixels)
- **Benchmark results**:
  - OCRBench v2: 62.0
  - OCRBench: 85.6
  - DocVQA: 94.39
  - ChartQA: 89.72
  - Video-MME: 65.9
  - Vision Average: 74.0
- **Free usage via OpenRouter**

## 🤝 Contributing

We welcome your contributions! Please send a pull request.

## 📄 License

This project is licensed under the MIT license.

## 🔗 Resources

- [OpenRouter](https://openrouter.ai/)
- [Nemotron Nano 12B 2 VL Model Page (OpenRouter)](https://openrouter.ai/nvidia/nemotron-nano-12b-v2-vl:free)
- [Nemotron Nano 12B 2 VL Model Card (Hugging Face)](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)
- [NVIDIA Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/vlm/nemotron-nano-v2-vl.html)
- [Model ArXiv Paper](https://arxiv.org/abs/2511.03929)
