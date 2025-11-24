"""
Nemotron Nano 12B 2 VL Video Understanding System - Usage Examples
"""

import os
import sys

# Add path to import module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Use importlib to import module names with hyphens in Python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "nano_video_understand", 
    os.path.join(current_dir, "nano-video-understand.py")
)
nano_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nano_module)
VideoUnderstandingAgent = nano_module.VideoUnderstandingAgent


def example_basic_analysis():
    """Basic video analysis example"""
    print("=" * 60)
    print("Example 1: Basic Video Analysis")
    print("=" * 60)
    
    # Set API key (or get from environment variable)
    api_key = "your key"
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY environment variable is not set!")
        return
    
    # Create agent
    agent = VideoUnderstandingAgent(api_key=api_key)
    
    # Example video URL (you can use your own video URL)
    video_url = "video url"  # Example
    
    # Analyze video - Analyze only first 1 minute and predict about remaining parts
    result = agent.understand_video(
        video_url=video_url,
        question=None,  # Default question will be used (first minute analysis + prediction)
        fps=2.0,  # Model recommendation: 2.0 FPS
        max_frames=10,  # OpenRouter API limit: 10 images
        use_transcript=True,  # Also use transcript
        analyze_first_minutes=1.0  # Analyze only first 1 minute
    )
    
    # Display results
    if result["success"]:
        print("\n✅ Analysis Successful!")
        print(f"\n📹 Video: {result['video_url']}")
        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"\n📊 Token Usage: {result['usage'].get('total_tokens', 'N/A')}")
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
    
    agent.close()


def example_detailed_analysis():
    """Detailed analysis example"""
    print("\n" + "=" * 60)
    print("Example 2: Detailed Video Analysis")
    print("=" * 60)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY environment variable is not set!")
        return
    
    agent = VideoUnderstandingAgent(api_key=api_key)
    
    video_url = "video url"  # Example
    
    # Comprehensive analysis
    result = agent.analyze_video_detailed(
        video_url=video_url,
        analysis_type="comprehensive"
    )
    
    if result["success"]:
        print("\n✅ Comprehensive Analysis Completed!")
        print(f"\n💬 Analysis:\n{result['answer']}")
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
    
    agent.close()


def example_custom_question():
    """Custom question example"""
    print("\n" + "=" * 60)
    print("Example 3: Ask Custom Question")
    print("=" * 60)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY environment variable is not set!")
        return
    
    agent = VideoUnderstandingAgent(api_key=api_key)
    
    video_url = "https://www.youtube.com/watch?v=XDzhP93M5o4"  # Example
    
    # Custom question
    custom_question = """
    What technical topics are covered in this video?
    List the steps shown in the video.
    """
    
    result = agent.understand_video(
        video_url=video_url,
        question=custom_question
    )
    
    if result["success"]:
        print("\n✅ Question Answered!")
        print(f"\n❓ Question: {result['question']}")
        print(f"\n💬 Answer:\n{result['answer']}")
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
    
    agent.close()


if __name__ == "__main__":
    print("🎥 Nemotron Nano 12B 2 VL Video Understanding System - Examples")
    print("\n⚠️  Note: OPENROUTER_API_KEY is required to run these examples!")
    print("⚠️  Replace video URLs with your own video links.\n")
    
    # Run examples
    try:
        example_basic_analysis()
        # example_detailed_analysis()  # Optional
        # example_custom_question()  # Optional
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
