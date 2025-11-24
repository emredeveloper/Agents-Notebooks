"""
YouTube Video Understanding System - NVIDIA Nemotron Nano 12B 2 VL
Uses Nemotron Nano model via OpenRouter API to understand YouTube videos.
"""

import os
import json
import base64
import tempfile
import shutil
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import httpx

# Required libraries for video processing
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("⚠️  yt-dlp is not installed. To download videos: pip install yt-dlp")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  opencv-python is not installed. To extract frames: pip install opencv-python")

# Load .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Continue if python-dotenv is not installed
    pass
except Exception:
    pass


class VideoUnderstandingAgent:
    """Agent that understands YouTube videos using Nemotron Nano 12B 2 VL"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: OpenRouter API key (can also be retrieved from OPENROUTER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required! "
                "Set OPENROUTER_API_KEY environment variable or provide api_key parameter."
            )
        
        self.model_id = "nvidia/nemotron-nano-12b-v2-vl:free"
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/emredeveloper/Agents-Notebooks",
                "X-Title": "YouTube Video Understanding Agent",
                "Content-Type": "application/json"
            },
            timeout=120.0  # Video processing can take a long time
        )
        
        print(f"✅ Nemotron Nano 12B 2 VL model configured")
        print(f"🔗 Model: {self.model_id}")
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extracts video ID from YouTube URL"""
        try:
            parsed_url = urlparse(url)
            
            if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
                if parsed_url.path == '/watch':
                    return parse_qs(parsed_url.query).get('v', [None])[0]
                elif parsed_url.path.startswith('/embed/'):
                    return parsed_url.path.split('/embed/')[1]
            elif parsed_url.hostname in ['youtu.be']:
                return parsed_url.path[1:].split('?')[0]
            
            return None
        except Exception as e:
            print(f"❌ URL parse error: {e}")
            return None
    
    def download_video(
        self, 
        video_url: str, 
        output_path: Optional[str] = None,
        max_duration: Optional[float] = None
    ) -> Optional[str]:
        """
        Downloads YouTube video
        
        Args:
            video_url: YouTube video URL
            output_path: Output file path (None = temporary file)
            max_duration: Maximum duration (seconds) - only download up to this duration (None = full video)
        """
        if not YT_DLP_AVAILABLE:
            return None
        
        try:
            if output_path is None:
                # Create temporary file
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, f"youtube_video_{os.getpid()}.mp4")
            
            if max_duration:
                print(f"📥 Downloading video (first {max_duration:.1f} seconds)...")
            else:
                print(f"📥 Downloading video...")
            
            # Model supports MP4, MKV, FLV, 3GP formats
            # Prefer MP4 format (most common and compatible)
            # Select low quality/resolution for faster download
            if max_duration:
                # If only a short segment will be downloaded, lowest quality is sufficient
                format_selector = 'worst[height<=480][ext=mp4]/worst[height<=720][ext=mp4]/worst[ext=mp4]/worst'
            else:
                # If full video will be downloaded, medium quality
                format_selector = 'best[height<=480][ext=mp4]/best[height<=720][ext=mp4]/best[ext=mp4]/best'
            
            ydl_opts = {
                'format': format_selector,
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
                'merge_output_format': 'mp4',  # Merge as MP4
            }
            
            # If only a specific duration will be downloaded, cut with ffmpeg
            if max_duration and max_duration > 0:
                # Check if ffmpeg is installed
                ffmpeg_path = shutil.which('ffmpeg')
                
                if ffmpeg_path:
                    # Use ffmpeg to get only first N seconds
                    ydl_opts['postprocessors'] = [{
                        'key': 'FFmpegVideoRemuxer',
                        'preferedformat': 'mp4',
                    }]
                    ydl_opts['postprocessor_args'] = {
                        'ffmpeg': ['-t', str(max_duration)]  # Get first N seconds
                    }
                    print(f"⚡ Downloading only first {max_duration:.1f} seconds with ffmpeg...")
                else:
                    print(f"⚠️  ffmpeg not found, full video will be downloaded (will be trimmed later)")
                    print(f"💡 To install ffmpeg: https://ffmpeg.org/download.html")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if os.path.exists(output_path):
                # If ffmpeg is not available and max_duration exists, trim with OpenCV
                if max_duration and max_duration > 0 and CV2_AVAILABLE:
                    # Check ffmpeg again
                    if not shutil.which('ffmpeg'):
                        print(f"✂️  Trimming video (first {max_duration:.1f} seconds)...")
                        trimmed_path = self._trim_video_opencv(output_path, max_duration)
                        if trimmed_path:
                            # Delete old file and use new file
                            try:
                                os.remove(output_path)
                            except:
                                pass
                            output_path = trimmed_path
                
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"✅ Video downloaded: {output_path} ({file_size:.1f} MB)")
                return output_path
            else:
                print("❌ Video could not be downloaded")
                return None
                
        except Exception as e:
            print(f"❌ Video download error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _trim_video_opencv(self, video_path: str, max_duration: float) -> Optional[str]:
        """Trims video with OpenCV (alternative if ffmpeg is not available)"""
        if not CV2_AVAILABLE:
            return None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # New file path
            trimmed_path = video_path.replace('.mp4', '_trimmed.mp4')
            out = cv2.VideoWriter(trimmed_path, fourcc, fps, (width, height))
            
            max_frame = int(max_duration * fps)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frame:
                    break
                
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            if os.path.exists(trimmed_path):
                return trimmed_path
            return None
            
        except Exception as e:
            print(f"⚠️  Video trimming error: {e}")
            return None
    
    def extract_frames(
        self, 
        video_path: str, 
        fps: float = 2.0, 
        min_frames: int = 8, 
        max_frames: int = 10,
        max_duration: Optional[float] = None
    ) -> List[str]:
        """
        Extracts frames from video and converts them to base64 format
        
        OpenRouter API limit: Maximum 10 images can be sent at once
        Frames are selected at equal intervals throughout the video (from beginning, middle, end)
        
        Args:
            video_path: Video file path
            fps: Frames per second to extract (default: 2.0 - model recommendation)
            min_frames: Minimum number of frames (default: 8)
            max_frames: Maximum number of frames (default: 10 - OpenRouter API limit)
            max_duration: Maximum duration (seconds) - only extract frames up to this duration (None = full video)
        """
        if not CV2_AVAILABLE:
            return []
        
        try:
            print(f"🎬 Extracting frames (fps: {fps}, min: {min_frames}, max: {max_frames})...")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("❌ Video could not be opened")
                return []
            
            # Video information
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            full_duration = total_frames / video_fps if video_fps > 0 else 0
            
            # Maximum duration check
            if max_duration is not None and max_duration > 0:
                max_frame = int(max_duration * video_fps) if video_fps > 0 else total_frames
                max_frame = min(max_frame, total_frames)
                duration = max_duration
                print(f"📊 Video info: {full_duration:.1f} seconds (total), {total_frames} frames, {video_fps:.2f} fps")
                print(f"⏱️  Only first {max_duration:.1f} seconds will be analyzed ({max_frame} frames)")
            else:
                max_frame = total_frames
                duration = full_duration
                print(f"📊 Video info: {duration:.1f} seconds, {total_frames} frames, {video_fps:.2f} fps")
            
            # OpenRouter API limit: Maximum 10 images
            # Select frames at equal intervals within specified duration
            target_frames = min(max_frames, max(min_frames, 10))  # API limit: 10
            print(f"🎯 Target frame count: {target_frames} (OpenRouter API limit: 10)")
            
            # Calculate frame positions at equal intervals within specified duration
            frame_positions = []
            if max_frame > 0:
                # Select frames at equal intervals within first N seconds
                step = max(1, max_frame // target_frames)
                for i in range(target_frames):
                    frame_pos = min(i * step, max_frame - 1)
                    frame_positions.append(frame_pos)
            else:
                # If no video info, take first N frames
                frame_positions = list(range(target_frames))
            
            frames_base64 = []
            frame_count = 0
            
            # Read frames at specific positions
            for frame_pos in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if not ret:
                    # If frame cannot be read at specific position, try next one
                    continue
                
                # Resize frame (according to model's resolution limits)
                height, width = frame.shape[:2]
                
                # Shrink very large frames (for performance)
                max_dimension = 1024  # Smaller than model's maximum supported size
                if max(height, width) > max_dimension:
                    scale = max_dimension / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Convert frame to JPEG format (optimize quality)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames_base64.append(frame_base64)
                frame_count += 1
                
                # Show progress
                time_sec = frame_pos / video_fps if video_fps > 0 else 0
                print(f"  📸 Frame {frame_count}/{target_frames} extracted (second: {time_sec:.1f})")
            
            cap.release()
            
            # Minimum frame check
            if len(frames_base64) < min_frames:
                print(f"⚠️  Only {len(frames_base64)} frames extracted (minimum {min_frames} recommended)")
            else:
                print(f"✅ {len(frames_base64)} frames extracted")
            
            return frames_base64
            
        except Exception as e:
            print(f"❌ Frame extraction error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def understand_video(
        self, 
        video_url: str, 
        question: Optional[str] = None,
            fps: float = 2.0,
            max_tokens: int = 4096,
            min_frames: int = 8,
            max_frames: int = 10,
            use_transcript: bool = True,
            analyze_first_minutes: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Understands and analyzes YouTube video
        
        Optimized according to model specifications:
        - Recommended FPS: 2.0 (model recommendation)
        - Min frames: 8, Max frames: 128 (model limits)
        - Token limit: 128K (input + output), here max_tokens is for output
        
        Args:
            video_url: YouTube video URL
            question: Question to ask about the video (optional)
            fps: Frames per second to extract from video (default: 2.0 - model recommendation)
            max_tokens: Maximum output token count (default: 4096)
            min_frames: Minimum frame count (default: 8 - model requirement)
            max_frames: Maximum frame count (default: 10 - OpenRouter API limit)
            use_transcript: Use transcript (True) or only images (False)
            analyze_first_minutes: Analyze only first N minutes and predict about remaining parts (None = full video)
            
        Returns:
            Dict containing model response and metadata
        """
        try:
            # Check video ID
            video_id = self.extract_video_id(video_url)
            if not video_id:
                return {
                    "success": False,
                    "error": "Invalid YouTube URL",
                    "video_url": video_url
                }
            
            print(f"🔵 Analyzing video: {video_url}")
            print(f"📹 Video ID: {video_id}")
            
            # Default question if none provided
            if not question:
                if analyze_first_minutes:
                    question = f"""I have analyzed only the first {analyze_first_minutes} minutes of this video. 

Please do the following:
1. Summarize in detail what was discussed in the first {analyze_first_minutes} minutes
2. What is the video's general topic and theme?
3. Based on the content in the first minutes, predict what will likely be discussed in the remaining parts of the video
4. Make inferences about the video's general structure and flow
5. Evaluate what viewers can learn from this video

Base your analysis on the images and transcript from the first {analyze_first_minutes} minutes, make reasonable predictions for the remaining parts."""
                else:
                    question = "Analyze this video in detail and summarize. What is being discussed in the video, what topics are covered, what is shown visually?"
            
            print(f"❓ Question: {question}")
            
            # Get transcript if used
            transcript_text = None
            if use_transcript:
                try:
                    from youtube_transcript_api import YouTubeTranscriptApi
                    print("📝 Getting transcript...")
                    ytt_api = YouTubeTranscriptApi()
                    
                    # Try different languages
                    for lang in ['tr', 'en']:
                        try:
                            fetched_transcript = ytt_api.fetch(video_id, languages=[lang])
                            raw_data = fetched_transcript.to_raw_data()
                            
                            # If analyzing only first N minutes, limit transcript too
                            if analyze_first_minutes:
                                # Check timestamps in transcript
                                filtered_data = []
                                for item in raw_data:
                                    start_time = item.get('start', 0)
                                    if start_time <= analyze_first_minutes * 60:  # Convert minutes to seconds
                                        filtered_data.append(item)
                                    else:
                                        break  # Time limit exceeded, stop
                                
                                transcript_text = " ".join([item['text'] for item in filtered_data])
                                print(f"✅ Transcript found ({lang}): {len(transcript_text)} characters (first {analyze_first_minutes} minutes)")
                            else:
                                transcript_text = " ".join([item['text'] for item in raw_data])
                                print(f"✅ Transcript found ({lang}): {len(transcript_text)} characters")
                            break
                        except:
                            continue
                    
                    if not transcript_text:
                        # Get one of available transcripts
                        try:
                            transcript_list = ytt_api.list(video_id)
                            for transcript in transcript_list:
                                try:
                                    fetched_transcript = transcript.fetch()
                                    raw_data = fetched_transcript.to_raw_data()
                                    
                                    # If analyzing only first N minutes, limit transcript too
                                    if analyze_first_minutes:
                                        filtered_data = []
                                        for item in raw_data:
                                            start_time = item.get('start', 0)
                                            if start_time <= analyze_first_minutes * 60:
                                                filtered_data.append(item)
                                            else:
                                                break
                                        transcript_text = " ".join([item['text'] for item in filtered_data])
                                        print(f"✅ Transcript found: {transcript.language} (first {analyze_first_minutes} minutes)")
                                    else:
                                        transcript_text = " ".join([item['text'] for item in raw_data])
                                        print(f"✅ Transcript found: {transcript.language}")
                                    break
                                except:
                                    continue
                        except:
                            pass
                    
                    if not transcript_text:
                        print("⚠️  Transcript not found, only images will be used")
                except ImportError:
                    print("⚠️  youtube-transcript-api is not installed, transcript cannot be used")
                except Exception as e:
                    print(f"⚠️  Could not get transcript: {e}")
            
            # Download video and extract frames
            video_path = None
            frames_base64 = []
            
            if YT_DLP_AVAILABLE and CV2_AVAILABLE:
                # Calculate maximum duration parameter (minutes -> seconds)
                max_duration_seconds = None
                if analyze_first_minutes:
                    max_duration_seconds = analyze_first_minutes * 60
                
                video_path = self.download_video(video_url, max_duration=max_duration_seconds)
                if video_path:
                    # Calculate maximum duration parameter (minutes -> seconds)
                    max_duration_seconds = None
                    if analyze_first_minutes:
                        max_duration_seconds = analyze_first_minutes * 60
                    
                    frames_base64 = self.extract_frames(
                        video_path, 
                        fps=fps, 
                        min_frames=min_frames, 
                        max_frames=max_frames,
                        max_duration=max_duration_seconds
                    )
                    # Delete temporary video file
                    try:
                        if os.path.exists(video_path):
                            os.remove(video_path)
                            print(f"🗑️  Temporary video file deleted")
                    except:
                        pass
            else:
                print("⚠️  Video download/frame extraction libraries are not installed")
                if not YT_DLP_AVAILABLE:
                    print("   To install yt-dlp: pip install yt-dlp")
                if not CV2_AVAILABLE:
                    print("   To install opencv-python: pip install opencv-python")
            
            # Prepare message content
            content_parts = []
            
            # Text part
            text_content = question
            if transcript_text:
                # Add transcript (considering token limit)
                # Model has 128K token limit (input + output)
                # We optimize the transcript
                transcript_preview = transcript_text[:3000]  # First 3000 characters
                if analyze_first_minutes:
                    text_content += f"\n\nVideo Transcript (first {analyze_first_minutes} minutes):\n{transcript_preview}"
                else:
                    text_content += f"\n\nVideo Transcript (first part):\n{transcript_preview}"
                if len(transcript_text) > 3000:
                    text_content += f"\n\n[Transcript continues, total {len(transcript_text)} characters]"
            elif not frames_base64:
                return {
                    "success": False,
                    "error": "Neither transcript nor images found. Video cannot be analyzed.",
                    "video_url": video_url
                }
            
            content_parts.append({
                "type": "text",
                "text": text_content
            })
            
            # Image parts (base64)
            # OpenRouter API limit: Maximum 10 images can be sent at once
            if len(frames_base64) > 10:
                print(f"⚠️  {len(frames_base64)} frames found, using first 10 due to OpenRouter API limit (10)")
                frames_base64 = frames_base64[:10]
            
            print(f"🖼️  Sending {len(frames_base64)} frames...")
            
            # Send frames (OpenRouter API limit: 10 images)
            for i, frame_base64 in enumerate(frames_base64):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_base64}"
                    }
                })
            
            print(f"✅ {len(frames_base64)} frames prepared")
            
            # Prepare OpenRouter API request
            messages = [
                {
                    "role": "user",
                    "content": content_parts
                }
            ]
            
            # API request
            payload = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            frame_info = f"{len(frames_base64)} images"
            if analyze_first_minutes:
                frame_info += f" (first {analyze_first_minutes} minutes)"
            transcript_info = f"{len(transcript_text) if transcript_text else 0} characters transcript"
            if analyze_first_minutes and transcript_text:
                transcript_info += f" (first {analyze_first_minutes} minutes)"
            
            print(f"⏳ Sending request to model... ({frame_info}, {transcript_info})")
            
            response = self.client.post(
                "/chat/completions",
                json=payload
            )
            
            if response.status_code != 200:
                error_text = response.text
                print(f"❌ API error: {response.status_code}")
                print(f"❌ Error details: {error_text}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "error_detail": error_text,
                    "video_url": video_url
                }
            
            result = response.json()
            
            # Parse response
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]
                content = message.get("content", "")
                
                # Metadata
                usage = result.get("usage", {})
                
                print("✅ Video analysis completed!")
                print(f"📊 Tokens used: {usage.get('total_tokens', 'N/A')}")
                
                return {
                    "success": True,
                    "video_url": video_url,
                    "video_id": video_id,
                    "question": question,
                    "answer": content,
                    "usage": usage,
                    "model": result.get("model", self.model_id),
                    "frames_analyzed": len(frames_base64),
                    "has_transcript": transcript_text is not None,
                    "raw_response": result
                }
            else:
                return {
                    "success": False,
                    "error": "No valid response received from model",
                    "raw_response": result,
                    "video_url": video_url
                }
                
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "Request timed out. Video may be too long or network connection may be slow.",
                "video_url": video_url
            }
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "video_url": video_url
            }
    
    def analyze_video_detailed(
        self,
        video_url: str,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Analyzes video from different angles
        
        Args:
            video_url: YouTube video URL
            analysis_type: Analysis type
                - "comprehensive": Comprehensive analysis
                - "summary": Summary
                - "topics": Topics
                - "key_points": Key points
                - "visual_description": Visual description
        
        Returns:
            Analysis results
        """
        questions = {
            "comprehensive": """
Analyze this video comprehensively:
1. What is the video's main topic?
2. What key points are covered?
3. What is shown visually in the video?
4. What is the video's purpose and target audience?
5. What are the important details and examples?

Provide a detailed and structured analysis.
""",
            "summary": "Summarize this video briefly and concisely. State the main message and key points.",
            "topics": "What topics are covered in this video? List the topics.",
            "key_points": "List the key points of this video item by item.",
            "visual_description": "What is shown visually in this video? Explain scenes, visual content and visual elements in detail."
        }
        
        question = questions.get(analysis_type, questions["comprehensive"])
        
        return self.understand_video(video_url, question=question)
    
    def compare_videos(
        self,
        video_urls: List[str],
        comparison_aspect: str = "general"
    ) -> Dict[str, Any]:
        """
        Compares multiple videos
        
        Args:
            video_urls: List of video URLs to compare
            comparison_aspect: Comparison aspect
        
        Returns:
            Comparison results
        """
        if len(video_urls) < 2:
            return {
                "success": False,
                "error": "At least 2 video URLs required"
            }
        
        question = f"""
Compare the following {len(video_urls)} videos from {comparison_aspect} perspective:
- What are the similarities?
- What are the differences?
- Which one is better and why?
- What are the strengths and weaknesses of each?

Provide a detailed comparison.
"""
        
        # Analyze first video
        results = []
        for i, video_url in enumerate(video_urls, 1):
            print(f"\n📹 Analyzing video {i}/{len(video_urls)}...")
            result = self.understand_video(video_url, question=question)
            results.append({
                "video_number": i,
                "video_url": video_url,
                "result": result
            })
        
        return {
            "success": True,
            "videos_analyzed": len(video_urls),
            "comparison_aspect": comparison_aspect,
            "results": results
        }
    
    def close(self):
        """Closes the client"""
        self.client.close()


def main():
    """Main function - CLI interface"""
    print("=" * 60)
    print("🎥 YouTube Video Understanding System")
    print("🤖 NVIDIA Nemotron Nano 12B 2 VL")
    print("=" * 60)
    print()
    
    # API key check
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY environment variable not found.")
        api_key = input("Enter your OpenRouter API Key: ").strip()
        if not api_key:
            print("❌ API key required!")
            return
    
    # Create agent
    try:
        agent = VideoUnderstandingAgent(api_key=api_key)
    except Exception as e:
        print(f"❌ Could not create agent: {e}")
        return
    
    print()
    print("📋 Usage:")
    print("1. Single video analysis")
    print("2. Detailed analysis (comprehensive)")
    print("3. Video comparison")
    print("4. Ask custom question")
    print()
    
    while True:
        try:
            choice = input("Your choice (1-4, 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                break
            
            if choice == '1':
                video_url = input("\nEnter YouTube video URL: ").strip()
                if video_url:
                    result = agent.understand_video(video_url)
                    display_result(result)
            
            elif choice == '2':
                video_url = input("\nEnter YouTube video URL: ").strip()
                if video_url:
                    print("\n📊 Analysis types:")
                    print("1. comprehensive - Comprehensive analysis")
                    print("2. summary - Summary")
                    print("3. topics - Topics")
                    print("4. key_points - Key points")
                    print("5. visual_description - Visual description")
                    
                    analysis_choice = input("Your choice (1-5) [1]: ").strip() or "1"
                    analysis_types = {
                        "1": "comprehensive",
                        "2": "summary",
                        "3": "topics",
                        "4": "key_points",
                        "5": "visual_description"
                    }
                    analysis_type = analysis_types.get(analysis_choice, "comprehensive")
                    
                    result = agent.analyze_video_detailed(video_url, analysis_type)
                    display_result(result)
            
            elif choice == '3':
                print("\n📹 Enter video URLs to compare (at least 2):")
                video_urls = []
                while True:
                    url = input(f"Video {len(video_urls) + 1} URL (Enter to finish): ").strip()
                    if not url:
                        break
                    video_urls.append(url)
                
                if len(video_urls) >= 2:
                    comparison_aspect = input("Comparison aspect [general]: ").strip() or "general"
                    result = agent.compare_videos(video_urls, comparison_aspect)
                    display_comparison_result(result)
                else:
                    print("❌ At least 2 video URLs required!")
            
            elif choice == '4':
                video_url = input("\nEnter YouTube video URL: ").strip()
                question = input("Your question: ").strip()
                if video_url and question:
                    result = agent.understand_video(video_url, question=question)
                    display_result(result)
            
            else:
                print("❌ Invalid choice!")
            
            print("\n" + "-" * 60 + "\n")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    agent.close()


def display_result(result: Dict[str, Any]):
    """Displays results in a nice format"""
    print("\n" + "=" * 60)
    
    if not result.get("success"):
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        if "error_detail" in result:
            print(f"📋 Details: {result['error_detail']}")
        return
    
    print("✅ Video Analysis Successful!")
    print("-" * 60)
    print(f"📹 Video URL: {result.get('video_url', 'N/A')}")
    print(f"📹 Video ID: {result.get('video_id', 'N/A')}")
    print(f"❓ Question: {result.get('question', 'N/A')}")
    print("-" * 60)
    
    if "usage" in result:
        usage = result["usage"]
        print(f"📊 Token Usage:")
        print(f"   - Prompt: {usage.get('prompt_tokens', 'N/A')}")
        print(f"   - Completion: {usage.get('completion_tokens', 'N/A')}")
        print(f"   - Total: {usage.get('total_tokens', 'N/A')}")
        print("-" * 60)
    
    print("\n💬 Analysis Result:")
    print("=" * 60)
    print(result.get("answer", "Response not found"))
    print("=" * 60)


def display_comparison_result(result: Dict[str, Any]):
    """Displays comparison results"""
    print("\n" + "=" * 60)
    
    if not result.get("success"):
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        return
    
    print("✅ Video Comparison Completed!")
    print("-" * 60)
    print(f"📹 Videos Analyzed: {result.get('videos_analyzed', 0)}")
    print(f"🔍 Comparison Aspect: {result.get('comparison_aspect', 'N/A')}")
    print("-" * 60)
    
    for item in result.get("results", []):
        print(f"\n📹 Video {item.get('video_number', 'N/A')}:")
        print(f"   URL: {item.get('video_url', 'N/A')}")
        
        video_result = item.get("result", {})
        if video_result.get("success"):
            print(f"   ✅ Analysis successful")
            if "usage" in video_result:
                usage = video_result["usage"]
                print(f"   📊 Tokens: {usage.get('total_tokens', 'N/A')}")
        else:
            print(f"   ❌ Error: {video_result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
