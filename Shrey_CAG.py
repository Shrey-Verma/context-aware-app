import cv2
import os
import numpy as np
import pytesseract
import whisper
import torch
from PIL import Image
from transformers import pipeline
from scenedetect import VideoManager, ContentDetector
import json
import matplotlib.pyplot as plt
from datetime import timedelta
import argparse
from tqdm import tqdm
from scenedetect import VideoManager, ContentDetector, SceneManager
from mistralai import Mistral


class VideoExplainer:
    def __init__(self, 
                video_path, 
                question, 
                output_dir="output", 
                screenshot_interval=2, 
                ui_detection_threshold=0.4,
                scene_threshold=30.0,
                use_mistral=False,
                mistral_api_key=None):
        """
        Initialize the VideoExplainer.
        
        Args:
            video_path: Path to the video file
            question: Question about the video content
            output_dir: Directory to save output files
            screenshot_interval: Time interval (in seconds) between screenshots
            ui_detection_threshold: Confidence threshold for UI element detection
            scene_threshold: Threshold for scene change detection
            use_mistral: Whether to use Mistral for generating answers
            mistral_api_key: API key for Mistral (for transcript analysis and answer generation)
        """
        self.video_path = video_path
        self.question = question
        self.output_dir = output_dir
        self.screenshot_interval = screenshot_interval
        self.ui_detection_threshold = ui_detection_threshold
        self.scene_threshold = scene_threshold
        self.use_mistral = use_mistral
        self.mistral_api_key = mistral_api_key  # Store the API key as an instance variable
        
        print(f"VideoExplainer initialized with: use_mistral={use_mistral}, API key provided: {bool(mistral_api_key)}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create screenshots directory
        self.screenshots_dir = os.path.join(output_dir, "screenshots")
        if not os.path.exists(self.screenshots_dir):
            os.makedirs(self.screenshots_dir)
        
        # Initialize models
        self._init_models(use_mistral, mistral_api_key)
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        
        # Store video analysis results
        self.transcript = None
        self.transcript_segments = []
        self.screenshots = []
        self.ui_elements = []
        self.scenes = []
        self.answer = None

    def _init_models(self, use_mistral, mistral_api_key):
        """Initialize the required models for analysis."""
        # Speech recognition model for transcription
        self.whisper_model = whisper.load_model("base")
        
        # UI element detection model
        self.ui_detector = pipeline("object-detection", 
                                model="facebook/detr-resnet-50", 
                                threshold=self.ui_detection_threshold)
        
        # Initialize Mistral client if requested
        self.mistral_client = None
        if use_mistral:
            try:
                # Debug output to verify the values
                print(f"Initializing Mistral client with use_mistral={use_mistral}, API key provided: {bool(mistral_api_key)}")
                
                # Use the new Mistral client as per migration guide
                try:
                    from mistralai import Mistral
                    print("Imported MistralClient from mistralai.client.mistral_client")
                except ImportError:
                    try:
                        # Alternative import for newer versions
                        from mistralai.async_client import MistralClient
                        print("Imported MistralClient from mistralai.async_client")
                    except ImportError:
                        try:
                            # Another alternative for the latest version
                            from mistralai.models.chat_completion import ChatMessage
                            from mistralai.client import MistralClient
                            print("Imported latest MistralClient version")
                        except ImportError:
                            raise ImportError("Could not import MistralClient. Install with 'pip install -U mistralai'")
                
                if mistral_api_key:
                    self.mistral_client = MistralClient(api_key=mistral_api_key)
                    print("Mistral client initialized successfully.")
                else:
                    print("Error: Mistral API key not provided.")
                    raise ValueError("Mistral API key is required when use_mistral=True")
            except ImportError as ie:
                print(f"Error: {ie}")
                raise ImportError("mistralai module is required for Mistral integration. Install with 'pip install -U mistralai'")
            except Exception as e:
                print(f"Error initializing Mistral client: {e}")
                print(f"Exception type: {type(e)}")
                # Install the correct version of the mistralai client
                print("\nTry running: pip install -U mistralai")
                raise
    
    def process(self):
        """Process the video and generate an answer."""
        print(f"Processing video: {self.video_path}")
        
        # Extract transcript
        self._extract_transcript()
        
        # Segment transcript
        self._segment_transcript()
        
        # Detect scenes and capture screenshots
        self._detect_scenes_and_screenshots()
        
        # Detect UI elements
        self._detect_ui_elements()
        
        # Match transcript segments with screenshots
        self._match_transcript_with_screenshots()
        
        # Generate answer
        self._generate_answer()
        
        return self.answer
    
    def _extract_transcript(self):
        """Extract transcript from the video."""
        print("Extracting transcript...")
        result = self.whisper_model.transcribe(self.video_path)
        self.transcript = result["text"]
        
        # Get segments with timestamps
        self.transcript_segments = result["segments"]
        
        print(f"Transcript extracted: {len(self.transcript_segments)} segments")
    
    def _segment_transcript(self):
        """Segment the transcript into meaningful chunks."""
        # Already segmented by whisper but we can refine segments if needed
        print("Refining transcript segments...")
        
        # Convert whisper segments to our custom format
        refined_segments = []
        for segment in self.transcript_segments:
            refined_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "screenshots": []
            })
        
        self.transcript_segments = refined_segments
    
    def _detect_scenes_and_screenshots(self):
        """Detect scene changes and capture screenshots."""
        print("Detecting scenes and capturing screenshots...")
        
        # Initialize scene detection
        video_manager = VideoManager([self.video_path])
        scene_detector = ContentDetector(threshold=self.scene_threshold)
        
        # Start video manager
        video_manager.start()
        
        # Create a scene list
        scene_manager = SceneManager()
        scene_manager.add_detector(scene_detector)
        
        # Detect scenes     
        scene_manager.detect_scenes(video_manager)
        scene_list = scene_manager.get_scene_list()
        
        # Extract scene boundaries
        self.scenes = []
        for scene in scene_list:
            start_frame = scene[0].frame_num
            end_frame = scene[1].frame_num - 1  # End frame is exclusive
            start_time = start_frame / self.fps
            end_time = end_frame / self.fps
            self.scenes.append({
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": start_frame,
                "end_frame": end_frame
            })
        
        # Capture screenshots at scene changes and regular intervals
        self._capture_screenshots()
        
        print(f"Detected {len(self.scenes)} scenes and captured {len(self.screenshots)} screenshots")
    
    def _capture_screenshots(self):
        """Capture screenshots from the video."""
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Calculate frame indices for screenshots
        screenshot_frames = set()
        
        # Add scene change frames
        for scene in self.scenes:
            screenshot_frames.add(scene["start_frame"])
        
        # Add regular interval frames
        interval_frames = int(self.fps * self.screenshot_interval)
        for i in range(0, self.frame_count, interval_frames):
            screenshot_frames.add(i)
        
        # Sort frames
        screenshot_frames = sorted(list(screenshot_frames))
        
        # Capture screenshots
        for frame_idx in tqdm(screenshot_frames):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                timestamp = frame_idx / self.fps
                screenshot_path = os.path.join(self.screenshots_dir, f"screenshot_{frame_idx:06d}.jpg")
                cv2.imwrite(screenshot_path, frame)
                
                self.screenshots.append({
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "path": screenshot_path,
                    "ui_elements": []
                })
    
    def _detect_ui_elements(self):
        """Detect UI elements in the screenshots."""
        print("Detecting UI elements in screenshots...")
        
        for screenshot in tqdm(self.screenshots):
            # Load image for UI detection
            image = Image.open(screenshot["path"])
            
            # Detect UI elements
            detections = self.ui_detector(image)
            
            # Extract text with OCR
            img_cv = cv2.imread(screenshot["path"])
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            try:
                ocr_text = pytesseract.image_to_string(gray)
            except Exception as e:
                print(f"Warning: OCR failed for {screenshot['path']}: {e}")
                ocr_text = ""
            
            # Store UI elements
            ui_elements = []
            for detection in detections:
                ui_elements.append({
                    "label": detection["label"],
                    "score": detection["score"],
                    "box": detection["box"]
                })
            
            screenshot["ui_elements"] = ui_elements
            screenshot["ocr_text"] = ocr_text
    
    def _match_transcript_with_screenshots(self):
        """Match transcript segments with relevant screenshots."""
        print("Matching transcript segments with screenshots...")
        
        for segment in self.transcript_segments:
            segment_start = segment["start"]
            segment_end = segment["end"]
            
            # Find screenshots that fall within this segment's time range
            matched_screenshots = []
            for screenshot in self.screenshots:
                if segment_start <= screenshot["timestamp"] <= segment_end:
                    matched_screenshots.append(screenshot)
            
            # If no screenshots in exact range, get closest ones
            if not matched_screenshots:
                # Find closest screenshot before segment start
                closest_before = None
                min_diff_before = float('inf')
                for screenshot in self.screenshots:
                    if screenshot["timestamp"] < segment_start:
                        diff = segment_start - screenshot["timestamp"]
                        if diff < min_diff_before:
                            min_diff_before = diff
                            closest_before = screenshot
                
                # Find closest screenshot after segment end
                closest_after = None
                min_diff_after = float('inf')
                for screenshot in self.screenshots:
                    if screenshot["timestamp"] > segment_end:
                        diff = screenshot["timestamp"] - segment_end
                        if diff < min_diff_after:
                            min_diff_after = diff
                            closest_after = screenshot
                
                # Add closest screenshots
                if closest_before:
                    matched_screenshots.append(closest_before)
                if closest_after:
                    matched_screenshots.append(closest_after)
            
            # Sort matched screenshots by timestamp
            matched_screenshots.sort(key=lambda x: x["timestamp"])
            
            # Add unique screenshots to segment
            segment["screenshots"] = []
            seen_paths = set()
            for screenshot in matched_screenshots:
                if screenshot["path"] not in seen_paths:
                    segment["screenshots"].append(screenshot)
                    seen_paths.add(screenshot["path"])
    
    def _generate_answer(self):
        """Generate an answer based on transcript and screenshots using Mistral."""
        print("Generating answer with Mistral...")
        
        # Debug the status of the Mistral client
        print(f"Mistral client status: use_mistral={self.use_mistral}, client is {'initialized' if self.mistral_client is not None else 'NOT initialized'}")
        
        if not self.use_mistral or self.mistral_client is None:
            raise ValueError("Mistral client is not initialized. Make sure to set use_mistral=True and provide a valid API key.")
        
        # Build context from transcript segments and screenshots
        context = []
        for i, segment in enumerate(self.transcript_segments):
            segment_context = {
                "id": i,
                "timestamp": f"{timedelta(seconds=segment['start'])} - {timedelta(seconds=segment['end'])}",
                "text": segment["text"],
                "screenshots": [os.path.basename(s["path"]) for s in segment["screenshots"]],
                "ui_elements": []
            }
            
            # Add UI elements from the screenshots
            for screenshot in segment["screenshots"]:
                for ui_element in screenshot["ui_elements"]:
                    segment_context["ui_elements"].append({
                        "label": ui_element["label"],
                        "screenshot": os.path.basename(screenshot["path"])
                    })
            
            context.append(segment_context)
        
        # Create prompt for Mistral
        prompt = self._create_prompt(context)
        
        # Call Mistral API
        try:
            print("Attempting to call Mistral API...")
            
            # Import needed for the latest Mistral client
            try:
                
                messages = [{"role": "user", "content": prompt}]
                response = mistral_client.chat.complete(model="mistral-large-latest", messages=messages)
                
                # Extract content based on new API structure
                mistral_answer = response.choices[0].message.content
                
            except ImportError:
                print("Falling back to alternative structure without ChatMessage")
                
                # Alternative structure without explicit ChatMessage import
                response = self.mistral_client.chat(
                    model="mistral-large-latest",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract content
                mistral_answer = response.choices[0].message.content
            
            print(f"Successfully received response from Mistral API")
            self.answer = mistral_answer
            
            # Save the answer to a file
            answer_path = os.path.join(self.output_dir, "answer.md")
            with open(answer_path, "w") as f:
                f.write(self.answer)
            
            print(f"Answer generated and saved to {answer_path}")
            
        except Exception as e:
            print(f"Error generating answer with Mistral: {e}")
            print(f"Exception type: {type(e)}")
            raise
        
        return self.answer
    
    def _create_prompt(self, context):
        """Create a prompt for Mistral API."""
        prompt = f"""
        I need you to analyze a video and answer this question: "{self.question}"
        
        Here is the transcript of the video, segmented with timestamps and associated screenshots:
        
        {json.dumps(context, indent=2)}
        
        Based on this information, please:
        1. Explain what's happening in the video
        2. Answer the question directly
        3. Include references to specific screenshots when relevant
        4. Format your response in markdown
        5. Be clear and concise
        
        Remember to focus on answering the question: "{self.question}"
        """
        return prompt
    
    def visualize_results(self):
        """Visualize the results with a timeline of screenshots and transcript."""
        print("Visualizing results...")
        
        # Create a timeline visualization
        plt.figure(figsize=(15, 8))
        
        # Plot transcript segments
        for i, segment in enumerate(self.transcript_segments):
            plt.barh(1, segment["end"] - segment["start"], left=segment["start"], height=0.5, 
                     color='lightblue', alpha=0.7)
            
            # Add segment text
            if i % 5 == 0:  # Show every 5th segment text to avoid clutter
                plt.text(segment["start"], 1.25, f"Segment {i}", fontsize=8)
        
        # Plot screenshots
        for i, screenshot in enumerate(self.screenshots):
            plt.scatter(screenshot["timestamp"], 2, color='red', s=50)
            
            # Add screenshot filename
            if i % 3 == 0:  # Show every 3rd screenshot name to avoid clutter
                plt.text(screenshot["timestamp"], 2.1, os.path.basename(screenshot["path"]), 
                         fontsize=8, rotation=90)
        
        # Plot scenes
        for i, scene in enumerate(self.scenes):
            plt.barh(3, scene["end_time"] - scene["start_time"], left=scene["start_time"], 
                     height=0.5, color='green', alpha=0.7)
            plt.text(scene["start_time"], 3.25, f"Scene {i}", fontsize=8)
        
        # Add labels
        plt.yticks([1, 2, 3], ["Transcript", "Screenshots", "Scenes"])
        plt.xlabel("Time (seconds)")
        plt.title("Video Analysis Timeline")
        
        # Save the visualization
        viz_path = os.path.join(self.output_dir, "visualization.png")
        plt.savefig(viz_path)
        plt.close()
        
        print(f"Visualization saved to {viz_path}")
    
    def create_report(self):
        """Create a comprehensive report with the analysis results."""
        print("Creating report...")
        
        # Create a markdown report
        report = f"# Video Analysis Report\n\n"
        report += f"## Question\n\n{self.question}\n\n"
        
        report += "## Video Information\n\n"
        report += f"- **Filename:** {os.path.basename(self.video_path)}\n"
        report += f"- **Duration:** {timedelta(seconds=self.duration)}\n"
        report += f"- **Frame Count:** {self.frame_count}\n"
        report += f"- **FPS:** {self.fps}\n\n"
        
        report += "## Transcript\n\n"
        report += "```\n"
        report += self.transcript[:1000] + "...\n"  # Truncate if too long
        report += "```\n\n"
        
        report += "## Screenshots\n\n"
        for i, screenshot in enumerate(self.screenshots[:5]):  # Show first 5 screenshots
            report += f"### Screenshot {i+1}\n\n"
            report += f"- **Timestamp:** {timedelta(seconds=screenshot['timestamp'])}\n"
            report += f"- **Frame:** {screenshot['frame']}\n"
            report += f"- **UI elements:** {len(screenshot['ui_elements'])}\n\n"
            report += f"![Screenshot {i+1}]({screenshot['path']})\n\n"
        
        report += "## Answer\n\n"
        report += self.answer
        
        # Save the report
        report_path = os.path.join(self.output_dir, "report.md")
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Video Explainer: Analyze videos and answer questions")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--question", required=True, help="Question about the video")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--interval", type=float, default=2.0, help="Screenshot interval in seconds")
    parser.add_argument("--use-mistral", action="store_true", help="Use Mistral for answer generation")
    parser.add_argument("--api-key", help="Mistral API key for answer generation")
    parser.add_argument("--visualize", action="store_true", help="Create visualization of the analysis")
    parser.add_argument("--report", action="store_true", help="Create a comprehensive report")
    
    args = parser.parse_args()
    
    # If your API key is hardcoded, you can set it here
    # This is useful for debugging but not recommended for production
    MISTRAL_API_KEY = "Pa802ifoFCMuGHfQH4KCiLmkmnFpUVRf"
    
    # Use the API key from command line or the hardcoded one
    api_key = args.api_key or MISTRAL_API_KEY
    
    # FORCE MISTRAL TO BE TRUE for debugging purposes
    # In production, you'd use args.use_mistral instead
    use_mistral = True  # Set this to True to force using Mistral
    
    # Validate that API key is provided when use-mistral is True
    if use_mistral and not api_key:
        print("Error: API key is required when use_mistral=True")
        return
    
    try:
        print(f"Creating VideoExplainer with use_mistral={use_mistral}, API key provided: {bool(api_key)}")
        
        # Create VideoExplainer
        explainer = VideoExplainer(
            video_path=args.video,
            question=args.question,
            output_dir=args.output,
            screenshot_interval=args.interval,
            use_mistral=use_mistral,  # Use our forced value instead of args.use_mistral
            mistral_api_key=api_key
        )
        
        # Process the video
        print("Starting video processing...")
        answer = explainer.process()
        
        # Create visualization if requested
        if args.visualize:
            explainer.visualize_results()
        
        # Create report if requested
        if args.report:
            explainer.create_report()
        
        print("\nDone! Results saved to:", args.output)
        print("\nAnswer preview:")
        print("-" * 40)
        print(answer[:500] + "..." if len(answer) > 500 else answer)
        print("-" * 40)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full traceback for better debugging
        raise

if __name__ == "__main__":
    main()