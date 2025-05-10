import cv2
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm
from mistralai import Mistral

# Import from utility modules
from utils.transcript import extract_transcript, segment_transcript
from utils.scene_detection import detect_scenes, capture_screenshots, detect_scenes_and_screenshots
from utils.ui_detection import detect_ui_elements, match_transcript_with_screenshots


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
        # Import the modules here to avoid circular imports
        import whisper
        from transformers import pipeline
        
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
                    self.mistral_client = Mistral(api_key=mistral_api_key)
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
        # Use the scene detection utility
        # from utils.scene_detection import detect_scenes_and_screenshots
        
        print("Detecting scenes and capturing screenshots...")
        scenes, screenshots = detect_scenes_and_screenshots(
            self.video_path, 
            self.screenshots_dir,
            self.fps,
            self.frame_count,
            self.cap,
            self.scene_threshold, 
            self.screenshot_interval
        )
        
        self.scenes = scenes
        self.screenshots = screenshots
        
        print(f"Detected {len(self.scenes)} scenes and captured {len(self.screenshots)} screenshots")
    
    def _detect_ui_elements(self):
        """Detect UI elements in the screenshots."""
        from utils.ui_detection import detect_ui_elements
        
        print("Detecting UI elements in screenshots...")
        self.screenshots = detect_ui_elements(self.screenshots, self.ui_detector)
    
    def _match_transcript_with_screenshots(self):
        """Match transcript segments with relevant screenshots."""
        from utils.ui_detection import match_transcript_with_screenshots
        
        print("Matching transcript segments with screenshots...")
        self.transcript_segments = match_transcript_with_screenshots(
            self.transcript_segments, 
            self.screenshots
        )
    
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
                response = self.mistral_client.chat.complete(model="mistral-large-latest", messages=messages)
                
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
        from video_explainer.utils.visualization import create_timeline_visualization
        
        print("Visualizing results...")
        viz_path = os.path.join(self.output_dir, "visualization.png")
        
        create_timeline_visualization(
            self.transcript_segments,
            self.screenshots,
            self.scenes,
            viz_path
        )
        
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