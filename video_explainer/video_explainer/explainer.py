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
from utils.scene_detection import capture_screenshots, detect_scenes_and_screenshots
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
        
        # Initialize both UI detection models
        # 1. General object detection with DETR (keep as fallback)
        self.general_ui_detector = pipeline("object-detection", 
                                model="facebook/detr-resnet-50", 
                                threshold=self.ui_detection_threshold)
        
        # 2. Specialized UI detection with Microsoft LayoutLM
        try:
            print("Initializing specialized UI detection model...")
            # Microsoft's LayoutLM for document and UI layout analysis
            # We'll use Hugging Face's pipeline for document-question-answering
            self.specialized_ui_detector = pipeline("document-question-answering", 
                                            model="impira/layoutlm-document-qa")
            self.use_specialized_ui = True
            print("Successfully initialized specialized LayoutLM model for UI detection")
        except Exception as e:
            print(f"Warning: Could not initialize specialized UI model: {e}")
            print("Falling back to general object detection only")
            self.use_specialized_ui = False
        
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
        from utils.ui_detection import detect_ui_elements, detect_specialized_ui_elements, annotate_ui_elements, create_annotation_legend
        
        print("Detecting UI elements in screenshots...")
        
        # Create annotated screenshots directory
        annotated_dir = os.path.join(self.output_dir, "annotated_screenshots")
        
        # Pass both detectors to the utility function
        if hasattr(self, 'use_specialized_ui') and self.use_specialized_ui:
            print("Using enhanced UI detection with Microsoft LayoutLM...")
            self.screenshots = detect_specialized_ui_elements(
                self.screenshots, 
                self.general_ui_detector, 
                self.specialized_ui_detector
            )
        else:
            # Fall back to original implementation
            print("Using standard UI detection...")
            self.screenshots = detect_ui_elements(
                self.screenshots, 
                self.general_ui_detector
            )
        
        # Annotate UI elements on screenshots
        print("Annotating detected UI elements...")
        self.screenshots = annotate_ui_elements(self.screenshots, annotated_dir)
        
        # Create a legend image to explain the annotation colors
        legend_path = create_annotation_legend(annotated_dir)
        print(f"Created annotation legend at {legend_path}")
    
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
            
            # Process the answer to include actual screenshot images
            # Find all references to screenshots in the answer
            enhanced_answer = self._enhance_answer_with_images(mistral_answer)
            
            # Save the enhanced answer to a file
            answer_path = os.path.join(self.output_dir, "answer.md")
            with open(answer_path, "w") as f:
                f.write(enhanced_answer)
            
            print(f"Enhanced answer generated and saved to {answer_path}")
            
            # Keep the original answer text for other uses
            self.answer = mistral_answer
            
        except Exception as e:
            print(f"Error generating answer with Mistral: {e}")
            print(f"Exception type: {type(e)}")
            raise
        
        return self.answer

    def _enhance_answer_with_images(self, answer_text):
        """
        Enhance the answer by adding actual screenshot images where they are referenced.
        
        Args:
            answer_text: The original answer text from Mistral
            
        Returns:
            str: Enhanced answer with embedded images
        """
        import re
        
        # Create a dictionary to map screenshot filenames to their full paths
        screenshot_map = {}
        
        # Collect all screenshots paths
        for screenshot in self.screenshots:
            # Use annotated screenshots if available
            if hasattr(self, 'annotate_ui') and self.annotate_ui and "annotated_path" in screenshot:
                path = screenshot["annotated_path"]
            else:
                path = screenshot["path"]
            
            # Map filename to full path
            basename = os.path.basename(path)
            screenshot_map[basename] = path
            
            # Also try matching without the "screenshot_" prefix
            if basename.startswith("screenshot_"):
                basename_short = basename[len("screenshot_"):]
                screenshot_map[basename_short] = path
        
        # Define patterns to look for screenshot references
        patterns = [
            r"screenshot_\d+\.jpg",
            r"screenshot[_\s]*(\d+)",
            r"Screenshot[_\s]*(\d+)",
            r"\(Screenshot[_\s]*(\d+)\)",
            r"screenshot (\d{6})",
            r"screenshot (\d+)"
        ]
        
        # Process the answer text
        enhanced_answer = answer_text
        
        # Track which screenshots have been added
        added_screenshots = set()
        
        # Search for patterns
        for pattern in patterns:
            matches = re.finditer(pattern, enhanced_answer)
            for match in matches:
                match_text = match.group(0)
                
                # Extract filename or number
                if "." in match_text:
                    # Full filename pattern
                    filename = match_text
                else:
                    # Extract number and try to find matching file
                    if '(' in match_text:
                        # Remove parentheses
                        match_text = match_text.replace('(', '').replace(')', '')
                    
                    # Extract digits
                    digits = re.search(r'(\d+)', match_text)
                    if digits:
                        number = digits.group(1)
                        
                        # Try different filename formats
                        for format_str in ["screenshot_{:06d}.jpg", "screenshot_{}.jpg"]:
                            filename = format_str.format(int(number))
                            if filename in screenshot_map:
                                break
                        else:
                            # If no match, try to find by index
                            try:
                                idx = int(number)
                                if 0 <= idx < len(self.screenshots):
                                    if "annotated_path" in self.screenshots[idx]:
                                        path = self.screenshots[idx]["annotated_path"]
                                    else:
                                        path = self.screenshots[idx]["path"]
                                    filename = os.path.basename(path)
                                else:
                                    continue  # Skip if index is out of range
                            except ValueError:
                                continue  # Skip if number conversion fails
                    else:
                        continue  # Skip if no digits found
                
                # Check if screenshot exists in our map
                if filename in screenshot_map and filename not in added_screenshots:
                    # Path to the screenshot
                    screenshot_path = screenshot_map[filename]
                    
                    # Create markdown for the image
                    image_markdown = f"\n\n![{match_text}]({screenshot_path})\n"
                    
                    # Replace the reference with the reference + image
                    # enhanced_answer = enhanced_answer.replace(match_text, match_text + image_markdown)
                    
                    # Add image at the end of the paragraph containing the reference
                    paragraph_end = enhanced_answer.find("\n\n", enhanced_answer.find(match_text))
                    if paragraph_end > 0:
                        enhanced_answer = enhanced_answer[:paragraph_end] + image_markdown + enhanced_answer[paragraph_end:]
                    else:
                        # If not found, append to the end
                        enhanced_answer += image_markdown
                    
                    # Mark as added to avoid duplicates
                    added_screenshots.add(filename)
        
        # Add a summary section with key screenshots if none were added
        if not added_screenshots and self.screenshots:
            enhanced_answer += "\n\n## Key Screenshots from the Video\n\n"
            
            # Add up to 5 key screenshots (evenly distributed)
            key_indices = []
            if len(self.screenshots) <= 5:
                key_indices = range(len(self.screenshots))
            else:
                step = len(self.screenshots) // 5
                key_indices = range(0, len(self.screenshots), step)[:5]
            
            for idx in key_indices:
                screenshot = self.screenshots[idx]
                if hasattr(self, 'annotate_ui') and self.annotate_ui and "annotated_path" in screenshot:
                    path = screenshot["annotated_path"]
                else:
                    path = screenshot["path"]
                
                timestamp = timedelta(seconds=screenshot["timestamp"])
                enhanced_answer += f"### Timestamp: {timestamp}\n\n"
                enhanced_answer += f"![Screenshot at {timestamp}]({path})\n\n"
        
        return enhanced_answer

    def _create_prompt(self, context):
        """Create a prompt for Mistral API."""
        prompt = f"""
        I need you to analyze a video and answer this question: "{self.question}"
        
        Here is the transcript of the video, segmented with timestamps and associated screenshots:
        
        {json.dumps(context, indent=2)}
        
        Based on this information, please:
        1. Explain what's happening in the video
        2. Answer the question directly
        3. Include references to specific screenshots when relevant (use the exact screenshot filename)
        4. Format your response in markdown
        5. Be clear and concise
        
        When referencing screenshots, please use the format "Screenshot X" or "screenshot_X.jpg" so they can be included in the final output.
        
        Remember to focus on answering the question: "{self.question}"
        """
        return prompt
    
    def visualize_results(self):
        """Visualize the results with a timeline of screenshots and transcript."""
        from utils.visualization import create_timeline_visualization
        
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
        
        # Add UI Detection Legend
        legend_path = os.path.join(self.output_dir, "annotated_screenshots", "annotation_legend.png")
        if os.path.exists(legend_path):
            report += "## UI Detection Legend\n\n"
            report += f"![UI Detection Legend]({legend_path})\n\n"
        
        report += "## Screenshots with UI Detection\n\n"
        for i, screenshot in enumerate(self.screenshots[:5]):  # Show first 5 screenshots
            report += f"### Screenshot {i+1}\n\n"
            report += f"- **Timestamp:** {timedelta(seconds=screenshot['timestamp'])}\n"
            report += f"- **Frame:** {screenshot['frame']}\n"
            
            # Add UI element statistics if available
            if "ui_structure" in screenshot:
                report += f"- **UI elements detected:** {screenshot['ui_structure']['total_elements']}\n"
                report += f"  - General elements (DETR): {screenshot['ui_structure']['general_elements']}\n"
                report += f"  - Specialized elements (LayoutLM): {screenshot['ui_structure']['specialized_elements']}\n"
                report += f"  - OCR-based elements: {screenshot['ui_structure']['ocr_elements']}\n"
            else:
                report += f"- **UI elements:** {len(screenshot['ui_elements'])}\n"
            
            report += "\n"
            
            # Use annotated image if available, otherwise use original
            image_path = screenshot.get("annotated_path", screenshot["path"])
            report += f"![Screenshot {i+1}]({image_path})\n\n"
            
            # Add OCR text if available and not empty
            if screenshot.get("ocr_text") and len(screenshot["ocr_text"].strip()) > 0:
                report += "**OCR Text:**\n```\n"
                report += screenshot["ocr_text"][:300] + ("..." if len(screenshot["ocr_text"]) > 300 else "")
                report += "\n```\n\n"
        
        report += "## Answer\n\n"
        report += self.answer
        
        # Save the report
        report_path = os.path.join(self.output_dir, "report.md")
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")