import cv2
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm
import openai  # Changed from mistralai import Mistral

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
                use_openai=False,  # Changed from use_mistral
                openai_api_key=None,
                annotate_ui = True):  # Changed from mistral_api_key
        """
        Initialize the VideoExplainer.
        
        Args:
            video_path: Path to the video file
            question: Question about the video content
            output_dir: Directory to save output files
            screenshot_interval: Time interval (in seconds) between screenshots
            ui_detection_threshold: Confidence threshold for UI element detection
            scene_threshold: Threshold for scene change detection
            use_openai: Whether to use OpenAI for generating answers  # Changed from use_mistral
            openai_api_key: API key for OpenAI (for transcript analysis and answer generation)  # Changed from mistral_api_key
        """
        self.video_path = video_path
        self.question = question
        self.output_dir = output_dir
        self.screenshot_interval = screenshot_interval
        self.ui_detection_threshold = ui_detection_threshold
        self.scene_threshold = scene_threshold
        self.use_openai = use_openai  # Changed from use_mistral
        self.openai_api_key = openai_api_key 
        self.annotate_ui = annotate_ui # Changed from mistral_api_key

        print(f"VideoExplainer initialized with: use_openai={use_openai}, API key provided: {bool(openai_api_key)}")

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create screenshots directory
        self.screenshots_dir = os.path.join(output_dir, "screenshots")
        if not os.path.exists(self.screenshots_dir):
            os.makedirs(self.screenshots_dir)

        # Initialize models
        self._init_models(use_openai, openai_api_key)  # Changed from use_mistral, mistral_api_key

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

    def _init_models(self, use_openai, openai_api_key):  # Changed from use_mistral, mistral_api_key
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

        # Initialize OpenAI client if requested  # Changed Mistral to OpenAI
        self.openai_client = None  # Changed from mistral_client
        if use_openai:  # Changed from use_mistral
            try:
                # Debug output to verify the values
                print(f"Initializing OpenAI client with use_openai={use_openai}, API key provided: {bool(openai_api_key)}")

                # Set up OpenAI client
                if openai_api_key:
                    self.openai_client = openai
                    print("OpenAI client initialized successfully.")
                else:
                    print("Error: OpenAI API key not provided.")
                    raise ValueError("OpenAI API key is required when use_openai=True")
            except ImportError as ie:
                print(f"Error: {ie}")
                raise ImportError("openai module is required for OpenAI integration. Install with 'pip install -U openai'")
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                print(f"Exception type: {type(e)}")
                # Install the correct version of the openai client
                print("\nTry running: pip install -U openai")
                raise

    def process(self):
        """Process the video and generate an answer."""
        print(f"Processing video: {self.video_path}")
        
        # Check if the video has audio
        has_audio = self._check_for_audio()
        
        if has_audio:
            # Extract transcript
            self._extract_transcript()
            
            # Segment transcript
            self._segment_transcript()
        else:
            print("No audio detected in the video. Skipping transcript extraction.")
            # Create empty transcript data (but we'll still create segments based on screenshots later)
            self.transcript = "No audio detected in the video."
            self.transcript_segments = []
        
        # Detect scenes and capture screenshots
        self._detect_scenes_and_screenshots()
        
        # Detect UI elements
        self._detect_ui_elements()
        
        # Match transcript segments with screenshots (or create segments if no transcript)
        # The updated function creates artificial segments for videos without audio
        self.transcript_segments = self._match_transcript_with_screenshots()
        
        # Generate answer
        self._generate_answer()
        
        return self.answer
    
    def _check_for_audio(self):
        """
        A simple check to determine if the video has audio.
        Uses a lightweight approach without external dependencies.
        
        Returns:
            bool: True if the video likely has audio, False otherwise
        """
        try:
            # Approach 1: Try to extract a small audio sample
            import tempfile
            import os
            import numpy as np
            
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Use moviepy to extract a small audio sample (first 1 second)
                from moviepy.editor import VideoFileClip
                
                video = VideoFileClip(self.video_path)
                if video.audio is None:
                    print("MoviePy reports no audio stream in video")
                    video.close()
                    return False
                    
                # Try to extract a small audio sample
                try:
                    # Get first 1 second, or the whole clip if shorter
                    duration = min(1.0, video.duration)
                    subclip = video.subclip(0, duration)
                    
                    # Write audio to temp file
                    if subclip.audio is not None:
                        subclip.audio.write_audiofile(temp_path, verbose=False, logger=None)
                        video.close()
                        
                        # Check if the file exists and has content
                        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
                            print("Audio sample successfully extracted from video")
                            return True
                        else:
                            print("Audio sample extraction yielded no substantial data")
                            return False
                    else:
                        print("Subclip has no audio")
                        video.close()
                        return False
                except Exception as e:
                    print(f"Audio extraction failed: {e}")
                    video.close()
                    return False
                    
            except (ImportError, Exception) as e:
                print(f"MoviePy approach failed: {e}. Trying OpenCV approach...")
                
                # Approach 2: OpenCV-based check (less reliable)
                import cv2
                
                cap = cv2.VideoCapture(self.video_path)
                
                # Some versions of OpenCV can check audio presence
                if hasattr(cv2, 'CAP_PROP_AUDIO_ENABLED'):
                    has_audio = bool(cap.get(cv2.CAP_PROP_AUDIO_ENABLED))
                    cap.release()
                    return has_audio
                
                # If we can't definitively check, look at file format/size heuristics
                file_size = os.path.getsize(self.video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                cap.release()
                
                # Very rough heuristic based on common video formats
                # Audio typically adds at least a few hundred KB
                expected_video_size = frame_count * width * height * 0.1  # Very rough estimate
                
                if (file_size > expected_video_size * 1.2 and  # Size is noticeably larger than expected
                    frame_count > 30):                          # Not just a few frames
                    print("File characteristics suggest video likely has audio")
                    return True
                else:
                    print("File characteristics suggest video might not have audio")
                    return False
                    
        except Exception as e:
            print(f"Audio detection failed with error: {e}")
            print("Defaulting to assume video has audio")
            return True
        finally:
            # Clean up temporary file
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

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
        from utils.ui_detection import detect_ui_elements, detect_specialized_ui_elements, annotate_ui_elements, create_annotation_legend, summarize_ui_detections
        import os
        
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
        
        # Generate and save a summary of detected UI elements
        summary_path = os.path.join(self.output_dir, "ui_detection_summary.txt")
        summarize_ui_detections(self.screenshots, summary_path)
        
        # Only annotate UI elements if requested
        if self.annotate_ui:
            # Annotate UI elements on screenshots
            print("Annotating detected UI elements...")
            os.makedirs(annotated_dir, exist_ok=True)
            self.screenshots = annotate_ui_elements(self.screenshots, annotated_dir)
            
            # Create a legend image to explain the annotation colors
            legend_path = create_annotation_legend(annotated_dir)
            print(f"Created annotation legend at {legend_path}")
        else:
            print("Skipping UI annotation (use --annotate-ui flag to enable)")

    def _match_transcript_with_screenshots(self):
        """Match transcript segments with relevant screenshots."""
        from utils.ui_detection import match_transcript_with_screenshots

        print("Matching transcript segments with screenshots...")
        updated_segments = match_transcript_with_screenshots(
            self.transcript_segments if self.transcript_segments is not None else [], 
            self.screenshots
        )
        
        # Always ensure we have valid segments
        if updated_segments is not None:
            self.transcript_segments = updated_segments
        elif self.transcript_segments is None:
            self.transcript_segments = []
        
        print(f"After matching: transcript_segments is {type(self.transcript_segments)} with {len(self.transcript_segments)} segments")
        return self.transcript_segments

    def _generate_answer(self):
        """Generate an answer based on transcript and screenshots using OpenAI."""
        print("Generating answer with OpenAI...")

        # Debug the status of the OpenAI client and transcript segments
        print(f"OpenAI client status: use_openai={self.use_openai}, client is {'initialized' if self.openai_client is not None else 'NOT initialized'}")
        print(f"Transcript segments status: {type(self.transcript_segments)}")

        if not self.use_openai or self.openai_client is None:
            raise ValueError("OpenAI client is not initialized. Make sure to set use_openai=True and provide a valid API key.")

        # Ensure transcript_segments is never None
        if self.transcript_segments is None:
            print("Warning: transcript_segments was None. Creating empty list...")
            self.transcript_segments = []
        
        # Build context from transcript segments and screenshots
        context = []
        
        # Safely check for real transcript
        has_real_transcript = False
        for segment in self.transcript_segments:
            # Use dict.get() to safely access text field
            segment_text = segment.get("text", "")
            if not segment_text.startswith("[No audio]"):
                has_real_transcript = True
                break

        print("Generating answer with OpenAI...")
        
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
                for ui_element in screenshot.get("ui_elements", []):
                    segment_context["ui_elements"].append({
                        "label": ui_element.get("label", "unknown"),
                        "screenshot": os.path.basename(screenshot["path"])
                    })

            context.append(segment_context)

        # Extract video filename from path
        video_filename = os.path.basename(self.video_path)
        
        # Create a video metadata object to include in the context
        video_metadata = {
            "filename": video_filename,
            "duration": f"{timedelta(seconds=self.duration)}",
            "frame_count": self.frame_count,
            "fps": self.fps,
            "has_audio": has_real_transcript,  # Accurately indicate if the video has real audio
            "screenshot_count": len(self.screenshots),
            "ui_element_count": sum(len(s.get("ui_elements", [])) for s in self.screenshots),
            "scene_count": len(self.scenes)
        }

        # Create prompt for OpenAI with special instructions for videos without audio
        prompt = self._create_prompt(context, video_metadata, has_audio=has_real_transcript)

        # Call OpenAI API
        try:
            print("Attempting to call OpenAI API...")

            # First, try the Anthropic Claude-compatible format API format 
            try:
                print("Trying GPT API...")
                response = self.openai_client.responses.create(
                    model="gpt-4.1",  # Try with the latest available model
                    input=[
                        {"role": "system", "content": f"You are a helpful assistant that analyzes videos based on {'transcripts and ' if has_real_transcript else ''}screenshots. The video being analyzed is {video_filename}.{'The video has no audio, so focus on visual analysis.' if not has_real_transcript else ''}"},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract content from response
                openai_answer = response.output_text
                print("Successfully used Claude-compatible API format")
                
            except (AttributeError, TypeError) as api_format_error:
                # If the first approach fails, try the standard OpenAI chat completions format
                print(f"Claude-compatible format failed ({api_format_error}), trying standard OpenAI chat completions format...")
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o" if hasattr(self, 'openai_model') and '4' in self.openai_model else "gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You are a helpful assistant that analyzes videos based on {'transcripts and ' if has_real_transcript else ''}screenshots. The video being analyzed is {video_filename}.{'The video has no audio, so focus on visual analysis.' if not has_real_transcript else ''}"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5
                )
                
                # Extract content from response
                openai_answer = response.choices[0].message.content
                print("Successfully used standard OpenAI chat completions format")

            print(f"Successfully received response from OpenAI API")
            self.answer = openai_answer

            # Process the answer to include actual screenshot images
            # Find all references to screenshots in the answer
            enhanced_answer = self._enhance_answer_with_images(openai_answer)

            # Save the enhanced answer to a file
            answer_path = os.path.join(self.output_dir, "answer.md")
            with open(answer_path, "w") as f:
                f.write(enhanced_answer)

            print(f"Enhanced answer generated and saved to {answer_path}")

            # Keep the original answer text for other uses
            self.answer = openai_answer

        except Exception as e:
            print(f"Error generating answer with OpenAI: {e}")  # Changed from Mistral
            print(f"Exception type: {type(e)}")
            raise

        return self.answer

    def _enhance_answer_with_images(self, answer_text):
        """
        Enhance the answer by adding actual screenshot images where they are referenced.
        
        Args:
            answer_text: The original answer text from OpenAI  # Changed from Mistral
            
        Returns:
            str: Enhanced answer with embedded images
        """
        import re

        # Create a dictionary to map screenshot filenames to their full paths
        screenshot_map = {}

        # Collect all screenshots paths
        # Inside the for loop where you build screenshot_map
        for screenshot in self.screenshots:
            # Use annotated screenshots if available
            if hasattr(self, 'annotate_ui') and self.annotate_ui and "annotated_path" in screenshot:
                path = os.path.join("../", screenshot["annotated_path"])
            else:
                path = os.path.join("../", screenshot["path"])

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

            # Inside the fallback summary section
            for idx in key_indices:
                screenshot = self.screenshots[idx]
                if hasattr(self, 'annotate_ui') and self.annotate_ui and "annotated_path" in screenshot:
                    path = os.path.join("../", screenshot["annotated_path"])
                else:
                    path = os.path.join("..", screenshot["path"])

                timestamp = timedelta(seconds=screenshot["timestamp"])
                enhanced_answer += f"### Timestamp: {timestamp}\n\n"
                enhanced_answer += f"![Screenshot at {timestamp}]({path})\n\n"

        return enhanced_answer

    def _create_prompt(self, context, video_metadata=None, has_audio=True):
        """Create a prompt for OpenAI API."""  # Changed from Mistral API
        
        video_info = ""
        if video_metadata:
            video_info = f"""
            Video Information:
            - Filename: {video_metadata['filename']}
            - Duration: {video_metadata['duration']}
            - Frame Count: {video_metadata['frame_count']}
            - FPS: {video_metadata['fps']}
            - Has Audio: {video_metadata['has_audio']}
            - Screenshot Count: {video_metadata['screenshot_count']}
            - UI Element Count: {video_metadata['ui_element_count']}
            - Scene Count: {video_metadata['scene_count']}
            """
        
        # Adjust the prompt based on whether the video has audio
        if has_audio:
            transcript_info = "Here is the transcript of the video, segmented with timestamps and associated screenshots:"
        else:
            transcript_info = "This video has no audio. Instead, I've created visual segments based on the screenshots, with timestamps:"
        
        prompt = f"""
        I need you to analyze a video and answer this question: "{self.question}"
        
        {video_info}
        
        {transcript_info}
        
        {json.dumps(context, indent=2)}
        I need you to analyze a video and answer this question: "{self.question}"
        
        You are assisting with analyzing a video walkthrough of a SaaS application. The goal is to understand what the user is doing and answer a specific question about their actions.

        Below is the transcript, segmented with timestamps and the corresponding screenshots extracted from the video. Each segment may also include information about detected UI elements.
        
        {json.dumps(context, indent=2)}
        
        Based on this information, please:
            1. Explain what's happening in the video in a step by step manner
            2. Answer the question directly
            3. Reference the most relevant screenshot used in your explanation. Ensure screenshots are referenced in ascending timestamp order, reflecting the true sequence of events. The user should know the scene changes and clicks that occurred.
            4. Format your response in markdown
            5. Be clear and concise
            6. Do NOT ask for clarification or additional information
        
        When referencing screenshots, please use the format "Screenshot X" or "screenshot_X.jpg" so they can be included in the final output.

        Focus on relevance and ensure that the visual context (screenshots) aligns correctly with the actions being described. Prioritize clarity and correct temporal ordering when walking through the user’s interaction.
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