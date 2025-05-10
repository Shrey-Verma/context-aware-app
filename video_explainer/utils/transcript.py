"""
Transcript extraction and processing utilities
"""
import whisper

def extract_transcript(video_path, model_name="base"):
    """
    Extract transcript from a video file
    
    Args:
        video_path: Path to the video file
        model_name: Whisper model name
        
    Returns:
        tuple: (transcript_text, transcript_segments)
    """
    # Load the model
    model = whisper.load_model(model_name)
    
    # Extract transcript
    result = model.transcribe(video_path)
    
    return result["text"], result["segments"]

def segment_transcript(transcript_segments):
    """
    Convert whisper segments to our custom format
    
    Args:
        transcript_segments: Raw segments from whisper
        
    Returns:
        list: Refined transcript segments
    """
    refined_segments = []
    for segment in transcript_segments:
        refined_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "screenshots": []
        })
    
    return refined_segments