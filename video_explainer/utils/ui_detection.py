"""
UI element detection and matching utilities
"""
import cv2
import pytesseract
from PIL import Image
from tqdm import tqdm

def detect_ui_elements(screenshots, ui_detector):
    """
    Detect UI elements in screenshots
    
    Args:
        screenshots: List of screenshot information
        ui_detector: UI element detection model
        
    Returns:
        list: Updated screenshots with UI elements
    """
    for screenshot in tqdm(screenshots):
        # Load image for UI detection
        image = Image.open(screenshot["path"])
        
        # Detect UI elements
        detections = ui_detector(image)
        
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
    
    return screenshots

def match_transcript_with_screenshots(transcript_segments, screenshots):
    """
    Match transcript segments with relevant screenshots
    
    Args:
        transcript_segments: List of transcript segments
        screenshots: List of screenshots
        
    Returns:
        list: Updated transcript segments with matched screenshots
    """
    for segment in transcript_segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        
        # Find screenshots that fall within this segment's time range
        matched_screenshots = []
        for screenshot in screenshots:
            if segment_start <= screenshot["timestamp"] <= segment_end:
                matched_screenshots.append(screenshot)
        
        # If no screenshots in exact range, get closest ones
        if not matched_screenshots:
            # Find closest screenshot before segment start
            closest_before = None
            min_diff_before = float('inf')
            for screenshot in screenshots:
                if screenshot["timestamp"] < segment_start:
                    diff = segment_start - screenshot["timestamp"]
                    if diff < min_diff_before:
                        min_diff_before = diff
                        closest_before = screenshot
            
            # Find closest screenshot after segment end
            closest_after = None
            min_diff_after = float('inf')
            for screenshot in screenshots:
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
    
    return transcript_segments