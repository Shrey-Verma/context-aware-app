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

def detect_specialized_ui_elements(screenshots, general_detector, specialized_detector):
    """
    Enhanced UI element detection using both general and specialized models
    
    Args:
        screenshots: List of screenshot information
        general_detector: General object detection model (DETR)
        specialized_detector: Specialized UI detection model (LayoutLM)
        
    Returns:
        list: Updated screenshots with enhanced UI elements
    """
    for screenshot in tqdm(screenshots):
        # Load image for UI detection
        image = Image.open(screenshot["path"])
        image_path = screenshot["path"]
        
        # 1. General object detection with DETR
        general_detections = general_detector(image)
        
        # 2. Extract text with OCR
        img_cv = cv2.imread(image_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        try:
            ocr_text = pytesseract.image_to_string(gray)
            # Also get detailed OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        except Exception as e:
            print(f"Warning: OCR failed for {image_path}: {e}")
            ocr_text = ""
            ocr_data = {"text": [], "conf": [], "left": [], "top": [], "width": [], "height": []}
        
        # 3. Use specialized LayoutLM model for UI element detection
        try:
            # Create UI element questions based on UI components we want to identify
            ui_questions = [
                "Find all buttons in this image",
                "Find all text input fields",
                "Find menu items",
                "Find all navigation elements"
            ]
            
            specialized_elements = []
            
            # Try each question to extract different UI elements
            for question in ui_questions:
                try:
                    result = specialized_detector(
                        image=image,
                        question=question
                    )
                    
                    # Process result
                    if isinstance(result, dict) and "answer" in result and len(result["answer"]) > 0:
                        # Create a label based on the question
                        if "button" in question.lower():
                            label = "button"
                        elif "input" in question.lower():
                            label = "input_field"
                        elif "menu" in question.lower():
                            label = "menu_item"
                        elif "navigation" in question.lower():
                            label = "navigation"
                        else:
                            label = "ui_element"
                        
                        # Add the detected element
                        specialized_elements.append({
                            "label": label,
                            "text": result["answer"],
                            "score": result.get("score", 0.7),
                            "box": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},  # Placeholder
                            "type": "layout_detected"
                        })
                    elif isinstance(result, list):
                        for item in result:
                            if isinstance(item, dict) and "answer" in item and len(item["answer"]) > 0:
                                # Similar processing for list results
                                if "button" in question.lower():
                                    label = "button"
                                elif "input" in question.lower():
                                    label = "input_field"
                                elif "menu" in question.lower():
                                    label = "menu_item"
                                elif "navigation" in question.lower():
                                    label = "navigation"
                                else:
                                    label = "ui_element"
                                
                                specialized_elements.append({
                                    "label": label,
                                    "text": item["answer"],
                                    "score": item.get("score", 0.7),
                                    "box": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},  # Placeholder
                                    "type": "layout_detected"
                                })
                except Exception as e:
                    print(f"Warning: Question '{question}' failed: {e}")
            
            print(f"LayoutLM detected {len(specialized_elements)} UI elements")
            
        except Exception as e:
            print(f"Warning: Specialized UI detection failed: {e}")
            print(f"Exception type: {type(e)}")
            specialized_elements = []
        
        # 4. Extract UI elements from OCR data to enhance detection
        ocr_ui_elements = []
        # Process OCR data to identify potential UI elements
        if "text" in ocr_data and len(ocr_data["text"]) > 0:
            for i in range(len(ocr_data["text"])):
                text = ocr_data["text"][i].strip()
                conf = int(ocr_data["conf"][i])
                
                # Skip empty or low confidence text
                if not text or conf < 50:
                    continue
                
                # Check if text matches common UI element patterns
                # Buttons often have short text with specific patterns
                if (len(text) < 20 and 
                   (text.lower() in ["ok", "cancel", "submit", "send", "login", "logout", "sign up", 
                                    "save", "delete", "edit", "update", "next", "previous", "back"] or
                    text.startswith("→") or text.endswith("→") or
                    text.startswith(">") or text.endswith(">"))):
                    
                    label = "button"
                    element_type = "ocr_button"
                    score = 0.8
                
                # Input fields often have label-like text
                elif (text.endswith(":") or 
                     text.lower() in ["username", "password", "email", "name", "address", 
                                     "phone", "search", "find"]):
                    
                    label = "input_field"
                    element_type = "ocr_input"
                    score = 0.7
                
                # Menu or navigation items
                elif (text.lower() in ["home", "about", "contact", "menu", "settings", "profile", 
                                      "help", "faq", "blog", "news", "products", "services"]):
                    
                    label = "menu_item"
                    element_type = "ocr_menu"
                    score = 0.75
                
                # Default to text element
                else:
                    label = "text"
                    element_type = "ocr_text"
                    score = 0.5
                
                # Add the OCR detected UI element
                ocr_ui_elements.append({
                    "label": label,
                    "text": text,
                    "score": score,
                    "box": {
                        "xmin": ocr_data["left"][i],
                        "ymin": ocr_data["top"][i],
                        "xmax": ocr_data["left"][i] + ocr_data["width"][i],
                        "ymax": ocr_data["top"][i] + ocr_data["height"][i]
                    },
                    "type": element_type
                })
        
        # 5. Combine all detections into a unified format
        ui_elements = []
        
        # Add general detections
        for detection in general_detections:
            ui_elements.append({
                "label": detection["label"],
                "score": detection["score"],
                "box": detection["box"],
                "type": "general_element"
            })
        
        # Add specialized elements if available
        for element in specialized_elements:
            ui_elements.append(element)
        
        # Add OCR-based UI elements
        for element in ocr_ui_elements:
            ui_elements.append(element)
        
        # Store UI elements and OCR text
        screenshot["ui_elements"] = ui_elements
        screenshot["ocr_text"] = ocr_text
        
        # Also store the structured analysis
        screenshot["ui_structure"] = {
            "general_elements": len(general_detections),
            "specialized_elements": len(specialized_elements),
            "ocr_elements": len(ocr_ui_elements),
            "total_elements": len(ui_elements)
        }
    
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