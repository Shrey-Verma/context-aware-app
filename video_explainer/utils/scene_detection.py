"""
Scene detection and screenshot capture utilities
"""
import cv2
import os
from tqdm import tqdm
from scenedetect import VideoManager, ContentDetector, SceneManager


def capture_screenshots(video_path, output_dir, scenes, screenshot_interval, fps, frame_count):
    """
    Capture screenshots from the video at scene changes and regular intervals
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save screenshots
        scenes: Scene boundaries
        screenshot_interval: Interval (in seconds) between screenshots
        fps: Frames per second
        frame_count: Total frame count
        
    Returns:
        list: Screenshot information
    """
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Calculate frame indices for screenshots
    screenshot_frames = set()
    
    # Add scene change frames
    for scene in scenes:
        screenshot_frames.add(scene["start_frame"])
    
    # Add regular interval frames
    interval_frames = int(fps * screenshot_interval)
    for i in range(0, frame_count, interval_frames):
        screenshot_frames.add(i)
    
    # Sort frames
    screenshot_frames = sorted(list(screenshot_frames))
    
    # Capture screenshots
    screenshots = []
    for frame_idx in tqdm(screenshot_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            timestamp = frame_idx / fps
            screenshot_path = os.path.join(output_dir, f"screenshot_{frame_idx:06d}.jpg")
            cv2.imwrite(screenshot_path, frame)
            
            screenshots.append({
                "frame": frame_idx,
                "timestamp": timestamp,
                "path": screenshot_path,
                "ui_elements": []
            })
    
    # Release video capture
    cap.release()
    
    return screenshots

def detect_scenes_and_screenshots(video_path, screenshots_dir, fps, frame_count, cap, scene_threshold, screenshot_interval):
    """
    Detect scenes and capture screenshots
    
    Args:
        video_path: Path to the video file
        screenshots_dir: Directory to save screenshots
        fps: Frames per second
        frame_count: Total frame count
        cap: OpenCV video capture object
        scene_threshold: Threshold for scene change detection
        screenshot_interval: Interval (in seconds) between screenshots
        
    Returns:
        tuple: (scenes, screenshots)
    """
    # Initialize scene detection
    video_manager = VideoManager([video_path])
    scene_detector = ContentDetector(threshold=scene_threshold)
    
    # Start video manager
    video_manager.start()
    
    # Create a scene list
    scene_manager = SceneManager()
    scene_manager.add_detector(scene_detector)
    
    # Detect scenes     
    scene_manager.detect_scenes(video_manager)
    scene_list = scene_manager.get_scene_list()
    
    # Extract scene boundaries
    scenes = []
    for scene in scene_list:
        start_frame = scene[0].frame_num
        end_frame = scene[1].frame_num - 1  # End frame is exclusive
        start_time = start_frame / fps
        end_time = end_frame / fps
        scenes.append({
            "start_time": start_time,
            "end_time": end_time,
            "start_frame": start_frame,
            "end_frame": end_frame
        })
    
    # Capture screenshots
    screenshots = capture_screenshots(
        video_path, 
        screenshots_dir, 
        scenes, 
        screenshot_interval, 
        fps, 
        frame_count
    )
    
    return scenes, screenshots