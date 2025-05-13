"""
Visualization utilities for the VideoExplainer
"""
import os
import matplotlib.pyplot as plt

def create_timeline_visualization(transcript_segments, screenshots, scenes, output_path):
    """
    Create a timeline visualization
    
    Args:
        transcript_segments: List of transcript segments
        screenshots: List of screenshots
        scenes: List of scenes
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(15, 8))
    
    # Plot transcript segments
    for i, segment in enumerate(transcript_segments):
        plt.barh(1, segment["end"] - segment["start"], left=segment["start"], height=0.5, 
                 color='lightblue', alpha=0.7)
        
        # Add segment text
        if i % 5 == 0:  # Show every 5th segment text to avoid clutter
            plt.text(segment["start"], 1.25, f"Segment {i}", fontsize=8)
    
    # Plot screenshots
    for i, screenshot in enumerate(screenshots):
        screenshot["path"] = os.path.join("../", screenshot["path"])
        plt.scatter(screenshot["timestamp"], 2, color='red', s=50)
        
        # Add screenshot filename
        if i % 3 == 0:  # Show every 3rd screenshot name to avoid clutter
            plt.text(screenshot["timestamp"], 2.1, os.path.basename(screenshot["path"]), 
                     fontsize=8, rotation=90)
            
        print(f"Screenshot {i}: {screenshot['path']} with path {screenshot['path']}")
    
    # Plot scenes
    for i, scene in enumerate(scenes):
        plt.barh(3, scene["end_time"] - scene["start_time"], left=scene["start_time"], 
                 height=0.5, color='green', alpha=0.7)
        plt.text(scene["start_time"], 3.25, f"Scene {i}", fontsize=8)
    
    # Add labels
    plt.yticks([1, 2, 3], ["Transcript", "Screenshots", "Scenes"])
    plt.xlabel("Time (seconds)")
    plt.title("Video Analysis Timeline")
    
    # Save the visualization
    plt.savefig(output_path)
    plt.close()