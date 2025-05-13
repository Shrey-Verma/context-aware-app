#!/usr/bin/env python3
import argparse
from video_explainer.explainer import VideoExplainer
import os

def main():
    parser = argparse.ArgumentParser(description="Video Explainer: Analyze videos and answer questions")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--question", required=True, help="Question about the video")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--interval", type=float, default=2.0, help="Screenshot interval in seconds")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI for answer generation")  # Changed from use-mistral
    parser.add_argument("--api-key", help="OpenAI API key for answer generation")  # Changed from Mistral
    parser.add_argument("--visualize", action="store_true", help="Create visualization of the analysis")
    parser.add_argument("--report", action="store_true", help="Create a comprehensive report")
    parser.add_argument("--annotate-ui", action="store_true", help="Annotate detected UI elements on screenshots")
    
    args = parser.parse_args()
    
    # If your API key is hardcoded, you can set it here
    # This is useful for debugging but not recommended for production
   
    # Use the API key from command line or the hardcoded one
    api_key = args.api_key  # Changed from MISTRAL_API_KEY
    
    # FORCE OpenAI to be TRUE for debugging purposes  # Changed from MISTRAL
    # In production, you'd use args.use_openai instead
    use_openai = True  # Set this to True to force using OpenAI  # Changed from use_mistral
    
    # Validate that API key is provided when use-openai is True  # Changed from use-mistral
    if use_openai and not api_key:  # Changed from use_mistral
        print("Error: API key is required when use_openai=True")  # Changed from use_mistral
        return
    
    try:
        print(f"Creating VideoExplainer with use_openai={use_openai}, API key provided: {bool(api_key)}")  # Changed from use_mistral
        
        # Create VideoExplainer
        explainer = VideoExplainer(
            video_path=args.video,
            question=args.question,
            output_dir=args.output,
            screenshot_interval=args.interval,
            use_openai=use_openai,  # Use our forced value instead of args.use_openai  # Changed from use_mistral
            openai_api_key=api_key,
            annotate_ui=args.annotate_ui  # Changed from mistral_api_ke  # Pass the UI annotation flag
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