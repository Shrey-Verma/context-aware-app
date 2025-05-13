#!/usr/bin/env python3
import argparse
from video_explainer.explainer import VideoExplainer

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
    
    # Use the API key from command line or the hardcoded one
    api_key = args.api_key
    
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