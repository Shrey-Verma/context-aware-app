# VideoExplainer

VideoExplainer is a tool for analyzing and answering questions about video content using advanced AI techniques. It extracts transcripts, detects scenes, captures screenshots, identifies UI elements, and generates comprehensive answers about the video content.

## Features

- Video transcript extraction using Whisper
- Scene change detection
- Automatic screenshot capture at key points
- UI element detection in screenshots
- Text recognition (OCR) in video frames
- Timeline visualization of video segments
- Answer generation using Mistral AI
- Comprehensive report creation

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/video-explainer.git
   cd video-explainer
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Additional requirements:
   - Tesseract OCR: Install from https://github.com/tesseract-ocr/tesseract
   - FFmpeg: Required for audio processing

## Usage

### Basic Usage

To analyze a video and answer a question:

```bash
python main.py --video path/to/your/video.mp4 --question "What happens in this video?" --api-key your_mistral_api_key
```

### Command Line Arguments

- `--video`: Path to the video file (required)
- `--question`: Question about the video content (required)
- `--output`: Output directory (default: "output")
- `--interval`: Screenshot interval in seconds (default: 2.0)
- `--use-mistral`: Use Mistral AI for answer generation
- `--api-key`: Mistral API key
- `--visualize`: Create visualization of the analysis
- `--report`: Create a comprehensive report

### Example

```bash
python main.py --video demo_video.mp4 --question "What is the main topic of this presentation?" --output analysis_results --interval 1.5 --use-mistral --api-key your_mistral_api_key
```

## Output

The tool generates the following outputs:

- Screenshots at regular intervals and scene changes
- Transcript of the video content
- UI element detection results
- Answer to the specified question (markdown format)
- Timeline visualization (if requested)
- Comprehensive report (if requested)

## Project Structure

```
video_explainer/
│
├── main.py                     # Entry point script
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
│
├── video_explainer/            # Main package
│   ├── __init__.py             # Package initialization
│   ├── explainer.py            # VideoExplainer class
│   ├── utils/                  # Utility modules
│   │   ├── __init__.py
│   │   ├── transcript.py       # Transcript extraction and processing
│   │   ├── scene_detection.py  # Scene detection functionality
│   │   ├── ui_detection.py     # UI elements detection
│   │   └── visualization.py    # Visualization utilities
│   │
│   └── output/                 # Default output directory
│       └── .gitkeep            # Placeholder to commit the directory
│
└── tests/                      # Tests (if needed in the future)
    └── __init__.py
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.