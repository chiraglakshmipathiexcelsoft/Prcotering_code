# Vision System - Windows GPU Conversion

This project has been converted from Raspberry Pi (Hailo-10H) to Windows (NVIDIA CUDA).

## Key Changes
- **Hardware**: Switched from Raspberry Pi camera and Hailo-10H chip to standard Webcam and NVIDIA GPU (RTX PRO 1000).
- **AI Engine**: Replaced Hailo post-processing with direct **YOLOv8 CUDA inference** and **InsightFace CUDA providers**.
- **Performance**: Expect significantly higher FPS (30+) and lower latency compared to the Raspberry Pi.

## Requirements
Ensure you have the following installed in your environment:
```bash
pip install ultralytics insightface onnxruntime-gpu flask opencv-python numpy
```

*Note: Since you already have PyTorch with CUDA support, the above will leverage your GPU automatically.*

## How to Run
1. Open a terminal in this directory.
2. Run the application:
   ```bash
   python app_windows.py
   ```
3. Open your browser to: `http://localhost:5000`

## Features
- **GPU Acceleration**: Both Detection and Pose estimation run on the RTX GPU.
- **Behavior Monitoring**: Real-time head turn, movement, and hand signal detection with Windows-native beeps and popups.
- **Face Training**: Capture and train faces locally; embeddings are saved to `face_embeddings.pkl`.
- **Export**: Export detection bursts and CSV logs directly from the UI.
