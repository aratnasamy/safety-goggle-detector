# Safety Goggle Detector

## Project Description

This project is a **Safety Goggle Detection System** built with **Computer Vision**.  
It detects faces and checks if each person is wearing **safety goggles** using face detection and a pre-trained YOLOv8 model.

---

## Prerequisites
Install uv - https://docs.astral.sh/uv/getting-started/installation/

## How to Run

1. Install dependencies using uv:
   ```bash
   uv sync
   ```

2. Run the app:
   ```bash
   uv run main.py
   ```

3. The camera will open and start detecting safety goggles. Allow access to open the camera if needed. Focus the camera window and hit "q" to exit.

## Run Model tester
Use the model tester to test YOLO models on images
```bash
uv run model_tester.py --image PATH_TO_IMAGE --model PATH_TO_MODEL
```