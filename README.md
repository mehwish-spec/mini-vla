# Mini-VLA: Open-Vocabulary Robot Action Predictor

## Overview
A Vision-Language-Action (VLA) pipeline that fuses CLIP vision encoder with natural language instructions to predict robot manipulation actions. Built with PyTorch, HuggingFace, FastAPI and Streamlit.

## Demo
Upload a robot scene image + type a natural language instruction → get predicted robot action with confidence scores in <350ms

## Action Classes
PICK | PLACE | PUSH | OPEN | CLOSE

## Model Architecture
- CLIP ViT-B/32 vision encoder (frozen — 151M params)
- Language encoder from CLIP (frozen)
- Cross-modal fusion layer (trainable)
- Action prediction head (trainable)
- Total trainable parameters: 689,669

## Results
- 100% F1-score on synthetic dataset
- <350ms inference latency
- 5 action classes

## Tech Stack
Python, PyTorch, CLIP, HuggingFace, FastAPI, MLflow, Docker, Streamlit

## Project Structure
mini-vla/
├── app/
│   ├── model.py         # VLA model architecture
│   ├── train.py         # Training pipeline + MLflow
│   └── main.py          # FastAPI inference server
├── streamlit_app/
│   └── app.py           # Streamlit UI
├── data/
│   └── dataset.py       # Data pipeline
├── models/
│   └── checkpoints/     # Saved model weights
└── requirements.txt

## Quick Start
```bash
pip install -r requirements.txt
python -m app.train
uvicorn app.main:app --port 8007
streamlit run streamlit_app/app.py
```

## Example
```bash
curl -X POST http://localhost:8007/predict \
  -F "file=@robot_scene.jpg" \
  -F "instruction=pick up the red block"

# Response
{
  "predicted_action": "PICK",
  "confidence": 0.9997,
  "latency_ms": 345.25
}
```
