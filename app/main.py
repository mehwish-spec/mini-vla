from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
import io
import time
import os
from app.model import VLAModel, ACTION_CLASSES

app = FastAPI(title="Mini-VLA Robot Action Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

def load_model():
    global model
    model = VLAModel(num_classes=5)
    checkpoint_path = "models/checkpoints/best_model.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print("Checkpoint loaded!")
    model.eval()
    print("Model ready!")

@app.on_event("startup")
async def startup():
    print("Loading VLA model...")
    load_model()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    instruction: str = Form(...)
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    t0 = time.perf_counter()
    action, confidence = model.predict(image, instruction)
    latency = (time.perf_counter() - t0) * 1000

    all_probs = []
    with torch.no_grad():
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(image)
        logits = model([img_tensor], [instruction])
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        all_probs = {ACTION_CLASSES[i]: round(probs[i], 4) for i in range(len(ACTION_CLASSES))}

    return {
        "predicted_action": action,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "instruction": instruction,
        "latency_ms": round(latency, 2)
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "action_classes": ACTION_CLASSES
    }

@app.get("/")
async def root():
    return {
        "message": "Mini-VLA Robot Action Predictor",
        "docs": "/docs",
        "endpoints": ["/predict", "/health"]
    }
