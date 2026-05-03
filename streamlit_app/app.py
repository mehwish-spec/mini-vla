import streamlit as st
import requests
from PIL import Image
import io
import torch
import numpy as np
import sys
import os
sys.path.insert(0, "/Users/mehwishrahman/Desktop/project-1/mini-vla")
from app.model import VLAModel, ACTION_CLASSES
from app.attention import get_attention_map, overlay_attention
from torchvision import transforms

st.set_page_config(
    page_title="Mini-VLA Robot Action Predictor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mini-VLA: Open-Vocabulary Robot Action Predictor")
st.markdown("*Vision-Language-Action model using CLIP + Fusion Head — predicts robot manipulation actions from image + instruction*")

@st.cache_resource
def load_model():
    model = VLAModel(num_classes=5)
    if os.path.exists('models/checkpoints/best_model.pt'):
        model.load_state_dict(torch.load('models/checkpoints/best_model.pt', map_location='cpu'))
    model.eval()
    return model

model = load_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload robot scene image", type=["jpg", "jpeg", "png"])

    instruction = st.text_input(
        "Natural language instruction",
        placeholder="e.g. pick up the red block"
    )

    st.markdown("**Quick examples:**")
    examples = [
        "pick up the red block",
        "place the object on the shelf",
        "push the cube forward",
        "open the drawer",
        "close the cabinet door"
    ]

    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=ex, use_container_width=True):
            instruction = ex
            st.session_state.instruction = ex

    if "instruction" in st.session_state:
        instruction = st.session_state.instruction

    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    show_attention = st.checkbox("Show attention heatmap", value=True)
    predict_btn = st.button("🔍 Predict Action", type="primary", use_container_width=True)

with col2:
    st.header("Results")

    if uploaded_file and instruction and predict_btn:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Running VLA inference..."):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            img_tensor = transform(image)

            with torch.no_grad():
                logits = model([img_tensor], [instruction])
                probs = torch.softmax(logits, dim=-1)[0]
                pred_idx = probs.argmax().item()
                confidence = probs[pred_idx].item()
                action = ACTION_CLASSES[pred_idx]

            action_emojis = {
                "PICK": "🟢 PICK",
                "PLACE": "🔵 PLACE",
                "PUSH": "🟡 PUSH",
                "OPEN": "🟠 OPEN",
                "CLOSE": "🔴 CLOSE"
            }

            if confidence >= confidence_threshold:
                st.success(f"### Predicted: {action_emojis[action]}")
            else:
                st.warning(f"### Low confidence: {action_emojis[action]}")

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Confidence", f"{confidence*100:.1f}%")
            col_m2.metric("Threshold", f"{confidence_threshold*100:.0f}%")

            st.markdown("**Action Probabilities:**")
            for i, act in enumerate(ACTION_CLASSES):
                prob = probs[i].item()
                emoji = action_emojis[act].split()[0]
                st.progress(prob, text=f"{emoji} {act}: {prob*100:.1f}%")

            st.divider()

            if show_attention:
                st.markdown("**Attention Heatmap:**")
                attn_map = get_attention_map(model, image, instruction)
                if attn_map is not None:
                    overlay = overlay_attention(image, attn_map)
                    img_col1, img_col2 = st.columns(2)
                    img_col1.image(image.resize((224, 224)), caption="Original", use_container_width=True)
                    img_col2.image(overlay, caption="Attention", use_container_width=True)
                else:
                    st.image(image, caption="Input Image", use_container_width=True)
            else:
                st.image(image, caption="Input Image", use_container_width=True)

    elif not uploaded_file:
        st.info("Upload an image to get started!")
    elif not instruction:
        st.info("Enter an instruction!")

st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
col_f1.markdown("**Model:** CLIP ViT-B/32 + Fusion")
col_f2.markdown("**Params:** 689K trainable")
col_f3.markdown("**Classes:** PICK/PLACE/PUSH/OPEN/CLOSE")
