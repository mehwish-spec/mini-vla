import streamlit as st
import requests
from PIL import Image
import io
import json

st.set_page_config(
    page_title="Mini-VLA Robot Action Predictor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mini-VLA: Open-Vocabulary Robot Action Predictor")
st.markdown("Upload a robot scene image and give a natural language instruction to predict the robot action.")

API_URL = "http://127.0.0.1:8007"

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload robot scene image", type=["jpg", "jpeg", "png"])
    instruction = st.text_input(
        "Natural language instruction",
        placeholder="e.g. pick up the red block"
    )

    examples = [
        "pick up the red block",
        "place the object on the shelf",
        "push the cube forward",
        "open the drawer",
        "close the cabinet door"
    ]
    st.markdown("**Example instructions:**")
    for ex in examples:
        if st.button(ex, key=ex):
            instruction = ex

    predict_btn = st.button("Predict Action", type="primary")

with col2:
    st.header("Prediction")

    if uploaded_file and instruction and predict_btn:
        with st.spinner("Predicting..."):
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")},
                data={"instruction": instruction}
            )

            if response.status_code == 200:
                result = response.json()

                action = result["predicted_action"]
                confidence = result["confidence"]
                latency = result["latency_ms"]

                action_colors = {
                    "PICK": "🟢",
                    "PLACE": "🔵",
                    "PUSH": "🟡",
                    "OPEN": "🟠",
                    "CLOSE": "🔴"
                }

                st.markdown(f"### {action_colors.get(action, '⚪')} Predicted Action: **{action}**")
                st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                st.markdown(f"**Latency:** {latency}ms")

                st.divider()
                st.markdown("**All Action Probabilities:**")

                probs = result["all_probabilities"]
                for act, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    color = action_colors.get(act, "⚪")
                    st.progress(prob, text=f"{color} {act}: {prob*100:.2f}%")

            else:
                st.error("Prediction failed!")

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

st.divider()
st.markdown("**Model:** CLIP ViT-B/32 + Fusion Head + Action Prediction Head")
st.markdown("**Action Classes:** PICK | PLACE | PUSH | OPEN | CLOSE")
st.markdown("**GitHub:** [mehwish-spec/mini-vla](https://github.com/mehwish-spec)")
