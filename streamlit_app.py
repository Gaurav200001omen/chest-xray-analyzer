import streamlit as st
import pandas as pd
from PIL import Image
import torch

from src.model import ChestXRayResNetSE
from src.dataset import preprocess_pil_for_inference
from src.gradcam import GradCAM, overlay_heatmap

st.set_page_config(page_title="Chest X-Ray AI Analyzer", layout="wide")


@st.cache_resource
def load_model(ckpt_path: str = "outputs/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]

    model = ChestXRayResNetSE(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, classes, img_size, device


st.title("Chest X-Ray Disease Analyzer")
st.warning(
    "Academic & Research Prototype Only: This model is trained from scratch for educational "
    "purposes and is not a certified medical diagnostic tool."
)

with st.sidebar:
    st.header("Model Info")
    st.markdown(
        "- **Architecture:** Custom ResNet-SE (from scratch)\n"
        "- **Preprocessing:** Grayscale + CLAHE\n"
        "- **Explainability:** Grad-CAM (Stage 4)"
    )
    alpha = st.slider("Grad-CAM Heatmap Opacity", min_value=0.1, max_value=0.8, value=0.4, step=0.05)

try:
    model, classes, img_size, device = load_model()
except FileNotFoundError:
    st.error("Trained model not found at `outputs/best_model.pth`. Run `python src/train.py` first.")
    st.stop()

uploaded = st.file_uploader("Upload a Chest X-Ray (PNG / JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    pil_img = Image.open(uploaded)
    input_tensor, enhanced_gray = preprocess_pil_for_inference(pil_img, img_size=img_size)
    input_tensor = input_tensor.to(device)

    gradcam = GradCAM(model, target_layer=model.stage4[-1])
    cam, probs, pred_idx = gradcam.generate(input_tensor)
    overlay_rgb = overlay_heatmap(enhanced_gray, cam, alpha=alpha)

    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0

    col1, col2, col3 = st.columns([1, 1, 1.2])
    with col1:
        st.subheader("CLAHE-Enhanced X-Ray")
        st.image(enhanced_gray, width="stretch", clamp=True)
    with col2:
        st.subheader("Grad-CAM Focus Map")
        st.image(overlay_rgb, width="stretch")
    with col3:
        st.subheader("Prediction Result")
        st.metric(label="Top Prediction", value=pred_class, delta=f"{confidence:.2f}% confidence")
        prob_df = pd.DataFrame({
            "Condition": classes,
            "Probability (%)": [round(float(p) * 100, 2) for p in probs],
        }).set_index("Condition")
        st.bar_chart(prob_df)