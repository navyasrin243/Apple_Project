import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Apple Disease Detector", layout="centered")

# ---------------- STYLING ----------------
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .title {text-align: center; font-size: 36px; font-weight: bold;}
    .subtitle {text-align: center; color: gray; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🍎 Apple Leaf Disease Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered disease detection with recommendations</div>', unsafe_allow_html=True)

# ---------------- CLASS NAMES ----------------
class_names = ["black_rot", "healthy", "rust", "scab"]

# ---------------- DISEASE INFO ----------------
disease_info = {
    "scab": "Fungal disease causing dark lesions on leaves and fruit.",
    "rust": "Causes yellow-orange spots on leaves.",
    "black_rot": "Leads to rotting and dark patches.",
    "healthy": "Leaf is healthy with no visible disease."
}

# ---------------- EXPLAINABLE AI ----------------
explain = {
    "scab": "Model detected irregular dark lesion patterns.",
    "rust": "Model focused on yellow-orange clustered spots.",
    "black_rot": "Model identified dark necrotic regions.",
    "healthy": "No abnormal patterns detected."
}

# ---------------- TREATMENT ----------------
treatment = {
    "scab": "Use fungicides like captan or myclobutanil. Remove infected leaves.",
    "rust": "Apply sulfur sprays. Avoid nearby juniper plants.",
    "black_rot": "Prune infected areas and apply fungicide regularly.",
    "healthy": "No treatment needed. Maintain regular care."
}

# ---------------- DEVICE ----------------
device = torch.device("cpu")

# ---------------- MODEL ----------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 4)

model_path = os.path.join(os.path.dirname(__file__), "model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------------- TRANSFORMS ----------------
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ---------------- FUNCTIONS ----------------
def predict(img):
    x = val_tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs,1)
    return class_names[pred.item()], conf.item(), probs.numpy()[0]

def get_severity(conf):
    if conf < 0.5:
        return "Low"
    elif conf < 0.75:
        return "Moderate"
    else:
        return "High"

# ---------------- INFO ----------------
st.info("Upload a clear apple leaf image for best results.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        img.verify()

        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)

        with col2:
            with st.spinner("🔍 Analyzing..."):
                label, conf, probs = predict(img)

                # Confidence + severity
                severity = get_severity(conf)

                st.markdown(f"### 🧠 Prediction: **{label.upper()}**")
                st.progress(int(conf * 100))
                st.write(f"Confidence: {conf:.2f}")
                st.write(f"🔥 Severity: {severity}")

                # Confidence interpretation
                if conf < 0.65:
                    st.warning("⚠️ Low confidence prediction — model unsure")
                elif conf < 0.85:
                    st.info("ℹ️ Moderate confidence")
                else:
                    st.success("✅ High confidence prediction")

                # Disease info
                if label in disease_info:
                    st.info(disease_info[label])

                # Explainable AI
                st.markdown("### 🧠 Why this prediction?")
                st.info(explain[label])

                # Treatment
                if label in treatment:
                    st.markdown("### 💊 Treatment Recommendation")
                    st.success(treatment[label])

                # Probabilities
                st.markdown("### 📊 Class Probabilities:")
                for i, cls in enumerate(class_names):
                    st.write(f"{cls}: {probs[i]:.2f}")

    except Exception:
        st.error("❌ Invalid or corrupted image. Please upload a proper leaf image.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ using Deep Learning")
