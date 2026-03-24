import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

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
st.markdown('<div class="subtitle">Upload a leaf image to detect disease</div>', unsafe_allow_html=True)

# ---------------- MODEL ----------------
class_names = ["black_rot", "healthy", "rust", "scab"]
device = torch.device("cpu")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 4)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ---------------- TRANSFORMS ----------------
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ---------------- PREDICTION ----------------
def predict(img):
    x = val_tf(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs,1)

    return class_names[pred.item()], conf.item(), probs.numpy()[0]

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🔍 Analyzing..."):
            try:
                label, conf, probs = predict(img)

                # Confidence threshold
                if conf < 0.65:
                    st.warning("⚠️ Low confidence prediction")
                else:
                    st.success("✅ Prediction complete")

                st.markdown(f"### 🧠 Prediction: **{label.upper()}**")
                st.progress(int(conf * 100))
                st.write(f"Confidence: {conf:.2f}")

                st.markdown("### 📊 Class Probabilities:")
                for i, cls in enumerate(class_names):
                    st.write(f"{cls}: {probs[i]:.2f}")

            except:
                st.error("❌ Invalid image. Try another.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ using Deep Learning")
