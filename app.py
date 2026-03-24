import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
from io import BytesIO

# -----------------------------
# 1. Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["black_rot", "healthy", "rust", "scab"]

# -----------------------------
# 2. Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)

    checkpoint = torch.load("model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------------
# 3. Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -----------------------------
# 4. Grad-CAM Setup
# -----------------------------
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# Register hooks once
model.features[-1].register_forward_hook(forward_hook)
model.features[-1].register_full_backward_hook(backward_hook)

def generate_cam(img_tensor, class_idx):
    features.clear()
    gradients.clear()

    output = model(img_tensor)

    model.zero_grad()
    output[0, class_idx].backward()

    # Safety check
    if len(features) == 0 or len(gradients) == 0:
        return np.zeros((224,224))

    grads = gradients[0].detach().cpu().numpy()[0]
    fmap = features[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam

# -----------------------------
# 5. Severity
# -----------------------------
def calculate_severity(cam, confidence):
    threshold = np.mean(cam) + np.std(cam)
    lesion = np.sum(cam > threshold)
    severity = (lesion / cam.size) * 100
    return severity * confidence

# -----------------------------
# 6. Recommendation
# -----------------------------
def get_recommendation(label):
    mapping = {
        "scab": "Use Mancozeb spray and avoid leaf wetness.",
        "rust": "Apply sulfur fungicide and reduce humidity.",
        "black_rot": "Prune infected areas immediately.",
        "healthy": "Leaf is healthy. No action needed."
    }
    return mapping[label]

# -----------------------------
# 7. UI
# -----------------------------
st.title("🍎 Apple Leaf Disease Diagnosis System")
st.write("Upload a leaf image to detect disease, severity, and treatment.")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file is not None:
    try:
        img = Image.open(BytesIO(file.read())).convert("RGB")
    except:
        st.error("❌ Invalid image file. Please upload a proper image.")
        st.stop()

    st.image(img, caption="Uploaded Image", width=400)

    # Prediction
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    confidence = conf.item()

    # Low confidence warning
    if confidence < 0.6:
        st.warning("⚠️ Low confidence prediction. Try a clearer image.")

    # Grad-CAM
    cam = generate_cam(x, pred.item())

    # Severity
    severity = calculate_severity(cam, confidence)

    # Recommendation
    rec = get_recommendation(label)

    # ---------------- OUTPUT ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Disease", label.replace("_"," ").title())
        st.metric("Confidence", f"{confidence*100:.2f}%")

    with col2:
        st.metric("Severity", f"{severity:.2f}%")
        risk = "High" if severity > 50 else "Moderate" if severity > 20 else "Low"
        st.metric("Risk", risk)

    st.subheader("💡 Recommendation")
    st.success(rec)

    # Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    st.subheader("🔥 Grad-CAM Heatmap")
    st.image(heatmap, caption="Model Attention")
