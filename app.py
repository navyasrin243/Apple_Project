
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F

# -----------------------------
# Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["black_rot", "healthy", "rust", "scab"]

# -----------------------------
# Load Model
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
# Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -----------------------------
# Grad-CAM
# -----------------------------
features = []
gradients = []

def forward_hook(m, i, o):
    features.append(o)

def backward_hook(m, gi, go):
    gradients.append(go[0])

model.features[-1].register_forward_hook(forward_hook)
model.features[-1].register_full_backward_hook(backward_hook)

def generate_cam(img_tensor, class_idx):
    features.clear()
    gradients.clear()

    out = model(img_tensor)
    model.zero_grad()
    out[0, class_idx].backward()

    grads = gradients[0].cpu().numpy()[0]
    fmap = features[0].cpu().numpy()[0]

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
# Logic
# -----------------------------
def calculate_severity(cam, confidence):
    threshold = np.mean(cam) + np.std(cam)
    lesion = np.sum(cam > threshold)
    severity = (lesion / cam.size) * 100
    return severity * confidence

def get_recommendation(label):
    return {
        "scab": "Use Mancozeb spray, avoid leaf wetness",
        "rust": "Apply sulfur fungicide, reduce humidity",
        "black_rot": "Prune infected areas immediately",
        "healthy": "Healthy leaf - no action required"
    }[label]

# -----------------------------
# UI
# -----------------------------
st.title("🍎 Apple Leaf Disease Diagnosis System")

file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    confidence = conf.item()

    cam = generate_cam(x, pred.item())
    severity = calculate_severity(cam, confidence)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Disease", label.replace("_"," ").title())
        st.metric("Confidence", f"{confidence*100:.2f}%")

    with col2:
        st.metric("Severity", f"{severity:.2f}%")
        st.metric("Risk", "High" if severity > 50 else "Moderate" if severity > 20 else "Low")

    st.subheader("💡 Recommendation")
    st.success(get_recommendation(label))

    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    st.subheader("🔥 Grad-CAM Heatmap")
    st.image(heatmap, caption="Model Attention")