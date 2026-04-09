import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import models, transforms

# --- 1. IMPROVED XAI ENGINE ---
class AppleDiagnostics:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o))
        self.target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0]))

    def analyze(self, img_pil, label_idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_224 = img_pil.resize((224, 224))
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_t = tf(img_224).unsqueeze(0).to(device)
        
        self.model.zero_grad()
        output = self.model(input_t)
        output[0, label_idx].backward()
        
        grads = self.gradients.cpu().data.numpy()[0]
        act = self.activations.cpu().data.numpy()[0]
        
        # IMPROVEMENT: Use 'Positive-Only' weights to ignore noise
        weights = np.maximum(np.mean(grads, axis=(1, 2)), 0)
        
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i, :, :]
            
        cam = np.maximum(cam, 0)
        heatmap = cv2.resize(cam, (224, 224))
        if heatmap.max() > 0: heatmap /= heatmap.max()

        # Severity Logic
        img_np = np.array(img_224)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        leaf_mask = cv2.inRange(hsv, (5, 30, 30), (95, 255, 255))
        # Refined threshold for disease detection
        disease_mask = (heatmap > 0.5).astype(np.uint8) * 255
        leaf_pixels = np.sum(leaf_mask > 0)
        disease_pixels = np.sum((disease_mask > 0) & (leaf_mask > 0))
        severity = (disease_pixels / leaf_pixels * 100) if leaf_pixels > 0 else 0
        
        return heatmap, round(min(severity, 100.0), 2)

# --- 2. AGRI-DATABASE ---
AGRI_DB = {
    "black_rot": {"med": "Mancozeb", "base": 550, "info": "Prune cankers immediately."},
    "healthy": {"med": "N/A", "base": 0, "info": "Standard maintenance."},
    "rust": {"med": "Myclobutanil", "base": 600, "info": "Check nearby Cedar trees."},
    "scab": {"med": "Captan 80 WDG", "base": 450, "info": "Apply pre-bloom fungicide."}
}

# --- 3. UI SETUP ---
st.set_page_config(page_title="AppleAI Pro", layout="wide")
st.sidebar.title("🔬 Research Methodology")
mode = st.sidebar.radio("Optimization Strategy:", ["Focal Loss (Weighted)", "SMOTE (Oversampled)"])

m_file = "model_focal.pth" if "Focal" in mode else "model_smote.pth"

@st.cache_resource
def load_sys(path):
    if not os.listdir('.') or path not in os.listdir('.'):
        return None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model, AppleDiagnostics(model, model.features[-1]), device

model, engine, device = load_sys(m_file)

if model is None:
    st.error(f"⚠️ Model `{m_file}` is missing from GitHub.")
    st.stop()

CLASS_NAMES = ['black_rot', 'healthy', 'rust', 'scab']
st.title("🍎 AppleAI: Pathological Assessment")

file = st.file_uploader("Scan Leaf...", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    c1, c2, c3 = st.columns([1, 1, 1])
    
    # Inference with Probability Calibration
    inf_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    input_inf = inf_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_inf)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)
        label = CLASS_NAMES[idx.item()]

    heatmap, sev = engine.analyze(img, idx.item())
    db = AGRI_DB[label]
    cost = int(db['base'] * (sev/100 + 1)) if label != "healthy" else 0

    with c1:
        st.image(img, caption="Original", use_container_width=True)
        st.metric("Detection", label.upper(), f"{conf.item()*100:.1f}% Match")
    with c2:
        img_224 = np.array(img.resize((224, 224)))
        h_map = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_224, 0.6, cv2.cvtColor(h_map, cv2.COLOR_BGR2RGB), 0.4, 0)
        st.image(overlay, caption="XAI Feature Localization", use_container_width=True)
    with c3:
        st.subheader("📊 Economic Report")
        st.write(f"**Severity:** {sev}%")
        st.write(f"**Treatment:** {db['med']}")
        st.metric("Est. Cost", f"₹{cost}")
        st.info(f"Insight: {db['info']}")
