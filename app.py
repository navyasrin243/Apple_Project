import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms

# --- 1. XAI & SEVERITY ENGINE ---
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
        weights = np.mean(self.gradients.cpu().data.numpy()[0], axis=(1, 2))
        cam = np.maximum(np.dot(weights, self.activations.cpu().data.numpy()[0]), 0)
        heatmap = cv2.resize(cam, (224, 224))
        if heatmap.max() > 0: heatmap /= heatmap.max()

        # Severity Logic
        img_np = np.array(img_224)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        leaf_mask = cv2.inRange(hsv, (5, 30, 30), (95, 255, 255))
        disease_mask = (heatmap > 0.4).astype(np.uint8) * 255
        leaf_pixels = np.sum(leaf_mask > 0)
        disease_pixels = np.sum((disease_mask > 0) & (leaf_mask > 0))
        severity = (disease_pixels / leaf_pixels * 100) if leaf_pixels > 0 else 0
        return heatmap, round(min(severity, 100.0), 2)

# --- 2. AGRI-DATABASE ---
AGRI_DB = {
    "black_rot": {"med": "Mancozeb", "base": 550, "info": "Prune cankers and apply fungicide."},
    "healthy": {"med": "N/A", "base": 0, "info": "Monitor and maintain irrigation."},
    "rust": {"med": "Myclobutanil", "base": 600, "info": "Remove nearby Red Cedar hosts."},
    "scab": {"med": "Captan 80 WDG", "base": 450, "info": "Apply protective fungicide."}
}

# --- 3. UI SETUP ---
st.set_page_config(page_title="AppleAI Pro", layout="wide")
st.sidebar.title("🔬 Research Controls")
mode = st.sidebar.radio("Logic:", ["Focal Loss (Optimized)", "SMOTE (Balanced)"])
m_file = "model_focal.pth" if "Focal" in mode else "model_smote.pth"

@st.cache_resource
def load_sys(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    except:
        st.error(f"⚠️ Model file '{path}' missing! Please run training first.")
    model.eval()
    return model, AppleDiagnostics(model, model.features[-1]), device

try:
    model, engine, device = load_sys(m_file)
    CLASS_NAMES = ['black_rot', 'healthy', 'rust', 'scab']
    st.title("🍎 AppleAI Pathological Assessment")
    
    file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        c1, c2, c3 = st.columns([1, 1, 1])
        
        # Inference
        inf_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        input_inf = inf_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            idx = model(input_inf).argmax(1).item()
            label = CLASS_NAMES[idx]
        
        heatmap, sev = engine.analyze(img, idx)
        db = AGRI_DB[label]
        cost = int(db['base'] * (sev/100 + 1)) if label != "healthy" else 0

        with c1:
            st.image(img, caption="Leaf Sample", use_container_width=True)
            st.metric("Diagnosis", label.upper())
        with c2:
            img_224 = np.array(img.resize((224, 224)))
            h_map = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_224, 0.6, cv2.cvtColor(h_map, cv2.COLOR_BGR2RGB), 0.4, 0)
            st.image(overlay, caption="Grad-CAM Hotspots", use_container_width=True)
        with c3:
            st.subheader("📋 Report")
            st.write(f"**Severity:** {sev}%")
            st.write(f"**Medicine:** {db['med']}")
            st.metric("Est. Cost", f"₹{cost}")
            st.info(f"Tip: {db['info']}")
except Exception as e:
    st.info("System initializing... please ensure model files exist.")
