import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import models, transforms

# --- 1. XAI & SEVERITY ENGINE ---
class AppleDiagnostics:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Hooks for capturing internal model behavior
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, model, input, output):
        self.activations = output

    def save_gradient(self, model, grad_input, grad_output):
        self.gradients = grad_output[0]

    def analyze(self, img_pil, label_idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        img_224 = img_pil.resize((224, 224))
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_t = tf(img_224).unsqueeze(0).to(device)
        
        self.model.zero_grad()
        output = self.model(input_t)
        output[0, label_idx].backward()
        
        # Grad-CAM Calculation
        grads = self.gradients.detach().cpu().numpy()[0]
        act = self.activations.detach().cpu().numpy()[0]
        weights = np.maximum(np.mean(grads, axis=(1, 2)), 0)
        
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i, :, :]
            
        cam = np.maximum(cam, 0)
        heatmap = cv2.resize(cam, (224, 224))
        if heatmap.max() > 0: heatmap /= heatmap.max()

        # Severity Logic (Visual Analysis)
        img_np = np.array(img_224)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        # Masking leaf tissue to calculate percentage of infection
        leaf_mask = cv2.inRange(hsv, (5, 30, 30), (95, 255, 255))
        disease_mask = (heatmap > 0.4).astype(np.uint8) * 255
        leaf_pixels = np.sum(leaf_mask > 0)
        disease_pixels = np.sum((disease_mask > 0) & (leaf_mask > 0))
        severity = (disease_pixels / leaf_pixels * 100) if leaf_pixels > 0 else 0
        
        return heatmap, round(min(severity, 100.0), 2)

# --- 2. KNOWLEDGE BASE ---
AGRI_DB = {
    "black_rot": {"med": "Mancozeb", "base": 550, "info": "Prune infected branches immediately. Apply copper-based fungicide."},
    "healthy": {"med": "N/A", "base": 0, "info": "No pathology detected. Maintain current irrigation levels."},
    "rust": {"med": "Myclobutanil", "base": 600, "info": "Fungal spores detected. Check for nearby Cedar/Juniper host trees."},
    "scab": {"med": "Captan 80 WDG", "base": 450, "info": "Apple scab thrives in humidity. Increase airflow around the canopy."}
}

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="AppleAI Pro", page_icon="🍎", layout="wide")
st.title("🍎 AppleAI: Automated Pathological Assessment")
st.markdown("🔍 *Integrated Computer Vision & Economic Estimator for Precision Agriculture*")
st.divider()

@st.cache_resource
def load_assets(path):
    if not os.path.exists(path): return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, 4)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m, AppleDiagnostics(m, m.features[-1])

model, engine = load_assets("model_smote.pth")

if model is None:
    st.error("⚠️ Model file 'model_smote.pth' not found in root directory.")
    st.stop()

# --- 4. EXECUTION ---
file = st.file_uploader("Upload Leaf Specimen", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    
    # Pre-processing & Prediction
    tf = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_t = tf(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_t)
        idx = torch.argmax(logits, 1).item()
        conf = torch.softmax(logits, 1)[0][idx].item()
    
    labels = ['black_rot', 'healthy', 'rust', 'scab']
    current_label = labels[idx]
    heatmap, severity = engine.analyze(img, idx)
    db = AGRI_DB[current_label]
    cost = int(db['base'] * (severity/100 + 1)) if current_label != "healthy" else 0

    # UI COLUMNS
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("📸 Original Image")
        st.image(img, use_container_width=True)
        st.metric("Detection", current_label.upper())
        st.progress(conf, text=f"Confidence: {conf*100:.1f}%")

    with col2:
        st.subheader("🔬 Grad-CAM Heatmap")
        img_np = np.array(img.resize((224, 224)))
        h_map_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(h_map_color, cv2.COLOR_BGR2RGB), 0.4, 0)
        st.image(overlay, use_container_width=True, caption="Lesion Localization")
        st.write(f"**Severity Index:** {severity}%")

    with col3:
        st.subheader("📋 Analysis Summary")
        st.metric("Estimated Cost", f"₹{cost}")
        st.write(f"**Recommended Medicine:** {db['med']}")
        st.info(f"**Agri-Expert Note:** {db['info']}")
        
        # Summary Box
        st.success(f"Final Assessment: {current_label.replace('_', ' ').title()} detected with {conf*100:.1f}% confidence. Recommended treatment with {db['med']} is expected to cost approximately ₹{cost}.")
