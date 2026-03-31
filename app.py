
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import os

# --- 1. THE DIAGNOSTIC ENGINE ---
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
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_t = tf(img_pil).unsqueeze(0).to(device)
        self.model.zero_grad()
        output = self.model(input_t)
        output[0, label_idx].backward()
        
        # --- ROBUST DIMENSION HANDLING ---
        # Ensure we have (C, H, W) by removing the batch dimension if present
        grads = self.gradients.detach().cpu().numpy().squeeze()
        acts = self.activations.detach().cpu().numpy().squeeze()

        # Calculate Global Average Pooled gradients (Importance Weights)
        # We average across the Spatial dimensions (H, W) which are indices 1 and 2
        weights = np.mean(grads, axis=(1, 2))
        
        # Create Heatmap: Weighted sum of all 1280 activation maps
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]
            
        # ReLU Activation: Only keep features that contribute POSITIVELY to the class
        cam = np.maximum(cam, 0)
        
        # Normalize for visualization
        heatmap = cv2.resize(cam, (224, 224))
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        # ----------------------------------

        img_np = np.array(img_pil.resize((224, 224)))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        # Masking the leaf (Green/Yellow range) to calculate severity
        leaf_mask = cv2.inRange(hsv, (5, 30, 30), (90, 255, 255))
        disease_mask = (heatmap > 0.4).astype(np.uint8) * 255 # 0.4 is a more sensitive threshold
        
        leaf_area = np.sum(leaf_mask > 0)
        disease_area = np.sum(np.bitwise_and(disease_mask > 0, leaf_mask > 0))
        
        severity = (disease_area / leaf_area * 100) if leaf_area > 0 else 0
        return heatmap, round(min(severity, 100.0), 2)

# --- 2. APP UI ---
st.set_page_config(page_title="AppleAI Pro", layout="wide")
st.title("🍎 AppleAI: Precision Field Diagnostic")

@st.cache_resource
def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    
    paths = ["model.pth", "/content/model.pth"]
    model_path = next((p for p in paths if os.path.exists(p)), None)
    
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        engine = AppleDiagnostics(model, model.features[-1])
        return model, engine, device
    return None, None, None

model, engine, device = load_all()

if model is None:
    st.error("🚨 model.pth not found!")
else:
    CLASS_NAMES = ['black_rot', 'healthy', 'rust', 'scab']
    TREAT_DATA = {
        "scab": {"med": "Captan 80 WDG", "price": 450},
        "rust": {"med": "Myclobutanil", "price": 600},
        "black_rot": {"med": "Mancozeb", "price": 550},
        "healthy": {"med": "None", "price": 0}
    }

    uploaded_file = st.file_uploader("Upload Leaf Photo to Start Analysis", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        
        # 1. Prediction
        tf_inf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        input_inf = tf_inf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input_inf)
            idx = torch.max(out, 1)[1].item()
            label = CLASS_NAMES[idx]
        
        # 2. Diagnostic Analysis
        heatmap, severity = engine.analyze(img, idx)
        
        # 3. Final Display
        st.success(f"Analysis Complete: **{label.upper()}** Detected")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
            st.metric("Infection Severity", f"{severity}%")
            
        with col2:
            img_np = np.array(img.resize((224, 224)))
            heatmap_c = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB), 0.4, 0)
            st.image(overlay, caption="Infection Hotspots (Grad-CAM)", use_container_width=True)
            
            if label != "healthy":
                cost = int(TREAT_DATA[label]["price"] * (severity / 100))
                st.warning(f"**Treatment:** {TREAT_DATA[label]['med']}")
                st.info(f"**Precision Cost Estimate:** ₹{max(cost, 50)}")
            else:
                st.balloons()
                st.write("Crop is in optimal health!")
