
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import os
import pandas as pd

# --- 1. THE DIAGNOSTIC ENGINE (XAI + Multimodal) ---
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
        
        # Extract and Process Grad-CAM
        grads = self.gradients.detach().cpu().numpy().squeeze()
        acts = self.activations.detach().cpu().numpy().squeeze()
        weights = np.mean(grads, axis=(1, 2))
        
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]
            
        cam = np.maximum(cam, 0)
        heatmap = cv2.resize(cam, (224, 224))
        
        # --- SPATIAL FILTERING: Remove Background/Stem Noise ---
        h, w = heatmap.shape
        mask = np.zeros((h, w))
        mask[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)] = 1
        heatmap = heatmap * mask
        
        if heatmap.max() > 0:
            heatmap /= (heatmap.max() + 1e-8)

        # SEVERITY & MULTIMODAL OVERRIDE
        img_np = np.array(img_pil.resize((224, 224)))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        leaf_mask = cv2.inRange(hsv, (5, 30, 30), (90, 255, 255))
        disease_mask = (heatmap > 0.4).astype(np.uint8) * 255
        
        energy = np.mean(heatmap[heatmap > 0.6]) if np.any(heatmap > 0.6) else 0
        leaf_area = np.sum(leaf_mask > 0)
        disease_area = np.sum(np.bitwise_and(disease_mask > 0, leaf_mask > 0))
        severity = (disease_area / leaf_area * 100) if leaf_area > 0 else 0
        
        return heatmap, round(min(severity, 100.0), 2), energy

# --- 2. LOADING & ARCHITECTURE ---
@st.cache_resource
def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(model.last_channel, 4))
    
    paths = ["model.pth", "/content/model.pth"]
    model_path = next((p for p in paths if os.path.exists(p)), None)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        engine = AppleDiagnostics(model, model.features[-1])
        return model, engine, device
    return None, None, None

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="AppleAI Pro Dashboard", layout="wide")
st.title("🍎 AppleAI: Orchard Batch Pathologist")

model, engine, device = load_all()

if model:
    CLASS_NAMES = ['black_rot', 'healthy', 'rust', 'scab']
    TREATMENT = {
        "scab": {"med": "Captan 80 WDG", "price": 450},
        "rust": {"med": "Myclobutanil", "price": 600},
        "black_rot": {"med": "Mancozeb", "price": 550},
        "healthy": {"med": "Monitoring", "price": 0}
    }

    uploaded_files = st.file_uploader("📂 Upload Leaf Images (Batch Mode)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        batch_results = []
        progress = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            img = Image.open(file).convert("RGB")
            
            # Inference
            tf_inf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            input_inf = tf_inf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_inf)
                idx = torch.max(out, 1)[1].item()
                label = CLASS_NAMES[idx]
            
            heatmap, severity, energy = engine.analyze(img, idx)
            
            # Multimodal Decision Logic
            if label == 'healthy' and energy > 0.55:
                status = "Early Infection Alert"
                med = "Preventative Fungicide"
                cost = 300
            else:
                status = label.replace('_', ' ').title()
                med = TREATMENT[label]["med"]
                cost = int(TREATMENT[label]["price"] * (severity / 100))

            batch_results.append({
                "Filename": file.name,
                "Diagnosis": status,
                "Severity": f"{severity}%",
                "Treatment": med,
                "Estimated Cost (₹)": max(cost, 0)
            })
            progress.progress((i + 1) / len(uploaded_files))

        # --- SUMMARY TABLE SECTION ---
        st.subheader("📊 Orchard Batch Diagnostic Report")
        df = pd.DataFrame(batch_results)
        st.dataframe(df, use_container_width=True)

        # Aggregate Insights
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Analyzed", len(df))
        c2.metric("Diseased Found", len(df[df['Diagnosis'] != 'Healthy']))
        total_val = df["Estimated Cost (₹)"].sum()
        c3.metric("Total Orchard Treatment Cost", f"₹{total_val}")

        # Visual Gallery (First 2 images as preview)
        st.divider()
        st.write("🔍 Visual Spotlight (Latest Uploads)")
        cols = st.columns(min(len(uploaded_files), 3))
        for j, col in enumerate(cols):
            col.image(uploaded_files[j], caption=f"Result: {batch_results[j]['Diagnosis']}", use_container_width=True)
else:
    st.error("Model not found. Please train first.")
