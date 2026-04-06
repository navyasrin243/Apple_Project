
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms

# --- 1. SCIENTIFIC DIAGNOSTIC ENGINE (XAI) ---
class AppleDiagnostics:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Register hooks for Grad-CAM (Ref: Selvaraju et al. 2017)
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
        
        # 1. Grad-CAM Generation
        self.model.zero_grad()
        output = self.model(input_t)
        output[0, label_idx].backward()

        weights = np.mean(self.gradients.cpu().data.numpy()[0], axis=(1, 2))
        cam = np.maximum(np.dot(weights, self.activations.cpu().data.numpy()[0]), 0)
        heatmap = cv2.resize(cam, (224, 224))
        if heatmap.max() > 0: heatmap /= heatmap.max()

        # 2. Mathematical Severity (FAO Percent Disease Index Logic)
        img_np = np.array(img_224)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        # Mask to isolate leaf (Green/Yellow/Brown range)
        leaf_mask = cv2.inRange(hsv, (5, 30, 30), (95, 255, 255))
        # Mask to isolate disease from AI Heatmap
        disease_mask = (heatmap > 0.4).astype(np.uint8) * 255
        
        leaf_area = np.sum(leaf_mask > 0)
        severity = (np.sum(disease_mask > 0) / leaf_area * 100) if leaf_area > 0 else 0
        
        return heatmap, round(min(severity, 100.0), 2)

# --- 2. STREAMLIT UI & AGRI-DATABASE ---
st.set_page_config(page_title="AppleAI Pro", layout="wide")
st.title("🍎 AppleAI: Automated Pathological Assessment")

# Standardized Treatment Database (Based on IPM Guidelines)
AGRI_DB = {
    "scab": {
        "med": "Captan 80 WDG", 
        "base_price": 450, 
        "desc": "Protective fungicide. Targeted at Venturia inaequalis."
    },
    "rust": {
        "med": "Myclobutanil", 
        "base_price": 600, 
        "desc": "Systemic treatment. Advised: Remove nearby Juniper hosts."
    },
    "black_rot": {
        "med": "Mancozeb", 
        "base_price": 550, 
        "desc": "Broad-spectrum control. Advised: Prune and burn cankers."
    },
    "healthy": {
        "med": "N/A", 
        "base_price": 0, 
        "desc": "Continue standard irrigation and nutrient monitoring."
    }
}

@st.cache_resource
def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    engine = AppleDiagnostics(model, model.features[-1])
    return model, engine, device

try:
    model, engine, device = load_all()
    CLASS_NAMES = ['black_rot', 'healthy', 'rust', 'scab']

    uploaded_files = st.file_uploader("Upload Orchard Samples", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        results_data = []
        
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            
            # Prediction
            inf_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
            input_inf = inf_tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_inf)
                idx = out.argmax(1).item()
                label = CLASS_NAMES[idx]
            
            # Diagnostics
            heatmap, severity = engine.analyze(img, idx)
            info = AGRI_DB[label]
            
            # Economic Calculation (Cost = Base + Severity Factor)
            total_cost = int(info['base_price'] * (severity/100 + 0.1)) if label != "healthy" else 0

            results_data.append({
                "File": file.name,
                "Diagnosis": label.upper(),
                "Severity": f"{severity}%",
                "Medicine": info['med'],
                "Treatment Cost": f"₹{total_cost}",
                "h": heatmap,
                "img": img
            })

        # Display Summary Table
        df = pd.DataFrame(results_data).drop(columns=['h', 'img'])
        st.table(df)

        # Display Visual Proofs
        st.subheader("Diagnostic Proofs (Grad-CAM XAI)")
        cols = st.columns(4)
        for i, res in enumerate(results_data):
            with cols[i%4]:
                img_np = np.array(res['img'].resize((224, 224)))
                h_map = cv2.applyColorMap(np.uint8(255 * res['h']), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(h_map, cv2.COLOR_BGR2RGB), 0.4, 0)
                st.image(overlay, caption=f"{res['File']} - {res['Diagnosis']}")

except Exception as e:
    st.error(f"System Ready. Please upload files. (Error trace: {e})")
