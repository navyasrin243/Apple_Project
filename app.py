
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import pandas as pd
import os

# --- 1. THE DIAGNOSTIC ENGINE (With Spatial Focus) ---
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
        
        # ZOOM FEATURE: Crop center 85% to help AI focus on leaf texture over background
        w, h = img_pil.size
        img_cropped = img_pil.crop((w*0.07, h*0.07, w*0.93, h*0.93))
        
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_t = tf(img_cropped).unsqueeze(0).to(device)
        
        self.model.zero_grad()
        output = self.model(input_t)
        output[0, label_idx].backward()
        
        # Process Grad-CAM
        grads = self.gradients.detach().cpu().numpy().squeeze()
        acts = self.activations.detach().cpu().numpy().squeeze()
        weights = np.mean(grads, axis=(1, 2))
        
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]
            
        cam = np.maximum(cam, 0)
        heatmap = cv2.resize(cam, (224, 224))
        if heatmap.max() > 0:
            heatmap /= (heatmap.max() + 1e-8)

        # Severity Calculation using HSV Masking
        img_np = np.array(img_cropped.resize((224, 224)))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        leaf_mask = cv2.inRange(hsv, (5, 30, 30), (90, 255, 255))
        disease_mask = (heatmap > 0.4).astype(np.uint8) * 255
        
        leaf_px = np.sum(leaf_mask > 0)
        disease_px = np.sum(np.bitwise_and(disease_mask > 0, leaf_mask > 0))
        severity = (disease_px / leaf_px * 100) if leaf_px > 0 else 0
        
        return heatmap, round(min(severity, 100.0), 2), img_cropped

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=device))
        model.eval()
        engine = AppleDiagnostics(model, model.features[-1])
        return model, engine, device
    return None, None, None

# --- 3. BATCH UI & LOGIC ---
st.set_page_config(page_title="AppleAI Pro Max", layout="wide")
st.title("🍎 AppleAI: Precision Orchard Diagnostics")

model, engine, device = load_all()

if model:
    CLASS_NAMES = ['black_rot', 'healthy', 'rust', 'scab']
    TREATMENT = {
        "scab": {"med": "Captan 80 WDG", "price": 450},
        "rust": {"med": "Myclobutanil", "price": 600},
        "black_rot": {"med": "Mancozeb", "price": 550},
        "healthy": {"med": "Monitoring", "price": 0}
    }

    files = st.file_uploader("Upload Leaf Images", type=["jpg","png","jpeg"], accept_multiple_files=True)
    
    if files:
        summary_list = []
        for file in files:
            img = Image.open(file).convert("RGB")
            
            # 1. Inference with Probability Override
            tf_inf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            input_inf = tf_inf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_inf)
                probs = torch.softmax(out, 1)[0]
                idx = torch.max(out, 1)[1].item()
                
                # SENSITIVITY CHECK: If AI says healthy but Rust probability is > 15%, call it Rust.
                if idx == 1 and probs[2] > 0.15:
                    idx = 2

            # 2. XAI Analysis
            heatmap, severity, cropped_img = engine.analyze(img, idx)
            label = CLASS_NAMES[idx]
            
            summary_list.append({
                "File": file.name,
                "Diagnosis": label.replace('_', ' ').upper(),
                "Severity (%)": severity,
                "Medication": TREATMENT[label]["med"],
                "Est. Cost (₹)": int(TREATMENT[label]["price"] * (severity/100)),
                "heatmap": heatmap,
                "img": cropped_img
            })

        # --- 4. OUTPUT SUMMARY TABLE ---
        df = pd.DataFrame(summary_list).drop(columns=['heatmap', 'img'])
        st.table(df)
        
        st.metric("Total Orchard Treatment Cost", f"₹{df['Est. Cost (₹)'].sum()}")

        # --- 5. VISUAL EVIDENCE GRID ---
        st.subheader("🔍 Visual Pathological Evidence (Grad-CAM)")
        cols = st.columns(min(len(summary_list), 4))
        for i, res in enumerate(summary_list):
            with cols[i % 4]:
                img_np = np.array(res['img'].resize((224, 224)))
                h_map = cv2.applyColorMap(np.uint8(255 * res['heatmap']), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(h_map, cv2.COLOR_BGR2RGB), 0.4, 0)
                st.image(overlay, caption=f"{res['File']}: {res['Diagnosis']}")
else:
    st.error("Model 'model.pth' not found. Please train the model first.")
