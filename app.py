import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import plotly.express as px

# --- 1. RESEARCH-GRADE TREATMENT DATABASE ---
TREATMENT_PLAN = {
    "scab": {
        "Low": "🌿 **Organic Detection:** Spray 1% Potassium Bicarbonate. Prune shaded inner branches to improve airflow.",
        "Medium": "⚠️ **Standard Treatment:** Apply Mancozeb (2g/L). Repeat after 14 days if humidity remains >70%.",
        "High": "🚨 **Emergency Action:** Use Myclobutanil (40WP) immediately. Burn heavily infected fallen leaves to stop spore cycle."
    },
    "rust": {
        "Low": "🌿 **Organic Detection:** Apply Neem Oil (5ml/L) at sunset. Monitor nearby Cedar trees (alternate host).",
        "Medium": "⚠️ **Standard Treatment:** Propiconazole (1ml/L). Ensure 100% coverage of both leaf surfaces.",
        "High": "🚨 **Emergency Action:** Systemic Fungicide (Tilt 250 EC). Apply at 10-day intervals until new growth is clear."
    },
    "black_rot": {
        "Low": "🌿 **Organic Detection:** Copper-based soap spray. Remove any 'mummy' fruits left from last season.",
        "Medium": "⚠️ **Standard Treatment:** Captan 50 WP (2.5g/L). Focus application on the lower fruit clusters.",
        "High": "🚨 **Emergency Action:** Flutriafol application. Aggressively prune and destroy infected wood cankers."
    },
    "healthy": {
        "Low": "✅ **Optimal:** Maintain current mulch. Balanced NPK (10-10-10) application recommended.",
        "Medium": "✅ **Optimal:** Add Calcium-Boron foliar spray to strengthen leaf cell walls against future fungi.",
        "High": "✅ **Optimal:** Field health is excellent. Ensure irrigation does not wet the foliage directly."
    }
}

# --- 2. XAI ENGINE (Sharpened Grad-CAM) ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.target_layer = model, target_layer
        self.gradients, self.activations = None, None

    def save_grad(self, m, gi, go): self.gradients = go[0]
    def save_act(self, m, i, o): self.activations = o

    def generate(self, x, idx):
        h1 = self.target_layer.register_forward_hook(self.save_act)
        h2 = self.target_layer.register_full_backward_hook(self.save_grad)
        try:
            self.model.zero_grad()
            self.model(x)[0, idx].backward()
            grads, acts = self.gradients.cpu().data.numpy()[0], self.activations.cpu().data.numpy()[0]
            weights = np.mean(grads, axis=(1, 2))
            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights): cam += w * acts[i, :, :]
            cam = np.maximum(cam, 0)
            cam = np.power(cam, 2) # Sharpening spots
            cam = cv2.resize(cam, (224, 224))
            denom = cam.max() - cam.min()
            return (cam - cam.min()) / (denom if denom != 0 else 1e-8)
        finally:
            h1.remove(); h2.remove()

# --- 3. MULTI-MODAL ANALYTICS ---
def run_analytics(img_pil, heatmap, label, hum):
    img_224 = np.array(img_pil.resize((224, 224)))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_RGB2HSV)
    leaf_mask = cv2.inRange(hsv, (5, 20, 20), (95, 255, 255))
    leaf_px = np.sum(leaf_mask > 0)
    
    if (leaf_px / leaf_mask.size) < 0.10: return None

    # Vision Logic: Spot detection
    color_mask = cv2.inRange(hsv, (10, 40, 20), (35, 255, 180))
    ai_mask = (heatmap > 0.60).astype(np.uint8) * 255
    disease_mask = cv2.bitwise_and(color_mask, ai_mask)
    vision_sev = (np.sum(disease_mask > 0) / leaf_px) * 100 if label != "healthy" else 0

    # Multi-Modal Logic: Weather Fusion
    risk_index = vision_sev * (1.5 if hum > 80 else 1.0)
    
    if risk_index < 8: level = "Low"
    elif risk_index < 30: level = "Medium"
    else: level = "High"

    return {"sev": round(risk_index, 2), "level": level, "advice": TREATMENT_PLAN[label][level]}

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="AppleAI Multi-Modal", layout="wide")
st.title("🍎 AppleAI: Multi-Modal Decision Intelligence")

# Sidebar Sensors
st.sidebar.header("📡 Live Field Sensors")
hum = st.sidebar.slider("Ambient Humidity (%)", 30, 100, 75)
temp = st.sidebar.slider("Temperature (°C)", 10, 45, 24)

files = st.file_uploader("Upload Leaf Samples", accept_multiple_files=True)

if files:
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, 4)
    m.load_state_dict(torch.load("model.pth", map_location="cpu"))
    m.eval()
    gcam = GradCAM(m, m.features[-1])
    summary = []

    for f in files:
        img = Image.open(f).convert("RGB")
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        x = tf(img).unsqueeze(0)
        
        with torch.set_grad_enabled(True):
            out = m(x)
            idx = torch.max(out, 1)[1].item()
            label = ["black_rot", "healthy", "rust", "scab"][idx]
            heatmap = gcam.generate(x, idx)

        res = run_analytics(img, heatmap, label, hum)
        if not res: continue

        summary.append({"File": f.name, "Diagnosis": label.title(), "Level": res['level'], "Risk": res['sev']})

        with st.expander(f"REPORT: {f.name} - {res['level']} RISK"):
            c1, c2 = st.columns([1, 2])
            
            # XAI Heatmap Visualization
            img_np = np.array(img.resize((224,224)))
            ht = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
            ov = cv2.addWeighted(img_np, 0.5, cv2.cvtColor(ht, cv2.COLOR_BGR2RGB), 0.5, 0)
            c1.image(ov, caption="Explainable AI Map")
            
            # Detailed Prescription
            c2.subheader(f"Diagnosis: {label.replace('_',' ').title()}")
            c2.metric("Combined Risk Index", f"{res['sev']}%")
            c2.info(res['advice'])

    if summary:
        st.divider()
        df = pd.DataFrame(summary)
        st.subheader("📊 Orchard Disease Distribution")
        fig = px.bar(df, x="File", y="Risk", color="Level", 
                     color_discrete_map={"Low":"green", "Medium":"orange", "High":"red"},
                     title="Spatial Severity Analysis")
        st.plotly_chart(fig, use_container_width=True)
