import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURATION ---
CLASS_NAMES = ["black_rot", "healthy", "rust", "scab"]

# Decision Database
STRATEGY_DB = {
    "scab": {"med": "Mancozeb", "org": "Neem Oil", "base_price": 650, "risk_temp": 22},
    "rust": {"med": "Myclobutanil", "org": "Sulfur", "base_price": 1100, "risk_temp": 25},
    "black_rot": {"med": "Captan", "org": "Copper Soap", "base_price": 850, "risk_temp": 28},
    "healthy": {"med": "N/A", "org": "Nutrients", "base_price": 150, "risk_temp": 0}
}

# --- 2. MULTI-MODAL SIDEBAR (Sensors) ---
st.sidebar.header("📡 Field Sensor Data (Multi-Modal)")
humidity = st.sidebar.slider("Relative Humidity (%)", 30, 100, 75)
temperature = st.sidebar.slider("Ambient Temperature (°C)", 10, 45, 24)
st.sidebar.info("High humidity (>80%) accelerates fungal spore germination.")

# --- 3. SHARPENED XAI ENGINE ---
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
            grads = self.gradients.cpu().data.numpy()[0]
            acts = self.activations.cpu().data.numpy()[0]
            weights = np.mean(grads, axis=(1, 2))
            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights): cam += w * acts[i, :, :]
            cam = np.maximum(cam, 0)
            cam = np.power(cam, 2) # Power Sharpening for spot accuracy
            cam = cv2.resize(cam, (224, 224))
            denom = cam.max() - cam.min()
            return (cam - cam.min()) / (denom if denom != 0 else 1e-8)
        finally:
            h1.remove(); h2.remove()

# --- 4. MULTI-MODAL ANALYTICS ENGINE ---
def analyze_multi_modal(img_pil, heatmap, label, hum, temp):
    img_224 = np.array(img_pil.resize((224, 224)))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_RGB2HSV)
    
    # Vision Modality
    leaf_mask = cv2.inRange(hsv, (5, 20, 20), (95, 255, 255))
    leaf_pixels = np.sum(leaf_mask > 0)
    if (leaf_pixels / leaf_mask.size) < 0.10: return None 

    if label == "healthy":
        return {"sev": 0, "level": "OPTIMAL", "health": 100, "cost": 150}

    # Precision Spot Detection (Vision + Color)
    color_mask = cv2.inRange(hsv, (10, 40, 20), (35, 255, 180)) 
    ai_mask = (heatmap > 0.65).astype(np.uint8) * 255
    disease_mask = cv2.bitwise_and(color_mask, ai_mask)
    
    vision_sev = (np.sum(disease_mask > 0) / leaf_pixels) * 100
    
    # Sensor Modality (Weather Fusion)
    # Scab/Rust risk multiplier based on Humidity
    weather_multiplier = 1.4 if hum > 80 else 1.0
    final_sev = min(100, vision_sev * weather_multiplier)
    
    # Categorization
    if final_sev < 8: level = "LOW (Trace)"
    elif final_sev < 30: level = "MEDIUM (Active)"
    else: level = "HIGH (Critical Outbreak)"
    
    cost = STRATEGY_DB[label]['base_price'] * (0.3 if final_sev < 8 else 0.7 if final_sev < 30 else 1.0)
    return {"sev": round(final_sev, 2), "level": level, "health": round(100-final_sev, 2), "cost": int(cost)}

# --- 5. STREAMLIT INTERFACE ---
st.set_page_config(page_title="AppleAI Multi-Modal", layout="wide")
st.title("🍎 Multi-Modal Orchard Intelligence")

files = st.file_uploader("Batch Upload Leaf Images", accept_multiple_files=True)

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
            label = CLASS_NAMES[idx]
            heatmap = gcam.generate(x, idx)

        res = analyze_multi_modal(img, heatmap, label, humidity, temperature)
        if res is None: continue

        summary.append({"File": f.name, "Diagnosis": label.title(), "Level": res['level'], "Severity": res['sev'], "Cost": res['cost']})

        with st.expander(f"Analysis: {f.name} - {res['level']}"):
            c1, c2, c3 = st.columns(3)
            c1.image(img, use_container_width=True, caption="Sample")
            img_np = np.array(img.resize((224,224)))
            ht = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
            ov = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(ht, cv2.COLOR_BGR2RGB), 0.4, 0)
            c2.image(ov, use_container_width=True, caption="AI Heatmap (Sharp)")
            c3.metric("Severity", f"{res['sev']}%")
            c3.subheader(f"Level: {res['level']}")
            c3.write(f"Treatment Cost: ₹{res['cost']}")

    if summary:
        df = pd.DataFrame(summary)
        st.divider()
        st.header("🚜 Orchard Dashboard (Multi-Modal)")
        k1, k2, k3 = st.columns(3)
        k1.metric("Avg Severity", f"{df['Severity'].mean():.1f}%")
        k2.metric("Total Treatment Cost", f"₹{df['Cost'].sum()}")
        k3.metric("Critical Alerts", len(df[df['Level'].str.contains("HIGH")]))
        st.plotly_chart(px.bar(df, x="File", y="Severity", color="Level", title="Field Severity Distribution"))
