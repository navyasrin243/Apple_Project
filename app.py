%%writefile app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURATION ---
CLASS_NAMES = ["black_rot", "healthy", "rust", "scab"]

# Research-Based Treatment & Pricing Database
STRATEGY_DB = {
    "scab": {"med": "Mancozeb", "org": "Neem Oil + Baking Soda", "base_price": 650},
    "rust": {"med": "Myclobutanil", "org": "Sulfur Spray", "base_price": 1100},
    "black_rot": {"med": "Captan", "org": "Copper Soap", "base_price": 850},
    "healthy": {"med": "N/A", "org": "Organic Nutrients", "base_price": 150}
}

# --- 2. XAI ENGINE (Grad-CAM) ---
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
            weights = np.mean(self.gradients.cpu().data.numpy()[0], axis=(1, 2))
            cam = np.maximum(np.dot(weights, self.activations.cpu().data.numpy()[0]), 0)
            cam = cv2.resize(cam, (224, 224))
            return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        finally:
            h1.remove(); h2.remove()

# --- 3. SMART ANALYTICS ENGINE ---
def analyze_sample(img_pil, heatmap, label):
    img_224 = np.array(img_pil.resize((224, 224)))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_RGB2HSV)
    
    # A. NON-LEAF FILTER
    leaf_mask = cv2.inRange(hsv, (5, 20, 20), (95, 255, 255))
    if (np.sum(leaf_mask > 0) / leaf_mask.size) < 0.15: return None 

    # B. HYBRID SEVERITY (Color Mask + AI Activation)
    if label == "healthy":
        return {"sev": 0.0, "health": 100, "status": "Optimal", "action": "Routine Organic Care", "cost": 150}

    color_mask = cv2.inRange(hsv, (10, 30, 40), (40, 255, 255))
    ai_mask = (heatmap > 0.45).astype(np.uint8) * 255
    disease_mask = cv2.bitwise_and(color_mask, ai_mask)
    
    severity = round((np.sum(disease_mask > 0) / np.sum(leaf_mask > 0)) * 100, 2)
    health = max(0, 100 - severity)
    
    # C. SMART RECOMMENDATION LOGIC
    db = STRATEGY_DB[label]
    if severity < 5:
        stat, act, cost = "Early Detection", f"🌱 Preventive: {db['org']}", db['base_price'] * 0.2
    elif severity < 20:
        stat, act, cost = "Initial Stage", f"🌱 Eco-Friendly: {db['org']}", db['base_price'] * 0.4
    elif severity < 50:
        stat, act, cost = "Moderate Infection", f"⚠️ Standard Dose: {db['med']}", db['base_price'] * 0.7
    else:
        stat, act, cost = "Critical Outbreak", f"🚨 Emergency: Full {db['med']} + Pruning", db['base_price']

    return {"sev": severity, "health": health, "status": stat, "action": act, "cost": int(cost)}

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="AppleAI Pro", layout="wide")
st.title("🍎 Apple Orchard Decision Intelligence")
st.markdown("---")

files = st.file_uploader("Upload Leaf Batches", accept_multiple_files=True)

if files:
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, 4)
    m.load_state_dict(torch.load("model.pth", map_location="cpu"))
    m.eval()
    gcam = GradCAM(m, m.features[-1])
    
    summary = []

    for f in files:
        img = Image.open(f).convert("RGB")
        x = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])(img).unsqueeze(0)
        
        with torch.set_grad_enabled(True):
            out = m(x)
            idx = torch.max(out, 1)[1].item()
            label = CLASS_NAMES[idx]
            heatmap = gcam.generate(x, idx)

        res = analyze_sample(img, heatmap, label)
        if res is None:
            st.warning(f"⚠️ {f.name} rejected: No leaf detected.")
            continue

        summary.append({"File": f.name, "Diagnosis": label.title(), "Health": res['health'], "Severity": res['sev'], "Cost": res['cost'], "Status": res['status']})

        with st.expander(f"Detail: {f.name} ({res['status']})"):
            c1, c2, c3 = st.columns(3)
            c1.image(img, use_container_width=True, caption="Sample")
            # Overlay
            img_np = np.array(img.resize((224,224)))
            ht = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
            ov = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(ht, cv2.COLOR_BGR2RGB), 0.4, 0)
            c2.image(ov, use_container_width=True, caption="AI Diagnosis Map")
            c3.metric("Health Score", f"{res['health']}%")
            c3.write(f"**Recommendation:** {res['action']}")
            c3.subheader(f"Cost: ₹{res['cost']}")

    if summary:
        df = pd.DataFrame(summary)
        st.divider()
        st.header("📊 Field Analytics Dashboard")
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Avg Field Health", f"{df['Health'].mean():.1f}%")
        # Acre Projection: Avg Cost * ~2.5 (Standard Labor/Density Factor)
        k2.metric("Projected Cost / Acre", f"₹{int(df['Cost'].mean() * 2.5)}")
        k3.metric("Critical Alerts", len(df[df['Severity'] > 40]))

        st.plotly_chart(px.bar(df, x="File", y="Severity", color="Diagnosis", title="Orchard Severity Breakdown"))
