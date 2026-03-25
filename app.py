import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# --- 1. SMART TREATMENT & COST ENGINE ---
DECISION_DB = {
    "scab": {"base_cost": 800, "med": "Mancozeb", "org": "Neem Oil"},
    "rust": {"base_cost": 1200, "med": "Myclobutanil", "org": "Sulfur Spray"},
    "black_rot": {"base_cost": 950, "med": "Captan", "org": "Copper Soap"},
    "healthy": {"base_cost": 200, "med": "N/A", "org": "Organic Nutrients"}
}

def get_smart_recommendation(label, severity, hum):
    db = DECISION_DB.get(label, DECISION_DB["healthy"])
    # Risk increases with humidity
    risk_score = severity * (1.5 if hum > 80 else 1.0)
    
    if label == "healthy" or risk_score < 0.1:
        return "OPTIMAL", "✅ Leaf appears healthy. Maintain routine organic nutrients.", 200, 0.0

    if risk_score < 10:
        level, cost = "LOW", db['base_cost'] * 0.4
        advice = f"🌿 **Mild:** Apply {db['org']} and monitor."
    elif risk_score < 40:
        level, cost = "MEDIUM", db['base_cost'] * 0.7
        advice = f"⚠️ **Moderate:** Systematic {db['med']} application."
    else:
        level, cost = "HIGH", db['base_cost']
        advice = f"🚨 **Critical:** Strong {db['med']} + prune infected area."
        
    return level, advice, int(cost), round(risk_score, 2)

# --- 2. XAI ENGINE ---
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
            logits = self.model(x)
            logits[0, idx].backward()
            grads, acts = self.gradients.cpu().data.numpy()[0], self.activations.cpu().data.numpy()[0]
            weights = np.mean(grads, axis=(1, 2))
            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights): cam += w * acts[i, :, :]
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (224, 224))
            return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), torch.softmax(logits, dim=1)
        finally:
            h1.remove(); h2.remove()

# --- 3. UI ---
st.set_page_config(page_title="AppleAI Final", layout="wide")
st.title("🍎 AppleAI Precision Diagnostic")

hum = st.sidebar.slider("Humidity (%)", 30, 100, 75)
files = st.file_uploader("Upload Leaves", accept_multiple_files=True)

if files:
    m = models.mobilenet_v2(weights=None); m.classifier[1] = nn.Linear(m.last_channel, 4)
    m.load_state_dict(torch.load("model.pth", map_location="cpu")); m.eval()
    gcam = GradCAM(m, m.features[-1])
    summary = []

    for f in files:
        img = Image.open(f).convert("RGB")
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        x = tf(img).unsqueeze(0)
        
        with torch.set_grad_enabled(True):
            heatmap, probs = gcam.generate(x, torch.max(m(x), 1)[1].item())
            conf, idx = torch.max(probs, 1)
            label = ["black_rot", "healthy", "rust", "scab"][idx.item()]
            conf_score = conf.item() * 100

        # --- STABLE SEVERITY LOGIC ---
        img_np = np.array(img.resize((224,224)))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        leaf_mask = cv2.inRange(hsv, (5, 20, 20), (95, 255, 255))
        
        # Simple detection: AI High-Activation Areas
        ai_mask = (heatmap > 0.5).astype(np.uint8) * 255
        affected_pct = (np.sum(ai_mask > 0) / np.sum(leaf_mask > 0) * 100) if label != "healthy" else 0
        
        # --- STABLE OVERRIDE ---
        # If AI confidence is low, ignore the disease label and call it Healthy
        if conf_score < 80.0:
            label = "healthy"
            affected_pct = 0.0

        level, advice, cost, risk_idx = get_smart_recommendation(label, affected_pct, hum)
        summary.append({"File": f.name, "Result": label.upper(), "Confidence": f"{conf_score:.1f}%", "Affected": f"{affected_pct:.1f}%", "Cost": f"₹{cost}"})

        st.subheader(f"Report: {f.name}")
        c1, c2 = st.columns(2)
        
        ht = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        ov = cv2.addWeighted(img_np, 0.5, cv2.cvtColor(ht, cv2.COLOR_BGR2RGB), 0.5, 0)
        c1.image(ov, caption="AI Detection Area")
        
        c2.metric("Diagnosis", label.upper())
        c2.metric("Affected Area", f"{affected_pct:.1f}%")
        c2.info(advice)

    if summary:
        st.divider()
        st.subheader("📊 Batch Summary")
        st.table(pd.DataFrame(summary))
