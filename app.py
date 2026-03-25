import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# --- 1. SMART TREATMENT & COST DATABASE ---
TREATMENT_DB = {
    "scab": {"cost": 850, "med": "Mancozeb", "org": "Neem Oil"},
    "rust": {"cost": 1250, "med": "Myclobutanil", "org": "Sulfur Spray"},
    "black_rot": {"cost": 900, "med": "Captan", "org": "Copper Soap"},
    "healthy": {"cost": 150, "med": "N/A", "org": "Organic Nutrients"}
}

def get_decision_metrics(label, severity, hum):
    db = TREATMENT_DB.get(label, TREATMENT_DB["healthy"])
    # Multi-modal risk: severity weighted by humidity
    risk_index = severity * (1.4 if hum > 80 else 1.0)
    
    if label == "healthy" or risk_index < 0.5:
        return "OPTIMAL", "✅ Leaf is healthy. Continue routine monitoring.", 150, 100.0

    if risk_index < 15:
        level, cost = "LOW", db['cost'] * 0.4
        advice = f"🌿 **Action:** Use {db['org']} (5ml/L). High recovery chance."
    elif risk_index < 45:
        level, cost = "MEDIUM", db['cost'] * 0.7
        advice = f"⚠️ **Action:** Systematic {db['med']} spray required immediately."
    else:
        level, cost = "HIGH", db['cost']
        advice = f"🚨 **Action:** Emergency {db['med']} + prune and burn infected leaves."
        
    return level, advice, int(cost), round(100 - severity, 1)

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

# --- 3. UI SETUP ---
st.set_page_config(page_title="AppleAI Pro", layout="wide")
st.title("🍎 AppleAI: Precision Decision Intelligence")

st.sidebar.header("📡 Live Field Sensors")
hum = st.sidebar.slider("Ambient Humidity (%)", 30, 100, 75)

files = st.file_uploader("Upload Leaf Images", accept_multiple_files=True)

if files:
    # Model Loading
    m = models.mobilenet_v2(weights=None); m.classifier[1] = nn.Linear(m.last_channel, 4)
    m.load_state_dict(torch.load("model.pth", map_location="cpu")); m.eval()
    gcam = GradCAM(m, m.features[-1])
    
    final_summary = []

    for f in files:
        img = Image.open(f).convert("RGB")
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        x = tf(img).unsqueeze(0)
        
        with torch.set_grad_enabled(True):
            heatmap, probs = gcam.generate(x, torch.max(m(x), 1)[1].item())
            conf, idx = torch.max(probs, 1)
            label = ["black_rot", "healthy", "rust", "scab"][idx.item()]
            conf_score = conf.item() * 100

        # Severity/Affected Area Logic
        img_np = np.array(img.resize((224,224)))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        leaf_mask = cv2.inRange(hsv, (5, 20, 20), (95, 255, 255))
        ai_mask = (heatmap > 0.55).astype(np.uint8) * 255
        
        leaf_area = np.sum(leaf_mask > 0)
        affected_area = np.sum(ai_mask > 0)
        sev_pct = (affected_area / leaf_area * 100) if (label != "healthy" and leaf_area > 0) else 0.0

        # Stability Filter: If confidence is low, force Healthy
        if conf_score < 78.0:
            label, sev_pct = "healthy", 0.0

        # Get Smart Recommendations
        level, advice, cost, health_score = get_decision_metrics(label, sev_pct, hum)

        # Build Individual Report
        st.divider()
        st.subheader(f"📄 Report: {f.name}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Diagnosis", label.upper())
        col2.metric("Affected Area", f"{sev_pct:.1f}%")
        col3.metric("AI Confidence", f"{conf_score:.1f}%")
        col4.metric("Health Score", f"{health_score}%")

        c_left, c_right = st.columns(2)
        # Heatmap Visualization
        ht = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        ov = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(ht, cv2.COLOR_BGR2RGB), 0.4, 0)
        c_left.image(ov, use_container_width=True, caption="Explainable AI Spot Detection")
        
        # Treatment details
        c_right.warning(f"**Risk Level:** {level}")
        c_right.info(f"**Treatment Advice:**\n{advice}")
        c_right.success(f"**Estimated Cost:** ₹{cost}")

        final_summary.append({
            "File": f.name, "Result": label.upper(), "Severity %": f"{sev_pct:.1f}%", 
            "Confidence": f"{conf_score:.1f}%", "Health": f"{health_score}%", "Cost": f"₹{cost}"
        })

    # --- BATCH SUMMARY TABLE ---
    if final_summary:
        st.divider()
        st.header("📊 Final Batch Summary")
        st.table(pd.DataFrame(final_summary))
