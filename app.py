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
    # Multi-modal risk adjustment: Risk increases with humidity
    risk_score = severity * (1.5 if hum > 80 else 1.0)
    
    if label == "healthy" or risk_score < 0.5:
        return "OPTIMAL", "✅ Leaf appears healthy. Maintain routine organic nutrients.", 200, 0.0

    if risk_score < 8:
        level, cost = "TRACE (Early Detection)", db['base_cost'] * 0.2
        advice = f"🌱 **Early Stage:** No chemicals. Use {db['org']} and prune for airflow."
    elif risk_score < 25:
        level, cost = "LOW", db['base_cost'] * 0.5
        advice = f"🌿 **Mild Infection:** Apply {db['org']} mixed with baking soda."
    elif risk_score < 50:
        level, cost = "MEDIUM", db['base_cost'] * 0.8
        advice = f"⚠️ **Moderate Damage:** Systematic application of {db['med']} (2g/L)."
    else:
        level, cost = "HIGH", db['base_cost']
        advice = f"🚨 **Critical Outbreak:** Strong {db['med']} + destroy infected leaves."
        
    return level, advice, int(cost), round(risk_score, 2)

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
            cam = np.power(cam, 2) # Focus on specific spots
            cam = cv2.resize(cam, (224, 224))
            return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), torch.softmax(logits, dim=1)
        finally:
            h1.remove(); h2.remove()

# --- 3. UI & PROCESSING ---
st.set_page_config(page_title="AppleAI Pro", layout="wide")
st.title("🍎 AppleAI: Multi-Modal Precision Diagnostic")

# Sidebar for Multi-Modal Field Data
st.sidebar.header("📡 Field Sensors")
hum = st.sidebar.slider("Ambient Humidity (%)", 30, 100, 75)
st.sidebar.info("High humidity increases fungal risk scores.")

files = st.file_uploader("Upload Leaf Samples", accept_multiple_files=True)

if files:
    # Model Setup
    m = models.mobilenet_v2(weights=None); m.classifier[1] = nn.Linear(m.last_channel, 4)
    try:
        m.load_state_dict(torch.load("model.pth", map_location="cpu")); m.eval()
    except:
        st.error("Missing model.pth!"); st.stop()
        
    gcam = GradCAM(m, m.features[-1])
    batch_summary = []

    for f in files:
        img = Image.open(f).convert("RGB")
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        x = tf(img).unsqueeze(0)
        
        with torch.set_grad_enabled(True):
            heatmap, probs = gcam.generate(x, torch.max(m(x), 1)[1].item())
            conf, idx = torch.max(probs, 1)
            label = ["black_rot", "healthy", "rust", "scab"][idx.item()]
            conf_score = conf.item() * 100

        # Severity Calculation (Hybrid AI + Color Mask)
        img_np = np.array(img.resize((224,224)))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        leaf_mask = cv2.inRange(hsv, (5, 20, 20), (95, 255, 255))
        color_mask = cv2.inRange(hsv, (10, 40, 20), (35, 255, 180))
        ai_mask = (heatmap > 0.65).astype(np.uint8) * 255
        disease_px = np.sum(cv2.bitwise_and(color_mask, ai_mask) > 0)
        leaf_px = np.sum(leaf_mask > 0)
        affected_pct = (disease_px / leaf_px * 100) if (leaf_px > 0) else 0

        # --- SMART BALANCED FILTER (Fixes Healthy predicted as Scab) ---
        if label != "healthy":
            # If area is tiny AND confidence is not very high, it's likely background noise (bark)
            if affected_pct < 1.2 and conf_score < 92.0:
                label = "healthy"
                affected_pct = 0.0

        level, advice, cost, risk_idx = get_smart_recommendation(label, affected_pct, hum)
        
        # Save to summary
        batch_summary.append({
            "File": f.name, "Diagnosis": label.upper(), "Confidence": f"{conf_score:.1f}%",
            "Affected Area": f"{affected_pct:.1f}%", "Risk Level": level, "Cost": f"₹{cost}"
        })

        # Display Diagnostic Report
        st.divider()
        st.subheader(f"📋 Diagnostic Report: {f.name}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Classification", label.upper())
        m2.metric("Affected Area", f"{affected_pct:.1f}%", delta=level, delta_color="inverse")
        m3.metric("AI Confidence", f"{conf_score:.1f}%")

        c1, c2 = st.columns(2)
        # Heatmap
        ht = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        ov = cv2.addWeighted(img_np, 0.5, cv2.cvtColor(ht, cv2.COLOR_BGR2RGB), 0.5, 0)
        c1.image(ov, use_container_width=True, caption="Explainable AI Spot Detection")
        
        # Treatment
        c2.info(f"**Smart Treatment Engine:**\n\n{advice}")
        c2.success(f"**Overall Health Score:** {100 - affected_pct:.1f}%")
        c2.warning(f"**Estimated Treatment Cost:** ₹{cost}")

    # --- FINAL BATCH SUMMARY TABLE ---
    if batch_summary:
        st.divider()
        st.header("📊 Orchard Batch Analysis Summary")
        summary_df = pd.DataFrame(batch_summary)
        st.table(summary_df)
        
        # Acre Projection Logic
        avg_cost = int(pd.to_numeric(summary_df['Cost'].str.replace('₹','')).mean() * 4.5)
        st.write(f"🚜 **Orchard Projection:** Based on these samples, the projected treatment cost for this zone is **₹{avg_cost} per acre**.")
