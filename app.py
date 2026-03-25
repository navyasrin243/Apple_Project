import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import plotly.express as px

# --- 1. SMART TREATMENT & COST ENGINE ---
# Database for Decision Intelligence
DECISION_DB = {
    "scab": {"base_cost": 800, "med": "Mancozeb", "org": "Neem Oil"},
    "rust": {"base_cost": 1200, "med": "Myclobutanil", "org": "Sulfur Spray"},
    "black_rot": {"base_cost": 950, "med": "Captan", "org": "Copper Soap"},
    "healthy": {"base_cost": 200, "med": "N/A", "org": "Organic Nutrients"}
}

def get_smart_recommendation(label, severity):
    db = DECISION_DB.get(label, DECISION_DB["healthy"])
    
    if severity < 5:
        level = "TRACE (Early Detection)"
        advice = f"🌱 **Early Stage:** No chemicals needed. Use {db['org']} and increase pruning for airflow."
        cost = db['base_cost'] * 0.2
    elif severity < 20:
        level = "LOW"
        advice = f"🌿 **Mild Infection:** Apply {db['org']} mixed with baking soda. Monitor weekly."
        cost = db['base_cost'] * 0.5
    elif severity < 50:
        level = "MEDIUM"
        advice = f"⚠️ **Moderate Damage:** Systematic application of {db['med']} (2g/L) required immediately."
        cost = db['base_cost'] * 0.8
    else:
        level = "HIGH"
        advice = f"🚨 **Critical Outbreak:** Strong intervention with {db['med']} + destroy infected fallen leaves."
        cost = db['base_cost']
        
    return level, advice, int(cost)

# --- 2. XAI & PREDICTION SETUP ---
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
            cam = np.power(cam, 2)
            cam = cv2.resize(cam, (224, 224))
            return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), torch.softmax(logits, dim=1)
        finally:
            h1.remove(); h2.remove()

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="AppleAI Precision", layout="wide")
st.title("🍎 Apple Orchard Decision Intelligence")

files = st.file_uploader("Batch Upload Leaf Samples", accept_multiple_files=True)

if files:
    # Model Init
    m = models.mobilenet_v2(weights=None); m.classifier[1] = nn.Linear(m.last_channel, 4)
    m.load_state_dict(torch.load("model.pth", map_location="cpu")); m.eval()
    gcam = GradCAM(m, m.features[-1])
    
    all_results = []

    for f in files:
        img = Image.open(f).convert("RGB")
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        x = tf(img).unsqueeze(0)
        
        with torch.set_grad_enabled(True):
            heatmap, probs = gcam.generate(x, torch.max(m(x), 1)[1].item())
            conf, idx = torch.max(probs, 1)
            label = ["black_rot", "healthy", "rust", "scab"][idx.item()]
            conf_score = conf.item() * 100

        # Severity Logic (Hybrid Vision + Color)
        img_np = np.array(img.resize((224,224)))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (5, 30, 20), (35, 255, 180))
        ai_mask = (heatmap > 0.6).astype(np.uint8) * 255
        sev = (np.sum(cv2.bitwise_and(mask, ai_mask) > 0) / np.sum(cv2.inRange(hsv, (5,20,20), (95,255,255)) > 0)) * 100
        sev = sev if label != "healthy" else 0
        
        # Smart Logic
        level, advice, cost = get_smart_recommendation(label, sev)
        health_score = 100 - sev

        all_results.append({
            "File": f.name, "Diagnosis": label.title(), "Confidence": conf_score,
            "Severity": sev, "Health": health_score, "Level": level, "Cost": cost, "Advice": advice
        })

        with st.expander(f"Analysis: {f.name}"):
            c1, c2, c3 = st.columns(3)
            # Explainable AI Overlay
            ht = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
            ov = cv2.addWeighted(img_np, 0.5, cv2.cvtColor(ht, cv2.COLOR_BGR2RGB), 0.5, 0)
            c1.image(ov, caption="Explainable AI Map")
            c2.metric("Health Score", f"{health_score:.1f}%")
            c2.metric("AI Confidence", f"{conf_score:.1f}%")
            c3.subheader(f"Level: {level}")
            c3.info(advice)
            c3.write(f"**Sample Treatment Cost:** ₹{cost}")

    # --- FIELD ANALYTICS (Multi-Image Tracking) ---
    if all_results:
        df = pd.DataFrame(all_results)
        st.divider()
        st.header("🚜 Orchard Field Analytics (Multi-Modal)")
        k1, k2, k3 = st.columns(3)
        k1.metric("Field Health Score", f"{df['Health'].mean():.1f}%")
        k2.metric("Est. Cost / Acre", f"₹{int(df['Cost'].mean() * 3.5)}") # Acre multiplier
        k3.metric("Critical Samples", len(df[df['Severity'] > 40]))
        
        st.plotly_chart(px.bar(df, x="File", y="Severity", color="Diagnosis", title="Field Disease Distribution"))
