
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms

# --- 1. THE DIAGNOSTIC ENGINE ---
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
        
        weights = np.mean(self.gradients.cpu().data.numpy()[0], axis=(1, 2))
        cam = np.maximum(np.dot(weights, self.activations.cpu().data.numpy()[0]), 0)
        heatmap = cv2.resize(cam, (224, 224))
        heatmap /= (heatmap.max() + 1e-8)

        img_np = np.array(img_pil.resize((224, 224)))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        # Green/Yellow mask to isolate the leaf from background
        leaf_mask = cv2.inRange(hsv, (5, 30, 30), (90, 255, 255))
        # Use AI heatmap to find diseased spots
        disease_mask = (heatmap > 0.5).astype(np.uint8) * 255
        
        severity = (np.sum(disease_mask > 0) / np.sum(leaf_mask > 0) * 100) if np.sum(leaf_mask > 0) > 0 else 0
        return heatmap, round(min(severity, 100.0), 2)

# --- 2. THE UI & AGRI-LOGIC ---
st.set_page_config(page_title="AppleAI Pro", layout="wide")
st.title("🍎 AppleAI: Precision Diagnostic & Treatment")

# Sidebar for Farmer Inputs
st.sidebar.header("Environment & Parameters")
acreage = st.sidebar.number_input("Orchard Size (Acres)", min_value=1, value=1)
humidity = st.sidebar.slider("Humidity (%)", 30, 100, 65)

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
    
    # Treatment Dictionary
    TREATMENT_PLAN = {
        "scab": {"med": "Captan 80 WDG", "base_price": 450, "desc": "Apply fungicide immediately. Improve air circulation."},
        "rust": {"med": "Myclobutanil", "base_price": 600, "desc": "Remove nearby juniper bushes (alternate host)."},
        "black_rot": {"med": "Mancozeb", "base_price": 550, "desc": "Prune dead wood and remove mummified fruit."},
        "healthy": {"med": "None", "base_price": 0, "desc": "Maintain regular watering and monitoring."}
    }

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        
        # 1. Prediction
        tf_inf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        input_inf = tf_inf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input_inf)
            idx = torch.max(out, 1)[1].item()
            label = CLASS_NAMES[idx]
            conf = torch.softmax(out, 1)[0, idx].item()

        # 2. Diagnostics
        heatmap, severity = engine.analyze(img, idx)
        
        # 3. Precision Cost Calculation
        # Humidity above 75% increases fungal growth risk (+30% cost)
        risk_factor = 1.3 if humidity > 75 else 1.0
        # Cost = (Base Price per Acre) * Severity Factor * Acreage * Risk Factor
        total_cost = int(TREATMENT_PLAN[label]["base_price"] * (severity / 100) * acreage * risk_factor)

        # UI Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption=f"Status: {label.upper()}", use_container_width=True)
            st.markdown(f"**Confidence:** {conf*100:.1f}%")
            if label == "healthy":
                st.balloons()
                st.success("✅ Your crop is healthy!")
            else:
                st.error(f"⚠️ Disease Detected: {label.replace('_', ' ').title()}")
                st.info(f"**Severity Level:** {severity}%")

        with col2:
            # Heatmap Visualization
            img_np = np.array(img.resize((224, 224)))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.4, 0)
            st.image(overlay, caption="Infected Hotspots (XAI Analysis)", use_container_width=True)
            
            if label != "healthy":
                st.subheader("📋 Recommended Treatment Plan")
                st.write(f"**Medicine:** {TREATMENT_PLAN[label]['med']}")
                st.write(f"**Guideline:** {TREATMENT_PLAN[label]['desc']}")
                st.markdown(f"### 💰 Estimated Cost: ₹{total_cost}")
                st.caption(f"Based on {acreage} acre(s) at {severity}% infection severity.")

except Exception as e:
    st.error("Model loading... Please ensure 'model.pth' is generated.")
