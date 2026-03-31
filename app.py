%%writefile app.py
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import os

# --- 1. THE DIAGNOSTIC ENGINE (Fixed for Dimension Mismatch) ---
class AppleDiagnostics:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Hook to capture internal features and gradients
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
        
        # Pull data to CPU and remove batch dimension
        grads = self.gradients.detach().cpu().numpy().squeeze()
        acts = self.activations.detach().cpu().numpy().squeeze()

        # Global Average Pooling of Gradients (Weights)
        # MobileNetV2 has 1280 channels at the final feature layer
        weights = np.mean(grads, axis=(1, 2))
        
        # Generate Weighted Heatmap
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]
            
        # ReLU: Keep only positive features
        cam = np.maximum(cam, 0)
        
        # Rescale and Normalize
        heatmap = cv2.resize(cam, (224, 224))
        if heatmap.max() > 0:
            heatmap /= (heatmap.max() + 1e-8)

        # Calculate Severity via Masking
        img_np = np.array(img_pil.resize((224, 224)))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        # Detect the leaf area (green/yellow spectrum)
        leaf_mask = cv2.inRange(hsv, (5, 30, 30), (90, 255, 255))
        # Detect the disease area from the heatmap hotspots
        disease_mask = (heatmap > 0.4).astype(np.uint8) * 255
        
        leaf_area = np.sum(leaf_mask > 0)
        disease_area = np.sum(np.bitwise_and(disease_mask > 0, leaf_mask > 0))
        
        severity = (disease_area / leaf_area * 100) if leaf_area > 0 else 0
        return heatmap, round(min(severity, 100.0), 2)

# --- 2. STREAMLIT UI ---
st.set_page_config(page_title="AppleAI Pro", layout="wide")
st.title("🍎 AppleAI: Precision Pathologist")

@st.cache_resource
def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize MobileNetV2 architecture
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    
    # Locate model.pth in standard or Colab paths
    paths = ["model.pth", "/content/model.pth"]
    model_path = next((p for p in paths if os.path.exists(p)), None)
    
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        engine = AppleDiagnostics(model, model.features[-1])
        return model, engine, device
    return None, None, None

model, engine, device = load_all()

if model is None:
    st.error("🚨 'model.pth' not found. Please run the training cell or upload the file.")
else:
    CLASS_NAMES = ['black_rot', 'healthy', 'rust', 'scab']
    TREATMENT = {
        "scab": {"med": "Captan 80 WDG", "price": 450},
        "rust": {"med": "Myclobutanil", "price": 600},
        "black_rot": {"med": "Mancozeb", "price": 550},
        "healthy": {"med": "None", "price": 0}
    }

    uploaded_file = st.file_uploader("Upload Leaf Photo", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        
        # 1. Classify
        tf_inf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        input_inf = tf_inf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input_inf)
            idx = torch.max(out, 1)[1].item()
            label = CLASS_NAMES[idx]
            conf = torch.softmax(out, 1)[0, idx].item()

        # 2. Analyze (Diagnosis)
        heatmap, severity = engine.analyze(img, idx)
        
        # 3. Display Results
        st.subheader(f"Detection Result: {label.upper()} ({conf*100:.1f}%)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Input", use_container_width=True)
            st.metric("Infection Severity", f"{severity}%")
            
        with col2:
            # Create Heatmap Overlay
            img_np = np.array(img.resize((224, 224)))
            heatmap_c = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB), 0.4, 0)
            st.image(overlay, caption="Disease Hotspots (Grad-CAM)", use_container_width=True)
            
            if label != "healthy":
                est_cost = int(TREATMENT[label]["price"] * (severity / 100))
                st.warning(f"**Treatment Recommendation:** {TREATMENT[label]['med']}")
                st.info(f"**Estimated Precision Cost:** ₹{max(est_cost, 50)}")
            else:
                st.success("✅ The leaf appears healthy. Continue normal monitoring.")
                st.balloons()
