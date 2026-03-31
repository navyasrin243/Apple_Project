
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import os

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
        leaf_mask = cv2.inRange(hsv, (5, 30, 30), (90, 255, 255))
        disease_mask = (heatmap > 0.5).astype(np.uint8) * 255
        severity = (np.sum(disease_mask > 0) / np.sum(leaf_mask > 0) * 100) if np.sum(leaf_mask > 0) > 0 else 0
        return heatmap, round(min(severity, 100.0), 2)

# --- 2. APP UI ---
st.set_page_config(page_title="AppleAI Pro", layout="wide")
st.title("🍎 AppleAI Precision Diagnostic")

@st.cache_resource
def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    
    # Check multiple possible paths for the model file
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
    st.error("🚨 Critical Error: 'model.pth' not detected in directory.")
    st.info("Please ensure you have run the training cell or uploaded the weight file.")
else:
    CLASS_NAMES = ['black_rot', 'healthy', 'rust', 'scab']
    uploaded_file = st.file_uploader("Upload Leaf Photo", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        # Diagnostic process...
        # [Rest of the display logic from previous blocks]
        st.success("Model loaded and ready for analysis!")
