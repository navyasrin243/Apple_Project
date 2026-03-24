import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os
import numpy as np
import cv2

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Apple Disease Detector", layout="centered")

# ---------------- HEADER ----------------
st.title("🍎 Apple Leaf Disease Detector")
st.caption("AI-powered detection with explainability & severity estimation")

# ---------------- CLASS NAMES ----------------
class_names = ["black_rot", "healthy", "rust", "scab"]

# ---------------- INFO ----------------
disease_info = {
    "scab": "Fungal disease causing dark lesions.",
    "rust": "Yellow-orange spots on leaves.",
    "black_rot": "Dark necrotic patches.",
    "healthy": "No disease detected."
}

treatment = {
    "scab": "Apply fungicides like captan.",
    "rust": "Use sulfur sprays.",
    "black_rot": "Prune infected areas.",
    "healthy": "No action needed."
}

explain = {
    "scab": "Irregular dark lesions detected.",
    "rust": "Clustered orange spots detected.",
    "black_rot": "Dark decaying regions found.",
    "healthy": "No abnormal patterns."
}

# ---------------- DEVICE ----------------
device = torch.device("cpu")

# ---------------- MODEL ----------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 4)

model_path = os.path.join(os.path.dirname(__file__), "model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------------- TRANSFORM ----------------
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ---------------- PREDICT ----------------
def predict(img):
    x = val_tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs,1)
    return class_names[pred.item()], conf.item(), probs.numpy()[0], out

# ---------------- TRUE SEVERITY ----------------
def calculate_severity(img):
    img = img.resize((224,224))
    img_np = np.array(img)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    lower = np.array([10, 50, 50])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    ratio = np.sum(mask > 0) / mask.size

    if ratio < 0.05:
        return "Low", ratio
    elif ratio < 0.15:
        return "Moderate", ratio
    else:
        return "High", ratio

# ---------------- GRADCAM ----------------
def generate_gradcam(img, model, target_layer):
    model.eval()
    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    x = val_tf(img).unsqueeze(0)
    out = model(x)
    pred = out.argmax()

    model.zero_grad()
    out[0, pred].backward()

    grads = gradients[0][0].detach().numpy()
    acts = activations[0][0].detach().numpy()

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))

    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    handle_f.remove()
    handle_b.remove()

    return cam

def overlay_cam(img, cam):
    img = img.resize((224,224))
    img_np = np.array(img)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = 0.5 * heatmap + 0.5 * img_np
    return overlay.astype(np.uint8)

# ---------------- UI ----------------
st.info("Upload a clear apple leaf image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        img.verify()

        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Input Image", use_column_width=True)

        with col2:
            label, conf, probs, raw_out = predict(img)

            st.subheader(f"Prediction: {label.upper()}")
            st.progress(int(conf * 100))
            st.write(f"Confidence: {conf:.2f}")

            # severity
            severity, ratio = calculate_severity(img)
            st.write(f"🔥 Severity: {severity} ({ratio*100:.1f}%)")

            # explanation
            st.write("🧠 Why this prediction?")
            st.info(explain[label])

            # disease info
            st.info(disease_info[label])

            # treatment
            st.write("💊 Treatment")
            st.success(treatment[label])

            # probabilities
            st.write("📊 Class Probabilities:")
            for i, cls in enumerate(class_names):
                st.write(f"{cls}: {probs[i]:.2f}")

        # Grad-CAM
        st.markdown("### 🔍 Model Attention (Grad-CAM)")
        cam = generate_gradcam(img, model, model.features[-1])
        overlay = overlay_cam(img, cam)
        st.image(overlay, caption="Red = model focus", use_column_width=True)

    except:
        st.error("Invalid image file")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built using Deep Learning with Explainable AI")
