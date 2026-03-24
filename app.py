import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# class names (must match training)
class_names = ["black_rot", "healthy", "rust", "scab"]

device = torch.device("cpu")

# load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 4)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# transforms
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def predict(img):
    x = val_tf(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs,1)

    if conf.item() < 0.65:
        return "Uncertain", conf.item()

    return class_names[pred.item()], conf.item()

# UI
st.title("🍎 Apple Leaf Disease Detector")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    try:
        label, conf = predict(img)
        st.write(f"### Prediction: {label}")
        st.write(f"### Confidence: {conf:.2f}")
    except:
        st.error("Invalid image")
