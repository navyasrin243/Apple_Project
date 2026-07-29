%%writefile app.py
"""
AppleAI Pro — Streamlit App
Run with:  streamlit run app.py
Requires:  model_best.pth in the same directory
"""
import os
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ── 1. Grad-CAM engine ────────────────────────────────────────
class GradCAMEngine:
    """
    Grad-CAM with per-class severity thresholds.
    Target: m.features[-1][0]  (ConvBNActivation inside the last
    InvertedResidual block — NOT the Sequential wrapper, which
    breaks hooks silently on some PyTorch versions).
    """
    # Empirically tuned per-class activation thresholds
    THRESHOLDS = {
        'black_rot': 0.50,
        'rust'     : 0.38,
        'scab'     : 0.45,
        'healthy'  : 0.99,
    }

    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, out):        self.activations = out
    def _save_grad(self, _, __, grad_out):  self.gradients   = grad_out[0]

    def analyze(self, img_pil, label_idx, label_name):
        device = next(self.model.parameters()).device
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
        inp = tf(img_pil).unsqueeze(0).to(device)
        inp.requires_grad_(True)

        self.model.zero_grad()
        out = self.model(inp)
        out[0, label_idx].backward()

        # Grad-CAM: global-average-pool the gradients → weight the activations
        grads   = self.gradients.detach().cpu().numpy()[0]   # (C,H,W)
        acts    = self.activations.detach().cpu().numpy()[0] # (C,H,W)
        weights = np.maximum(np.mean(grads, axis=(1, 2)), 0) # relu on weights
        cam     = np.sum(weights[:, None, None] * acts, axis=0)
        cam     = np.maximum(cam, 0)

        heatmap = cv2.resize(cam, (224, 224))
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # Severity: diseased pixels / total leaf pixels
        img_np  = np.array(img_pil.resize((224, 224)))
        hsv     = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        leaf_mask    = cv2.inRange(hsv, (5, 30, 30), (95, 255, 255))
        thr          = self.THRESHOLDS.get(label_name, 0.4)
        disease_mask = (heatmap > thr).astype(np.uint8) * 255
        leaf_px      = max(np.sum(leaf_mask > 0), 1)
        disease_px   = np.sum((disease_mask > 0) & (leaf_mask > 0))
        severity     = round(min(disease_px / leaf_px * 100, 100.0), 2)

        return heatmap, severity


# ── 2. Knowledge base ─────────────────────────────────────────
AGRI_DB = {
    'black_rot': {
        'med' : 'Mancozeb',
        'base': 550,
        'info': ('Prune infected branches immediately. '
                 'Apply copper-based fungicide after pruning. '
                 'Destroy all infected debris.'),
    },
    'healthy': {
        'med' : 'N/A',
        'base': 0,
        'info': ('No pathology detected. '
                 'Maintain current irrigation and fertilisation schedule.'),
    },
    'rust': {
        'med' : 'Myclobutanil',
        'base': 600,
        'info': ('Fungal spores detected. '
                 'Check for nearby Cedar or Juniper host trees. '
                 'Apply protective fungicide before rain events.'),
    },
    'scab': {
        'med' : 'Captan 80 WDG',
        'base': 450,
        'info': ('Apple scab thrives in humidity. '
                 'Increase canopy airflow via selective pruning. '
                 'Apply fungicide at green tip through petal fall.'),
    },
}

LABELS = ['black_rot', 'healthy', 'rust', 'scab']


# ── 3. Cached model loader ────────────────────────────────────
@st.cache_resource
def load_model(path='model_best.pth'):
    if not os.path.exists(path):
        return None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, 4)
    m.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    m.eval().to(device)
    # FIX: hook ConvBNActivation inside last block, not the Sequential
    engine = GradCAMEngine(m, m.features[-1][0])
    return m, engine


# ── 4. Page layout ────────────────────────────────────────────
st.set_page_config(
    page_title='AppleAI Pro',
    page_icon='🍎',
    layout='wide',
)

st.title('🍎 AppleAI Pro: Automated Pathological Assessment')
st.caption('Integrated Computer Vision & Economic Estimator '
           'for Precision Apple Agriculture')
st.divider()

model, engine = load_model()

if model is None:
    st.error(
        "⚠️ **model_best.pth not found.**  "
        "Run the training notebook first, then re-launch this app."
    )
    st.stop()

# ── 5. Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header('ℹ️ About')
    st.write(
        'Model: **MobileNetV2** fine-tuned on PlantVillage + PlantDoc.  \n'
        'XAI  : **Grad-CAM** lesion localisation.  \n'
        'Classes: Black Rot | Healthy | Rust | Scab'
    )
    st.divider()
    st.caption('Upload a clear photo of a single apple leaf.')

# ── 6. File upload & inference ────────────────────────────────
uploaded = st.file_uploader(
    'Upload Leaf Specimen',
    type=['jpg', 'jpeg', 'png'],
    help='JPG or PNG, ideally 224×224 px or larger'
)

if uploaded:
    img    = Image.open(uploaded).convert('RGB')
    device = next(model.parameters()).device

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        logits   = model(tf(img).unsqueeze(0).to(device))
        probs    = torch.softmax(logits, 1)[0].cpu().numpy()
        idx      = int(probs.argmax())
        conf     = float(probs[idx])

    label    = LABELS[idx]
    heatmap, severity = engine.analyze(img, idx, label)
    db   = AGRI_DB[label]
    cost = int(db['base'] * (severity / 100 + 1)) if label != 'healthy' else 0

    # ── Columns ───────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader('📸 Original Leaf')
        st.image(img, use_container_width=True)
        st.metric('Diagnosis', label.replace('_', ' ').title())
        st.progress(conf, text=f'Confidence: {conf*100:.1f}%')

        # All-class probability bar chart
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=(4, 2.5))
        colors  = ['#e74c3c' if i == idx else '#95a5a6'
                   for i in range(4)]
        ax.barh(LABELS, probs * 100, color=colors)
        ax.set_xlabel('Probability (%)')
        ax.set_title('Class Probabilities')
        ax.set_xlim(0, 100)
        _plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.subheader('🔬 Grad-CAM Heatmap')
        img_np     = np.array(img.resize((224, 224)))
        heat_color = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(
            img_np, 0.6,
            cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB), 0.4, 0
        )
        st.image(overlay, caption='Lesion Localisation',
                 use_container_width=True)

        sev_color = (
            '🟢' if severity < 20 else
            '🟡' if severity < 50 else '🔴'
        )
        st.metric('Severity Index', f'{severity}%',
                  delta=sev_color, delta_color='off')

    with c3:
        st.subheader('📋 Treatment Plan')
        st.metric('Estimated Treatment Cost', f'₹ {cost:,}')
        st.write(f'**Recommended Fungicide:** {db["med"]}')
        st.info(f'🌿 **Expert Note:** {db["info"]}')
        st.divider()
        if label == 'healthy':
            st.success('✅ Leaf is healthy. No treatment required.')
        else:
            st.warning(
                f'⚠️ **{label.replace("_"," ").title()}** detected  \n'
                f'Confidence: {conf*100:.1f}%  \n'
                f'Severity: {severity}%  \n'
                f'Treat with **{db["med"]}** — est. cost **₹ {cost:,}**'
            )
