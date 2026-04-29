

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import os
import time

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CNN Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS  (dark industrial theme)
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #0d0f14;
    --surface:  #161920;
    --border:   #2a2e3a;
    --accent:   #00f5c4;
    --accent2:  #7c6af5;
    --warn:     #f5a623;
    --text:     #e2e8f0;
    --muted:    #6b7280;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Titles */
h1 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.6rem !important;
     background: linear-gradient(135deg, var(--accent), var(--accent2));
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2, h3 { font-family: 'Syne', sans-serif; font-weight: 600; color: var(--text); }

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    transition: border-color .25s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* Progress bars */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)); }

/* Metric labels */
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .78rem !important;
    font-family: 'Space Mono', monospace !important; text-transform: uppercase; letter-spacing:.08em; }
[data-testid="stMetricValue"]   { font-family: 'Space Mono', monospace !important;
    color: var(--accent) !important; }

/* Tag pill */
.pill {
    display: inline-block;
    padding: .25rem .75rem;
    border-radius: 999px;
    font-size: .78rem;
    font-family: 'Space Mono', monospace;
    background: rgba(0,245,196,.12);
    border: 1px solid rgba(0,245,196,.35);
    color: var(--accent);
    margin-right: .4rem;
    margin-top: .3rem;
}

/* Architecture table */
.arch-row {
    display: flex;
    gap: 1rem;
    padding: .55rem 0;
    border-bottom: 1px solid var(--border);
    font-family: 'Space Mono', monospace;
    font-size: .82rem;
    align-items: center;
}
.arch-row:last-child { border-bottom: none; }
.arch-label { color: var(--accent); width: 120px; flex-shrink:0; }
.arch-desc  { color: var(--muted); }

/* Prediction bar */
.pred-bar-bg {
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 4px;
}
.pred-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width .6s ease;
}
.pred-label {
    display: flex;
    justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: .8rem;
    color: var(--muted);
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    padding: .55rem 1.4rem !important;
}
.stButton > button:hover { opacity: .88; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD MODEL (cached)
# ──────────────────────────────────────────────
MODEL_PATH       = "cnn_cifar10.h5"
CLASS_NAMES_PATH = "class_names.json"

@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        names = json.load(f)
    return model, names

model, CLASS_NAMES = load_model_and_labels()

CLASS_EMOJIS = {
    "airplane":"✈️","automobile":"🚗","bird":"🐦","cat":"🐱","deer":"🦌",
    "dog":"🐶","frog":"🐸","horse":"🐴","ship":"🚢","truck":"🚛",
}

# ──────────────────────────────────────────────
# SIDEBAR – Architecture info
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 CNN Architecture")
    st.markdown("**Model:** Custom 3-block CNN")
    st.markdown("**Dataset:** CIFAR-10  (60 000 images, 10 classes)")
    st.markdown("---")

    layers_info = [
        ("Input",      "32×32×3  RGB image"),
        ("Conv Block 1","2× Conv2D(32) + BN + MaxPool + Dropout"),
        ("Conv Block 2","2× Conv2D(64) + BN + MaxPool + Dropout"),
        ("Conv Block 3","2× Conv2D(128)+ BN + MaxPool + Dropout"),
        ("Flatten",    "4×4×128 → 2 048 units"),
        ("Dense",      "512 units + BN + ReLU + Dropout(0.5)"),
        ("Output",     "10 units  (Softmax)"),
    ]
    for lname, ldesc in layers_info:
        st.markdown(f"""
        <div class="arch-row">
          <span class="arch-label">{lname}</span>
          <span class="arch-desc">{ldesc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Optimizer:** Adam (lr = 1e-3)")
    st.markdown("**Loss:** Categorical Cross-entropy")
    st.markdown("**Augmentation:** Flip · Rotate · Zoom · Shift")

    if model:
        total_params = model.count_params()
        st.markdown(f"**Parameters:** {total_params:,}")

    st.markdown("---")
    st.markdown("""
    <span class="pill">Conv2D</span>
    <span class="pill">BatchNorm</span>
    <span class="pill">ReLU</span>
    <span class="pill">MaxPool</span>
    <span class="pill">Dropout</span>
    <span class="pill">Dense</span>
    <span class="pill">Softmax</span>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────
st.title("CNN Image Classifier")
st.markdown("**CIFAR-10 · Deep Learning · Real-time Inference**")

if model is None:
    st.error("""
    ⚠️  **Model not found.**  
    Please run `python train_cnn.py` first to train and save `cnn_cifar10.h5`.
    """)
    st.stop()

st.markdown("---")

col_upload, col_result = st.columns([1, 1], gap="large")

# ── Upload Column ────────────────────────────
with col_upload:
    st.markdown("### 📤 Upload Image")
    st.caption("Supported classes: airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck")

    uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)

        # Preprocess
        img_resized = img.resize((32, 32))
        img_array  = np.array(img_resized).astype("float32") / 255.0
        img_batch  = np.expand_dims(img_array, axis=0)        # (1, 32, 32, 3)

        predict_btn = st.button("🔍  Run Inference")

# ── Results Column ───────────────────────────
with col_result:
    st.markdown("### 📊 Predictions")

    if uploaded and predict_btn:
        with st.spinner("Running CNN forward pass…"):
            time.sleep(0.4)                      # visual feedback pause
            preds = model.predict(img_batch, verbose=0)[0]   # (10,)

        top_idx   = int(np.argmax(preds))
        top_label = CLASS_NAMES[top_idx]
        top_conf  = float(preds[top_idx]) * 100
        top_emoji = CLASS_EMOJIS.get(top_label, "🔍")

        # Winner card
        st.markdown(f"""
        <div class="card" style="border-color:rgba(0,245,196,.4); text-align:center;">
          <div style="font-size:3.5rem">{top_emoji}</div>
          <div style="font-size:1.6rem; font-weight:800; font-family:'Syne',sans-serif; margin:.3rem 0">
            {top_label.upper()}
          </div>
          <div style="font-size:1.1rem; color:#00f5c4; font-family:'Space Mono',monospace">
            {top_conf:.1f}% confidence
          </div>
        </div>
        """, unsafe_allow_html=True)

        # All class probabilities
        st.markdown("#### All class probabilities")
        sorted_idx = np.argsort(preds)[::-1]
        for idx in sorted_idx:
            label = CLASS_NAMES[idx]
            prob  = float(preds[idx]) * 100
            emoji = CLASS_EMOJIS.get(label, "")
            bar_color = "var(--accent)" if idx == top_idx else "var(--border)"
            st.markdown(f"""
            <div style="margin-bottom:.6rem">
              <div class="pred-label">
                <span>{emoji} {label}</span>
                <span>{prob:.1f}%</span>
              </div>
              <div class="pred-bar-bg">
                <div class="pred-bar-fill" style="width:{prob}%; background:{'linear-gradient(90deg,#00f5c4,#7c6af5)' if idx==top_idx else '#2a2e3a'}"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem 2rem;">
          <div style="font-size:3rem; margin-bottom:1rem">🖼️</div>
          <div style="color:var(--muted); font-family:'Space Mono',monospace; font-size:.85rem">
            Upload an image on the left<br>and click <strong>Run Inference</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# BOTTOM – HOW IT WORKS
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚙️ How the CNN Works")

c1, c2, c3, c4 = st.columns(4)
steps = [
    ("1️⃣", "Input", "32×32 RGB image fed into the network as a tensor of pixel values."),
    ("2️⃣", "Convolution", "Learnable filters slide over the image, detecting edges, textures, and patterns."),
    ("3️⃣", "Pooling", "Max-pooling reduces spatial dimensions, keeping dominant features and cutting computation."),
    ("4️⃣", "Classification", "Flattened features pass through Dense layers; Softmax outputs class probabilities."),
]
for col, (num, title, desc) in zip([c1, c2, c3, c4], steps):
    with col:
        st.markdown(f"""
        <div class="card" style="height:160px">
          <div style="font-size:1.6rem">{num}</div>
          <div style="font-weight:700; font-size:1rem; margin:.3rem 0">{title}</div>
          <div style="color:var(--muted); font-size:.82rem; line-height:1.5">{desc}</div>
        </div>
        """, unsafe_allow_html=True)