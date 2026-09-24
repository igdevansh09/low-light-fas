import streamlit as st

import numpy as np
import os
import sys
import torch
from PIL import Image

# --- PATH ROUTING ---
# Bind the models to Python's execution path
sys.path.append(os.path.abspath("antispoof_model"))
sys.path.append(os.path.abspath("zero_dce/Zero-DCE_code"))

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import model as dce_model
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Face Anti-Spoofing Pipeline")

# Cache the models so they don't reload every time you click a button
@st.cache_resource
def load_models():
    # Load Baseline Anti-Spoof Model
    orig_dir = os.getcwd()
    os.chdir("antispoof_model")
    spoof_model = AntiSpoofPredict(0)
    cropper = CropImage()
    os.chdir(orig_dir)
    
    # Load Zero-DCE Model
    dce_net = dce_model.enhance_net_nopool()
    dce_net.load_state_dict(torch.load('zero_dce/Zero-DCE_code/snapshots/Epoch99.pth', map_location='cpu'))
    dce_net.eval()
    
    return spoof_model, cropper, dce_net

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_image(image_cv2, dce_net):
    # Convert cv2 (BGR) to RGB for the neural network
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    data_lowlight = (np.asarray(image_rgb)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1).unsqueeze(0)
    
    with torch.no_grad():
        _, enhanced_image, _ = dce_net(data_lowlight)
        
    # Convert tensor back to cv2 image
    enhanced_image = enhanced_image.squeeze(0).permute(1,2,0).numpy()
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    return cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)

def get_spoof_score(image, model_test, cropper):
    model_dir = "antispoof_model/resources/anti_spoof_models"
    image_bbox = model_test.get_bbox(image)
    if sum(image_bbox) == 0:
        return None, 0.0
        
    prediction = np.zeros((1, 3))
    for model_name in os.listdir(model_dir):
        if not model_name.endswith('.pth'): continue
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {"org_img": image, "bbox": image_bbox, "scale": scale, "out_w": w_input, "out_h": h_input, "crop": True}
        if scale is None: param["crop"] = False
        img = cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        
    label = np.argmax(prediction)
    score = prediction[0][label] / 2.0
    return label, score

def display_result(label, score):
    if label is None:
        st.error("No Face Detected (Too Dark)")
    elif label == 1:
        st.success(f"REAL FACE\nConfidence: {score:.4f}")
    else:
        st.error(f"SPOOF DETECTED\nConfidence: {score:.4f}")

# --- UI LAYOUT ---
st.title("Low-Light Face Anti-Spoofing: Zero-DCE Recovery")
st.markdown("Upload an image to see how extreme low-light conditions break legacy security models, and how estimating pixel-wise light curves recovers classification accuracy.")

st.sidebar.header("Pipeline Controls")
uploaded_file = st.sidebar.file_uploader("Upload Test Image", type=["jpg", "jpeg", "png"])
gamma = st.sidebar.slider("Synthetic Degradation (Gamma)", min_value=0.1, max_value=1.0, value=0.1, step=0.1)

if uploaded_file is not None:
    spoof_model, cropper, dce_net = load_models()
    
    # Read image into OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    orig_img = cv2.imdecode(file_bytes, 1)
    
    col1, col2, col3 = st.columns(3)
    
    # 1. BASELINE
    with col1:
        st.subheader("1. Original Image")
        st.image(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        label, score = get_spoof_score(orig_img, spoof_model, cropper)
        display_result(label, score)
        
    # 2. DEGRADATION
    with col2:
        st.subheader(f"2. Degraded (Gamma {gamma})")
        dark_img = adjust_gamma(orig_img, gamma)
        st.image(cv2.cvtColor(dark_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        label, score = get_spoof_score(dark_img, spoof_model, cropper)
        display_result(label, score)
        
    # 3. ENHANCEMENT
    with col3:
        st.subheader("3. Zero-DCE Enhanced")
        # CPU warning based on your 43-second processing logs
        with st.spinner("Recovering light curves (~40s on CPU)..."): 
            enhanced_img = enhance_image(dark_img, dce_net)
        st.image(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        label, score = get_spoof_score(enhanced_img, spoof_model, cropper)
        display_result(label, score)
