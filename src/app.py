"""
ExDark Low-Light Vision & Object Detection App
Streamlit Web Interface with 2-Output Pipeline:
- Input: Image ONLY (No labels or class hints given)
- Output 1: Clean Zero-DCE Enhanced Image
- Output 2: Labeled Enhanced Image (Model automatically detects objects, boxes, and confidence)
Optimized for 6GB RTX 4050 GPU with automatic memory management & safe resolution limiting.
Strictly avoids OpenCV (uses Pillow and PyTorch).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import random
import base64
from io import BytesIO
import streamlit as st
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

from src.dataset import CLASS_NAMES
from src.enhancement.model import ZeroDCE
from src.detection.model import build_detector, filter_coco_to_exdark
from src.detection.utils import draw_detections

st.set_page_config(
    page_title="Low-Light Object Detection & Enhancement",
    page_icon="🌙",
    layout="wide"
)

# Custom CSS for styling with image hovering effects
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #9E9E9E;
        margin-bottom: 1.5rem;
    }
    .card-header {
        font-weight: 600;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        text-align: center;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    .card-header:hover {
        transform: translateY(-2px);
        filter: brightness(1.15);
    }
    .card-input { background-color: #263238; color: #ECEFF1; border: 1px solid #37474F; }
    .card-out1 { background-color: #1B5E20; color: #E8F5E9; border: 1px solid #2E7D32; }
    .card-out2 { background-color: #E65100; color: #FFF3E0; border: 1px solid #EF6C00; }

    /* Interactive Hover Effect on Output & Input Images */
    [data-testid="stImage"] {
        overflow: visible !important;
    }

    /* Column 1 (Input): Subtle hover lift */
    [data-testid="column"]:nth-of-type(1) [data-testid="stImage"] img {
        border: 2px solid transparent;
        border-radius: 10px;
        transition: transform 0.35s cubic-bezier(0.25, 1, 0.5, 1), box-shadow 0.35s ease, border-color 0.35s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    [data-testid="column"]:nth-of-type(1) [data-testid="stImage"] img:hover {
        transform: translateY(-5px) scale(1.025);
        border-color: #607D8B;
        box-shadow: 0 16px 32px rgba(0, 0, 0, 0.5);
        cursor: pointer;
    }

    /* Column 2 (Output 1: Clean Enhanced): Emerald Glow on Hover */
    [data-testid="column"]:nth-of-type(2) [data-testid="stImage"] img {
        border: 2px solid transparent;
        border-radius: 10px;
        transition: transform 0.35s cubic-bezier(0.25, 1, 0.5, 1), box-shadow 0.35s ease, filter 0.35s ease, border-color 0.35s ease;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
    }
    [data-testid="column"]:nth-of-type(2) [data-testid="stImage"] img:hover {
        transform: translateY(-6px) scale(1.035);
        border-color: #4CAF50;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), 0 0 25px rgba(76, 175, 80, 0.55);
        filter: brightness(1.04);
        cursor: pointer;
    }

    /* Column 3 (Output 2: Labeled Enhanced): Amber Glow & Object Highlight on Hover */
    [data-testid="column"]:nth-of-type(3) [data-testid="stImage"] img {
        border: 2px solid transparent;
        border-radius: 10px;
        transition: transform 0.35s cubic-bezier(0.25, 1, 0.5, 1), box-shadow 0.35s ease, filter 0.35s ease, border-color 0.35s ease;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
    }
    [data-testid="column"]:nth-of-type(3) [data-testid="stImage"] img:hover {
        transform: translateY(-6px) scale(1.035);
        border-color: #FF9800;
        box-shadow: 0 22px 44px rgba(0, 0, 0, 0.65), 0 0 32px rgba(255, 152, 0, 0.65);
        filter: contrast(1.06) brightness(1.03);
        cursor: crosshair;
    }
</style>
""", unsafe_allow_html=True)

def limit_resolution(img, max_dim=1024):
    """
    Safely resizes ultra-high-resolution images (e.g. 4K) to prevent GPU VRAM Out-of-Memory.
    Maintains aspect ratio and visual fidelity.
    """
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img

@st.cache_resource
def load_models():
    """Loads and caches Zero-DCE Enhancer and Faster R-CNN Detector."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Zero-DCE Enhancer
    enhancer_path = "weights/enhancer_zero_dce.pth"
    enhancer = ZeroDCE().to(device)
    if os.path.exists(enhancer_path):
        ckpt = torch.load(enhancer_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        enhancer.load_state_dict(state_dict)
    enhancer.eval()
    
    # 2. Faster R-CNN Detector (pre-trained ResNet-50-FPN mapped to 12 ExDark classes)
    detector_path = "weights/detector_fasterrcnn.pth"
    if os.path.exists(detector_path):
        detector = build_detector(mode="custom", pretrained=False).to(device)
        ckpt = torch.load(detector_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        detector.load_state_dict(state_dict)
        use_remapping = False
    else:
        detector = build_detector(mode="coco_mapped", pretrained=True).to(device)
        use_remapping = True
        
    detector.eval()
    return enhancer, detector, use_remapping, device

@st.cache_data
def get_sample_image_list():
    """Collects an unlabelled list of sample dark images from ExDark."""
    all_imgs = glob.glob("ExDark/*/*.*")
    random.seed(42)
    selected = random.sample(all_imgs, min(30, len(all_imgs)))
    return selected

def main():
    st.markdown('<div class="main-title">🌙 Low-Light Enhancement & Object Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Input: Image ONLY. The model automatically brightens the scene and detects objects.</div>', unsafe_allow_html=True)
    
    # Sidebar: ONLY Image Input and Threshold Settings
    st.sidebar.header("📥 Input Dark Image")
    
    device_name = "NVIDIA GeForce RTX 4050 (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"**Hardware Acceleration**: {device_name}")
    
    input_source = st.sidebar.radio(
        "Choose how to provide the image:",
        ["📤 Upload Your Own Dark Image", "🎲 Select a Dark Image Sample"]
    )
    
    input_pil = None
    
    if input_source == "📤 Upload Your Own Dark Image":
        uploaded_file = st.sidebar.file_uploader(
            "Upload any dark image (JPG, PNG)",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            input_pil = Image.open(uploaded_file).convert("RGB")
    else:
        samples = get_sample_image_list()
        if samples:
            sample_labels = [f"Dark Sample #{i+1}" for i in range(len(samples))]
            chosen_idx = st.sidebar.selectbox("Choose sample:", range(len(samples)), format_func=lambda i: sample_labels[i])
            chosen_path = samples[chosen_idx]
            input_pil = Image.open(chosen_path).convert("RGB")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Detection Sensitivity")
    score_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.30, step=0.05)
    st.sidebar.caption("💡 *Tip: Dark & shadowed objects typically score between 25% – 50% confidence. Lower the slider to detect subtle objects.*")
    box_thickness = st.sidebar.slider("Bounding Box Width", min_value=1, max_value=6, value=3, step=1)

    if input_pil is None:
        st.info("👆 Please upload a dark image or select a sample from the sidebar to begin processing.")
        return

    # Safely limit resolution to avoid CUDA OOM on massive images
    input_pil = limit_resolution(input_pil, max_dim=1024)

    # Load models
    enhancer, detector, use_remapping, device = load_models()
    
    # Run Inference with VRAM management
    with st.spinner("Processing image through Zero-DCE and Faster R-CNN..."):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            img_tensor = TF.to_tensor(input_pil).unsqueeze(0).to(device)
            
            # 1. Output 1: Zero-DCE Enhancement
            with torch.no_grad():
                enhanced_tensor, _ = enhancer(img_tensor)
                
            enhanced_pil = TF.to_pil_image(enhanced_tensor.squeeze(0).cpu())
            
            # 2. Output 2: Faster R-CNN Detection
            with torch.no_grad():
                raw_detections = detector([enhanced_tensor.squeeze(0)])[0]
                if use_remapping:
                    detections = filter_coco_to_exdark(raw_detections)
                else:
                    detections = raw_detections
                
            labeled_pil, detected_objects = draw_detections(
                enhanced_pil,
                detections["boxes"],
                detections["labels"],
                detections["scores"],
                CLASS_NAMES,
                score_threshold=score_threshold,
                box_width=box_thickness
            )
            
        except torch.OutOfMemoryError:
            # Fallback to CPU if GPU runs out of memory
            st.warning("⚠️ High resolution image triggered GPU memory limit. Falling back to CPU processing...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            cpu_device = torch.device("cpu")
            enhancer_cpu = enhancer.to(cpu_device)
            detector_cpu = detector.to(cpu_device)
            
            img_tensor = TF.to_tensor(input_pil).unsqueeze(0).to(cpu_device)
            with torch.no_grad():
                enhanced_tensor, _ = enhancer_cpu(img_tensor)
                enhanced_pil = TF.to_pil_image(enhanced_tensor.squeeze(0))
                raw_detections = detector_cpu([enhanced_tensor.squeeze(0)])[0]
                if use_remapping:
                    detections = filter_coco_to_exdark(raw_detections)
                else:
                    detections = raw_detections
                    
            labeled_pil, detected_objects = draw_detections(
                enhanced_pil,
                detections["boxes"],
                detections["labels"],
                detections["scores"],
                CLASS_NAMES,
                score_threshold=score_threshold,
                box_width=box_thickness
            )
            # Restore models back to GPU
            enhancer.to(device)
            detector.to(device)
            
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Compute brightness statistics
    raw_np = np.array(input_pil, dtype=float)
    enh_np = np.array(enhanced_pil, dtype=float)
    raw_brightness = raw_np.mean()
    enh_brightness = enh_np.mean()
    brightness_gain = enh_brightness - raw_brightness

    # 3-Column Visual Layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card-header card-input">Input: Dark Image</div>', unsafe_allow_html=True)
        st.image(input_pil, width="stretch")
        st.caption(f"Mean Luminance: {raw_brightness:.1f} / 255.0")
        
    with col2:
        st.markdown('<div class="card-header card-out1">Output 1: Clean Enhanced Image</div>', unsafe_allow_html=True)
        st.image(enhanced_pil, width="stretch")
        st.caption(f"Mean Luminance: {enh_brightness:.1f} / 255.0 (+{brightness_gain:.1f})")
        
    with col3:
        st.markdown('<div class="card-header card-out2">Output 2: Labeled Enhanced Image</div>', unsafe_allow_html=True)
        st.image(labeled_pil, width="stretch")
        st.caption(f"Detected: {len(detected_objects)} object(s)")

    st.markdown("---")
    
    # Results Section
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.subheader("📊 Luminance Analysis")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div style="background-color: #1A222D; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #2D3748;">
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 4px;">Input Brightness</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #F1F5F9;">{raw_brightness:.1f}</div>
                <div style="font-size: 0.75rem; color: #64748B;">scale 0-255</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div style="background-color: #1A222D; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #2D3748;">
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 4px;">Output 1 Brightness</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #34D399;">{enh_brightness:.1f}</div>
                <div style="font-size: 0.75rem; color: #10B981;">+{brightness_gain:.1f} boost</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div style="background-color: #1A222D; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #2D3748;">
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 4px;">Objects Discovered</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #FBBF24;">{len(detected_objects)}</div>
                <div style="font-size: 0.75rem; color: #64748B;">threshold &ge; {score_threshold*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
    with res_col2:
        st.subheader("🏷️ Model-Detected Objects")
        if detected_objects:
            table_data = [
                {
                    "#": idx + 1,
                    "Detected Class": obj["class_name"],
                    "Confidence": f"{obj['confidence'] * 100:.1f}%",
                    "Bounding Box [x1, y1, x2, y2]": f"{obj['box']}"
                }
                for idx, obj in enumerate(detected_objects)
            ]
            st.table(table_data)
        else:
            top_score = float(detections["scores"][0].item()) if len(detections["scores"]) > 0 else 0.0
            if top_score > 0:
                top_label_id = int(detections["labels"][0].item())
                top_label_name = CLASS_NAMES[top_label_id] if top_label_id < len(CLASS_NAMES) else "Object"
                st.warning(f"No objects displayed at threshold &ge; {score_threshold * 100:.0f}%. The model detected a candidate **{top_label_name}** with **{top_score * 100:.1f}%** confidence! Lower your Confidence Threshold slider below {top_score * 100:.0f}% to display it.")
            else:
                st.info(f"No objects detected at confidence threshold >= {score_threshold * 100:.0f}%. Try lowering the confidence slider.")

if __name__ == "__main__":
    main()
