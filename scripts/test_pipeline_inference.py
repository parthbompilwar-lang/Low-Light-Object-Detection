"""
Unified 2-Output Inference Pipeline: Zero-DCE Enhancement + Faster R-CNN Detection.
Generates:
- Output 1: Clean Enhanced Image
- Output 2: Labeled Enhanced Image (Bounding boxes, classes, confidence scores)
Strictly avoids OpenCV (uses Pillow and PyTorch).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from src.dataset import CLASS_NAMES
from src.enhancement.model import ZeroDCE
from src.detection.model import build_detector, filter_coco_to_exdark
from src.detection.utils import draw_detections

def run_pipeline(
    image_path,
    enhancer_weights="weights/enhancer_zero_dce.pth",
    detector_weights="weights/detector_fasterrcnn.pth",
    score_threshold=0.45,
    save_dir="outputs",
    device=None
):
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print("=" * 60, flush=True)
    print("      ExDark 2-Output Pipeline: Enhancement + Detection    ", flush=True)
    print("=" * 60, flush=True)
    print(f"--> Using device: {device}", flush=True)
    print(f"--> Input Image: {image_path}", flush=True)

    # 1. Load Input Image via Pillow
    raw_img = Image.open(image_path).convert("RGB")
    raw_w, raw_h = raw_img.size
    img_tensor = TF.to_tensor(raw_img).unsqueeze(0).to(device)

    # 2. Output 1: Zero-DCE Enhancement
    print(f"--> Loading Zero-DCE Enhancer from {enhancer_weights}...", flush=True)
    enhancer = ZeroDCE().to(device)
    if os.path.exists(enhancer_weights):
        ckpt = torch.load(enhancer_weights, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        enhancer.load_state_dict(state_dict)
    enhancer.eval()

    with torch.no_grad():
        enhanced_tensor, _ = enhancer(img_tensor)
        
    enhanced_pil = TF.to_pil_image(enhanced_tensor.squeeze(0).cpu())
    out1_path = os.path.join(save_dir, "output1_enhanced.png")
    enhanced_pil.save(out1_path)
    print(f"--> [SUCCESS] Output 1 (Clean Enhanced Image) saved to: {out1_path}", flush=True)

    # 3. Output 2: Faster R-CNN Object Detection on Enhanced Image
    if os.path.exists(detector_weights):
        print(f"--> Loading custom fine-tuned detector from {detector_weights}...", flush=True)
        detector = build_detector(mode="custom", pretrained=False).to(device)
        ckpt = torch.load(detector_weights, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        detector.load_state_dict(state_dict)
        use_remapping = False
    else:
        print("--> Using pretrained ResNet-50-FPN detector mapped to ExDark classes...", flush=True)
        detector = build_detector(mode="coco_mapped", pretrained=True).to(device)
        use_remapping = True
        
    detector.eval()

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
        score_threshold=score_threshold
    )
    
    out2_path = os.path.join(save_dir, "output2_labeled.png")
    labeled_pil.save(out2_path)
    print(f"--> [SUCCESS] Output 2 (Labeled Enhanced Image) saved to: {out2_path}", flush=True)
    print(f"--> Detected {len(detected_objects)} object(s) with confidence >= {score_threshold*100:.0f}%:")
    for obj in detected_objects:
        print(f"    - {obj['class_name']}: {obj['confidence']*100:.1f}% @ {obj['box']}", flush=True)

    # 4. Create Side-by-Side 3-Panel Visual Comparison
    padding = 20
    header_h = 45
    total_w = raw_w * 3 + padding * 4
    total_h = raw_h + header_h + padding * 2

    canvas = Image.new("RGB", (total_w, total_h), color=(25, 28, 35))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(raw_img, (padding, header_h + padding))
    canvas.paste(enhanced_pil, (raw_w + padding * 2, header_h + padding))
    canvas.paste(labeled_pil, (raw_w * 2 + padding * 3, header_h + padding))

    draw.text((padding + 10, 15), "Input: Dark Image", fill=(200, 200, 200))
    draw.text((raw_w + padding * 2 + 10, 15), "Output 1: Zero-DCE Enhanced", fill=(100, 255, 180))
    draw.text((raw_w * 2 + padding * 3 + 10, 15), "Output 2: Labeled Enhanced Image", fill=(255, 215, 0))

    comp_path = os.path.join(save_dir, "demo_two_outputs_comparison.png")
    canvas.save(comp_path)
    print(f"--> [SUCCESS] Full 3-Panel Comparison saved to: {comp_path}", flush=True)

    return enhanced_pil, labeled_pil, detected_objects

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="", help="Path to input dark image")
    parser.add_argument("--threshold", type=float, default=0.45, help="Confidence threshold")
    args = parser.parse_args()

    test_img = args.image
    if not test_img:
        # Default test image from ExDark Car
        test_img = os.path.join("ExDark", "Car", os.listdir("ExDark/Car")[0])

    run_pipeline(test_img, score_threshold=args.threshold)
