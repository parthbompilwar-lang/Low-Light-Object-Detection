"""
Inference test script for Zero-DCE enhancement model.
Compares original dark image with Output 1: Enhanced image side-by-side using Pillow.
No OpenCV required.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from src.enhancement.model import ZeroDCE

def enhance_image(
    image_path,
    weights_path="weights/enhancer_zero_dce.pth",
    output_path="demo_output1_comparison.png",
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"--> Loading model from {weights_path} onto {device}...")
    model = ZeroDCE().to(device)
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("--> Weights loaded successfully.")
    else:
        print(f"[!] Warning: weights file not found at {weights_path}. Running with initialized weights.")
        
    model.eval()

    # Load image with Pillow
    print(f"--> Opening input image: {image_path}")
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    # Convert to tensor [0, 1]
    img_tensor = TF.to_tensor(orig_img).unsqueeze(0).to(device)

    # Enhance
    with torch.no_grad():
        enhanced_tensor, _ = model(img_tensor)

    # Convert back to Pillow image
    enhanced_tensor = enhanced_tensor.squeeze(0).cpu()
    enhanced_img = TF.to_pil_image(enhanced_tensor)

    # Create side-by-side comparison image
    padding = 20
    header_height = 40
    comp_w = orig_w * 2 + padding * 3
    comp_h = orig_h + header_height + padding * 2

    comparison = Image.new("RGB", (comp_w, comp_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(comparison)

    # Paste images
    comparison.paste(orig_img, (padding, header_height + padding))
    comparison.paste(enhanced_img, (orig_w + padding * 2, header_height + padding))

    # Add text headers
    draw.text((padding + 10, 15), "Original Dark Image", fill=(220, 220, 220))
    draw.text((orig_w + padding * 2 + 10, 15), "Output 1: Zero-DCE Enhanced Image", fill=(100, 255, 150))

    # Save output
    comparison.save(output_path)
    print(f"--> Comparison image saved to: {output_path}")

    # Brightness stats
    import numpy as np
    orig_arr = np.array(orig_img, dtype=float)
    enh_arr = np.array(enhanced_img, dtype=float)
    print(f"--> Original Mean Brightness: {orig_arr.mean():.2f} / 255.0")
    print(f"--> Enhanced Mean Brightness: {enh_arr.mean():.2f} / 255.0 (+{(enh_arr.mean() - orig_arr.mean()):.2f})")

    return enhanced_img

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="", help="Path to test dark image")
    parser.add_argument("--weights", type=str, default="weights/enhancer_zero_dce.pth", help="Path to weights")
    args = parser.parse_args()

    # Default to an ExDark sample if not provided
    test_img = args.image
    if not test_img:
        test_img = os.path.join("ExDark", "Car", os.listdir("ExDark/Car")[0])

    enhance_image(test_img, weights_path=args.weights)
