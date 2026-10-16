"""
Evaluation and Comparison Script:
Evaluates detector performance on Original Dark Images vs. Zero-DCE Enhanced Images.
Quantifies the detection gain produced by the enhancement branch.
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from collections import Counter

from src.dataset import ExDarkDataset, collate_fn, CLASS_NAMES
from src.detection.model import build_detector, filter_coco_to_exdark
from src.enhancement.model import ZeroDCE

def evaluate_comparison(
    detector_weights="weights/detector_fasterrcnn.pth",
    enhancer_weights="weights/enhancer_zero_dce.pth",
    num_samples=50,
    score_threshold=0.45
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65, flush=True)
    print("  ExDark Detection Benchmark: Original Dark vs. Zero-DCE Enhanced ", flush=True)
    print("=" * 65, flush=True)
    print(f"--> Using device: {device}", flush=True)

    # 1. Load Models
    print(f"--> Loading Detector...", flush=True)
    if os.path.exists(detector_weights):
        detector = build_detector(mode="custom", pretrained=False).to(device)
        ckpt = torch.load(detector_weights, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        detector.load_state_dict(state_dict)
        use_remapping = False
    else:
        print("--> Using ResNet-50-FPN detector mapped to ExDark classes...", flush=True)
        detector = build_detector(mode="coco_mapped", pretrained=True).to(device)
        use_remapping = True
        
    detector.eval()

    print(f"--> Loading Zero-DCE Enhancer...", flush=True)
    enhancer = ZeroDCE().to(device)
    if os.path.exists(enhancer_weights):
        ckpt = torch.load(enhancer_weights, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        enhancer.load_state_dict(state_dict)
    enhancer.eval()

    # 2. Test Dataset (using official Test split)
    test_ds = ExDarkDataset(root_dir=".", split="test")
    indices = list(range(min(num_samples, len(test_ds))))
    test_subset = torch.utils.data.Subset(test_ds, indices)
    loader = DataLoader(test_subset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    print(f"--> Evaluating on {len(test_subset)} ExDark test images at score threshold >= {score_threshold*100:.0f}%...\n", flush=True)

    orig_detections = 0
    enh_detections = 0
    orig_conf_sum = 0.0
    enh_conf_sum = 0.0

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            raw_imgs = [img.to(device) for img in images]
            
            # 1. Detection on Original Dark Images
            raw_preds = detector(raw_imgs)
            for raw_p in raw_preds:
                pred = filter_coco_to_exdark(raw_p) if use_remapping else raw_p
                mask = pred["scores"] >= score_threshold
                boxes = pred["boxes"][mask]
                scores = pred["scores"][mask]
                
                orig_detections += len(boxes)
                orig_conf_sum += scores.sum().item()

            # 2. Detection on Zero-DCE Enhanced Images
            enhanced_imgs = []
            for r_img in raw_imgs:
                e_img, _ = enhancer(r_img.unsqueeze(0))
                enhanced_imgs.append(e_img.squeeze(0))
                
            enh_preds = detector(enhanced_imgs)
            for enh_p in enh_preds:
                pred = filter_coco_to_exdark(enh_p) if use_remapping else enh_p
                mask = pred["scores"] >= score_threshold
                boxes = pred["boxes"][mask]
                scores = pred["scores"][mask]
                
                enh_detections += len(boxes)
                enh_conf_sum += scores.sum().item()

    elapsed = time.time() - start_time
    avg_orig_conf = (orig_conf_sum / max(1, orig_detections)) * 100
    avg_enh_conf = (enh_conf_sum / max(1, enh_detections)) * 100
    diff = enh_detections - orig_detections
    gain_pct = (diff / max(1, orig_detections)) * 100

    print("\n" + "=" * 65, flush=True)
    print("                    COMPARISON RESULTS                     ", flush=True)
    print("=" * 65, flush=True)
    print(f"  Total Images Evaluated : {len(test_subset)}", flush=True)
    print(f"  Confidence Threshold   : >= {score_threshold * 100:.0f}%\n", flush=True)
    print(f"  [Original Dark Images]  Total Detections : {orig_detections:>4} | Avg Conf: {avg_orig_conf:.2f}%", flush=True)
    print(f"  [Enhanced Images]       Total Detections : {enh_detections:>4} | Avg Conf: {avg_enh_conf:.2f}%", flush=True)
    print(f"  --> Detection Gain     : {diff:+d} objects ({gain_pct:+.1f}%)", flush=True)
    print("-" * 65, flush=True)
    print(f"  Evaluation completed in {elapsed:.1f}s on {device}.", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50, help="Number of test samples to evaluate")
    parser.add_argument("--threshold", type=float, default=0.45, help="Confidence threshold")
    args = parser.parse_args()

    evaluate_comparison(num_samples=args.samples, score_threshold=args.threshold)
