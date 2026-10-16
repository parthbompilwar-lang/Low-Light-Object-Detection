"""
Visualization utilities for Object Detection using Pillow.
Renders bounding boxes, class labels, and confidence scores. Strictly avoids OpenCV.
"""

from PIL import Image, ImageDraw, ImageFont
import torch

# Distinct high-visibility color palette for 12 ExDark classes
CLASS_COLORS = {
    1:  (0, 255, 128),    # Bicycle - Mint Green
    2:  (0, 191, 255),    # Boat - Deep Sky Blue
    3:  (255, 215, 0),    # Bottle - Gold
    4:  (255, 105, 180),  # Bus - Hot Pink
    5:  (30, 144, 255),   # Car - Dodger Blue
    6:  (255, 140, 0),    # Cat - Dark Orange
    7:  (186, 85, 211),   # Chair - Medium Orchid
    8:  (50, 205, 50),    # Cup - Lime Green
    9:  (255, 69, 0),     # Dog - Orange Red
    10: (0, 250, 154),    # Motorbike - Medium Spring Green
    11: (255, 20, 147),   # People - Deep Pink
    12: (218, 165, 32),   # Table - Goldenrod
}

import torchvision.ops as ops

def draw_detections(
    pil_image,
    boxes,
    labels,
    scores,
    class_names,
    score_threshold=0.5,
    box_width=3,
    iou_threshold=0.30
):
    """
    Overlays bounding boxes, labels, and confidence scores onto a PIL Image.
    Applies Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes.
    """
    output_image = pil_image.copy()
    draw = ImageDraw.Draw(output_image)
    
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.int64)
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
        
    detected_objects = []
    
    # 1. Filter by confidence score threshold
    mask = scores >= score_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    if len(boxes) == 0:
        return output_image, detected_objects
        
    # 2. Apply Class-aware Non-Maximum Suppression (NMS) to remove duplicate overlapping boxes
    keep = ops.batched_nms(boxes.float(), scores.float(), labels.long(), iou_threshold=iou_threshold)
    boxes = boxes[keep].detach().cpu().numpy()
    labels = labels[keep].detach().cpu().numpy()
    scores = scores[keep].detach().cpu().numpy()
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    for i in range(len(boxes)):
        score = float(scores[i])
        if score < score_threshold:
            continue
            
        box = boxes[i]
        label_id = int(labels[i])
        class_name = class_names[label_id] if label_id < len(class_names) else f"Class {label_id}"
        color = CLASS_COLORS.get(label_id, (0, 255, 0))
        
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        
        # Prepare label text
        text = f"{class_name}: {score * 100:.1f}%"
        
        # Calculate text bounding box
        if hasattr(draw, "textbbox") and font:
            tb = draw.textbbox((x1, y1), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        else:
            tw, th = len(text) * 6, 12
            
        # Draw background badge for label readability
        badge_y1 = max(0, y1 - th - 4)
        badge_y2 = badge_y1 + th + 4
        badge_x2 = x1 + tw + 6
        
        draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=color)
        draw.text((x1 + 3, badge_y1 + 2), text, fill=(0, 0, 0), font=font)
        
        detected_objects.append({
            "class_id": label_id,
            "class_name": class_name,
            "confidence": score,
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        })
        
    return output_image, detected_objects
