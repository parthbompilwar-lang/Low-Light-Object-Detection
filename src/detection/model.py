"""
Faster R-CNN Detector for ExDark Classes.
Supports both:
1. Direct COCO-to-ExDark mapping (instant inference without waiting).
2. Custom fine-tuned head (13 classes).
Strictly avoids OpenCV (uses Pillow and PyTorch).
"""

import torch
import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 13  # 1 background + 12 ExDark classes

# Mapping from COCO categories to ExDark class names and IDs
# 1: Bicycle, 2: Boat, 3: Bottle, 4: Bus, 5: Car, 6: Cat,
# 7: Chair, 8: Cup, 9: Dog, 10: Motorbike, 11: People, 12: Table
COCO_TO_EXDARK = {
    1:  (11, "People"),
    2:  (1,  "Bicycle"),
    3:  (5,  "Car"),
    4:  (10, "Motorbike"),
    6:  (4,  "Bus"),
    9:  (2,  "Boat"),
    17: (6,  "Cat"),
    18: (9,  "Dog"),
    44: (3,  "Bottle"),
    47: (8,  "Cup"),
    62: (7,  "Chair"),
    67: (12, "Table"),
}

def build_detector(num_classes=NUM_CLASSES, pretrained=True, mode="coco_mapped", min_size=480, max_size=640):
    """
    Builds Faster R-CNN detector.
    Args:
        mode: 'coco_mapped' (uses rich COCO pretrained weights filtered to the 12 ExDark classes)
              or 'custom' (13 classes for fine-tuning).
    """
    weights = detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        min_size=min_size,
        max_size=max_size
    )
    
    if mode == "custom":
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    return model

def filter_coco_to_exdark(detections):
    """
    Filters COCO detector predictions down to the 12 ExDark classes and remaps labels.
    """
    boxes = detections["boxes"]
    labels = detections["labels"]
    scores = detections["scores"]
    
    keep_indices = []
    remapped_labels = []
    
    for i in range(len(labels)):
        coco_id = labels[i].item()
        if coco_id in COCO_TO_EXDARK:
            keep_indices.append(i)
            exdark_id, _ = COCO_TO_EXDARK[coco_id]
            remapped_labels.append(exdark_id)
            
    if len(keep_indices) > 0:
        keep = torch.as_tensor(keep_indices, dtype=torch.long, device=boxes.device)
        return {
            "boxes": boxes[keep],
            "labels": torch.as_tensor(remapped_labels, dtype=torch.long, device=boxes.device),
            "scores": scores[keep]
        }
    else:
        return {
            "boxes": torch.zeros((0, 4), device=boxes.device),
            "labels": torch.zeros((0,), dtype=torch.long, device=boxes.device),
            "scores": torch.zeros((0,), device=boxes.device)
        }
