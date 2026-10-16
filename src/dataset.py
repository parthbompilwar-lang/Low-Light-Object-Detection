"""
ExDark Dataset Loader for PyTorch and Pillow
Loads low-light images and multi-object bounding boxes using official splits.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

CLASS_NAMES = [
    "__background__",  # 0
    "Bicycle",         # 1
    "Boat",            # 2
    "Bottle",          # 3
    "Bus",             # 4
    "Car",             # 5
    "Cat",             # 6
    "Chair",           # 7
    "Cup",             # 8
    "Dog",             # 9
    "Motorbike",       # 10
    "People",          # 11
    "Table"            # 12
]

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ExDark annotation class name normalization
NAME_MAP = {
    "bicycle": 1,
    "boat": 2,
    "bottle": 3,
    "bus": 4,
    "car": 5,
    "cat": 6,
    "chair": 7,
    "cup": 8,
    "dog": 9,
    "motorbike": 10,
    "motorcycle": 10,
    "people": 11,
    "person": 11,
    "table": 12,
}

class ExDarkDataset(Dataset):
    def __init__(self, root_dir=".", split="train", transform=None, is_enhancement_only=False, filter_empty=False, max_dim=640):
        """
        Args:
            root_dir (str): Root directory of the repository.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied.
            is_enhancement_only (bool): If True, returns only image tensor for Zero-DCE training.
            filter_empty (bool): If True, filters out samples with zero annotated bounding boxes.
            max_dim (int, optional): Safe resolution clamp for images to prevent GPU OOM.
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split.lower()
        self.transform = transform
        self.is_enhancement_only = is_enhancement_only
        self.filter_empty = filter_empty
        self.max_dim = max_dim
        
        split_map = {"train": "1", "val": "2", "test": "3"}
        if self.split not in split_map:
            raise ValueError(f"Invalid split: {split}. Choose from 'train', 'val', or 'test'.")
        target_split_id = split_map[self.split]
        
        split_file = os.path.join(root_dir, "Groundtruth", "imageclasslist.txt")
        self.samples = []
        
        # Build image location index across class folders
        img_location_cache = {}
        for cname in CLASS_NAMES[1:]:
            cdir = os.path.join(root_dir, "ExDark", cname)
            if os.path.exists(cdir):
                for f in os.listdir(cdir):
                    img_location_cache[f] = (cname, os.path.join(cdir, f))

        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Name"):
                    continue
                tokens = line.split()
                if len(tokens) >= 5:
                    fname, main_class_id, light, in_out, split_id = tokens[0], tokens[1], tokens[2], tokens[3], tokens[4]
                    if split_id == target_split_id:
                        if fname in img_location_cache:
                            cname, img_path = img_location_cache[fname]
                            anno_path = os.path.join(root_dir, "ExDark_Annno", cname, fname + ".txt")
                            
                            # Check empty annotations if filtering requested
                            if self.filter_empty and os.path.exists(anno_path):
                                with open(anno_path, "r") as af:
                                    has_valid_box = any(
                                        not line.strip().startswith("%") and len(line.strip().split()) >= 5
                                        for line in af if line.strip()
                                    )
                                if not has_valid_box:
                                    continue
                                    
                            self.samples.append({
                                "filename": fname,
                                "class_id": int(main_class_id),
                                "class_name": cname,
                                "light_type": int(light),
                                "in_out": int(in_out),
                                "img_path": img_path,
                                "anno_path": anno_path
                            })

    def __len__(self):
        return len(self.samples)

    def _parse_annotations(self, anno_path, img_width, img_height):
        boxes = []
        labels = []
        
        if os.path.exists(anno_path):
            with open(anno_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("%"):
                        continue
                    tokens = line.split()
                    if len(tokens) >= 5:
                        obj_name = tokens[0].lower()
                        class_id = NAME_MAP.get(obj_name, 0)
                        
                        try:
                            l = float(tokens[1])
                            t = float(tokens[2])
                            w = float(tokens[3])
                            h = float(tokens[4])
                        except ValueError:
                            continue
                            
                        # ExDark bounding box is [left, top, width, height]
                        xmin = max(0.0, l)
                        ymin = max(0.0, t)
                        xmax = min(float(img_width), l + w)
                        ymax = min(float(img_height), t + h)
                        
                        # Only keep valid bounding boxes
                        if xmax > xmin + 1.0 and ymax > ymin + 1.0 and class_id > 0:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(class_id)
                            
        if len(boxes) == 0:
            # Fallback for images without annotated objects
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        return boxes, labels

    def __getitem__(self, idx):
        meta = self.samples[idx]
        
        # Load image via Pillow and ensure standard 3-channel RGB
        with Image.open(meta["img_path"]) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            
            if self.max_dim is not None and max(orig_w, orig_h) > self.max_dim:
                scale = self.max_dim / float(max(orig_w, orig_h))
                new_w = max(1, int(round(orig_w * scale)))
                new_h = max(1, int(round(orig_h * scale)))
                img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                w, h = new_w, new_h
            else:
                scale = 1.0
                w, h = orig_w, orig_h
            
        if self.is_enhancement_only:
            # Returns normalized float tensor in [0, 1] for Zero-DCE training
            tensor_img = TF.to_tensor(img)
            if self.transform is not None:
                tensor_img = self.transform(tensor_img)
            return tensor_img
            
        boxes, labels = self._parse_annotations(meta["anno_path"], orig_w, orig_h)
        
        if scale != 1.0 and len(boxes) > 0:
            boxes[:, 0] = (boxes[:, 0] * (w / float(orig_w))).clamp(min=0.0, max=float(w))
            boxes[:, 2] = (boxes[:, 2] * (w / float(orig_w))).clamp(min=0.0, max=float(w))
            boxes[:, 1] = (boxes[:, 1] * (h / float(orig_h))).clamp(min=0.0, max=float(h))
            boxes[:, 3] = (boxes[:, 3] * (h / float(orig_h))).clamp(min=0.0, max=float(h))
            
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "orig_size": torch.as_tensor([h, w]),
            "filename": meta["filename"]
        }
        
        tensor_img = TF.to_tensor(img)
        if self.transform is not None:
            tensor_img, target = self.transform(tensor_img, target)
            
        return tensor_img, target

def collate_fn(batch):
    """Custom collator for detection batches with variable number of boxes."""
    return tuple(zip(*batch))
