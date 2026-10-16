import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.dataset import ExDarkDataset, collate_fn, CLASS_NAMES

def test_dataset():
    print("Testing ExDarkDataset module...")
    
    # 1. Train split
    train_ds = ExDarkDataset(root_dir=".", split="train")
    print(f"Train Dataset size: {len(train_ds)} (Expected: 3000)")
    assert len(train_ds) == 3000, f"Expected 3000 train images, got {len(train_ds)}"
    
    # 2. Val split
    val_ds = ExDarkDataset(root_dir=".", split="val")
    print(f"Val Dataset size: {len(val_ds)} (Expected: 1800)")
    assert len(val_ds) == 1800, f"Expected 1800 val images, got {len(val_ds)}"
    
    # 3. Test split
    test_ds = ExDarkDataset(root_dir=".", split="test")
    print(f"Test Dataset size: {len(test_ds)} (Expected: 2563)")
    assert len(test_ds) == 2563, f"Expected 2563 test images, got {len(test_ds)}"
    
    # 4. Check DataLoader batching
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    images, targets = next(iter(loader))
    print(f"\nDataLoader batch test (batch_size=4):")
    print(f"Images in batch: {len(images)}")
    for i in range(len(images)):
        img = images[i]
        tgt = targets[i]
        labels = [CLASS_NAMES[lbl.item()] for lbl in tgt["labels"]]
        print(f"  Image {i+1} [{tgt['filename']}]: Shape={tuple(img.shape)}, Boxes={len(tgt['boxes'])}, Classes={labels}")
        
    print("\n[SUCCESS] ExDarkDataset verified and ready for training!")

if __name__ == "__main__":
    test_dataset()
