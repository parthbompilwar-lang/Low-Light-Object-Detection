# Model Checkpoints

This folder contains the neural network weights for the 2-stage low-light pipeline:

1. **`enhancer_zero_dce.pth` (~970 KB)**:
   - Trained Zero-DCE CNN model for non-reference illumination enhancement.
   - Tracked in Git repository.

2. **`detector_fasterrcnn.pth` (~315 MB)**:
   - Custom 13-class Faster R-CNN (ResNet-50-FPN) fine-tuned on ExDark.
   - Note: Exceeds GitHub's 100MB file limit. Generated via:
     ```powershell
     python scripts/train_detection.py --epochs 3 --batch_size 4 --lr 0.001
     ```
