"""
Zero-DCE (Zero-Reference Deep Curve Estimation) Network
Predicts pixel-wise, high-order curve parameter maps to iteratively enhance low-light images.
"""

import torch
import torch.nn as nn

class DCENet(nn.Module):
    """
    DCE-Net: Lightweight CNN predicting parameter maps for the Quadratic Light Enhancement Curve.
    Total parameters: ~79K (extremely fast and lightweight on RTX 4050).
    """
    def __init__(self, num_filters=32, num_iterations=8):
        super().__init__()
        self.num_iterations = num_iterations
        
        # 7-layer symmetrical CNN with skip connections
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.conv5 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(num_filters * 2, 3 * num_iterations, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input low-light image tensor of shape (B, 3, H, W) in [0, 1].
        Returns:
            enhanced_image (torch.Tensor): Enhanced image in [0, 1].
            curve_params (torch.Tensor): Predicted curve parameter maps A of shape (B, 24, H, W).
        """
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        curve_params = self.tanh(self.conv7(torch.cat([x1, x6], dim=1)))
        
        # Apply iterative curve enhancement: LE_n(x) = LE_{n-1}(x) + A_n * LE_{n-1}(x) * (1 - LE_{n-1}(x))
        enhanced = x
        for i in range(self.num_iterations):
            a_n = curve_params[:, i * 3:(i + 1) * 3, :, :]
            enhanced = enhanced + a_n * enhanced * (1.0 - enhanced)
            
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        return enhanced, curve_params

# Alias for standard naming
ZeroDCE = DCENet
