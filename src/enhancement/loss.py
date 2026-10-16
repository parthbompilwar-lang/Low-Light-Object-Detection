"""
Non-Reference Loss Functions for Zero-DCE Low-Light Enhancement
Implements Spatial Consistency, Exposure Control, Color Constancy, and Illumination Smoothness losses.
No paired ground-truth images required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialConsistencyLoss(nn.Module):
    """
    Encourages spatial coherence by preserving the difference between adjacent regions
    across the original and enhanced images.
    """
    def __init__(self, patch_size=4):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        
        # 4 directional kernel filters
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer("weight_left", kernel_left)
        self.register_buffer("weight_right", kernel_right)
        self.register_buffer("weight_up", kernel_up)
        self.register_buffer("weight_down", kernel_down)

    def forward(self, org, enhance):
        # Convert to single-channel luminance proxy via channel mean
        org_mean = torch.mean(org, dim=1, keepdim=True)
        enh_mean = torch.mean(enhance, dim=1, keepdim=True)
        
        org_pool = self.pool(org_mean)
        enh_pool = self.pool(enh_mean)
        
        d_org_l = F.conv2d(org_pool, self.weight_left, padding=1)
        d_org_r = F.conv2d(org_pool, self.weight_right, padding=1)
        d_org_u = F.conv2d(org_pool, self.weight_up, padding=1)
        d_org_d = F.conv2d(org_pool, self.weight_down, padding=1)
        
        d_enh_l = F.conv2d(enh_pool, self.weight_left, padding=1)
        d_enh_r = F.conv2d(enh_pool, self.weight_right, padding=1)
        d_enh_u = F.conv2d(enh_pool, self.weight_up, padding=1)
        d_enh_d = F.conv2d(enh_pool, self.weight_down, padding=1)
        
        loss = (
            torch.pow(d_org_l - d_enh_l, 2) +
            torch.pow(d_org_r - d_enh_r, 2) +
            torch.pow(d_org_u - d_enh_u, 2) +
            torch.pow(d_org_d - d_enh_d, 2)
        )
        return torch.mean(loss)

class ExposureControlLoss(nn.Module):
    """
    Penalizes deviations of local patch intensity from a target exposure level E (default 0.6).
    """
    def __init__(self, patch_size=16, target_exposure=0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        self.target_exposure = target_exposure

    def forward(self, enhance):
        # Average intensity across color channels
        mean_enh = torch.mean(enhance, dim=1, keepdim=True)
        patch_mean = self.pool(mean_enh)
        loss = torch.mean(torch.pow(patch_mean - self.target_exposure, 2))
        return loss

class ColorConstancyLoss(nn.Module):
    """
    Based on the Gray-World assumption: penalizes significant discrepancies
    between the average values of the color channels to prevent color casting.
    """
    def __init__(self):
        super().__init__()

    def forward(self, enhance):
        mean_rgb = torch.mean(enhance, dim=[2, 3])  # Shape: (B, 3)
        mr = mean_rgb[:, 0]
        mg = mean_rgb[:, 1]
        mb = mean_rgb[:, 2]
        
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mg - mb, 2)
        
        loss = torch.mean(torch.sqrt(d_rg + d_rb + d_gb + 1e-6))
        return loss

class IlluminationSmoothnessLoss(nn.Module):
    """
    Total variation loss on the estimated curve parameter maps A
    to preserve monotonic and smooth illumination transitions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, curve_params):
        # curve_params shape: (B, 24, H, W)
        batch_size = curve_params.size(0)
        h_x = curve_params.size(2)
        w_x = curve_params.size(3)
        
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        
        h_tv = torch.pow(curve_params[:, :, 1:, :] - curve_params[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(curve_params[:, :, :, 1:] - curve_params[:, :, :, :-1], 2).sum()
        
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class ZeroDCELoss(nn.Module):
    """
    Composite non-reference loss for training Zero-DCE.
    """
    def __init__(self, w_spa=1.0, w_exp=10.0, w_col=5.0, w_tv=200.0, target_exposure=0.6):
        super().__init__()
        self.spa_loss = SpatialConsistencyLoss()
        self.exp_loss = ExposureControlLoss(target_exposure=target_exposure)
        self.col_loss = ColorConstancyLoss()
        self.tv_loss = IlluminationSmoothnessLoss()
        
        self.w_spa = w_spa
        self.w_exp = w_exp
        self.w_col = w_col
        self.w_tv = w_tv

    def forward(self, org, enhance, curve_params):
        l_spa = self.spa_loss(org, enhance)
        l_exp = self.exp_loss(enhance)
        l_col = self.col_loss(enhance)
        l_tv = self.tv_loss(curve_params)
        
        total_loss = (
            self.w_spa * l_spa +
            self.w_exp * l_exp +
            self.w_col * l_col +
            self.w_tv * l_tv
        )
        
        return total_loss, {
            "spa": l_spa.item(),
            "exp": l_exp.item(),
            "col": l_col.item(),
            "tv": l_tv.item(),
            "total": total_loss.item()
        }
