"""
PyTorch Neural Architectures & Classical Baselines for CHB-MIT Seizure Prediction.
Task 11:
  - Baselines: Majority-Class, Logistic Regression, Random Forest, Plain 1D CNN.
  - Deep Models: MSTM (Multi-Scale Temporal Model with Causal Dilated Convolutions),
                 SBAM (Synchronization-Based Multi-Head Spatial Attention),
                 STFM Fusion (Spatio-Temporal Fusion).
  - Ablations: No dilation, single kernel, no attention, single head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CausalConv1d(nn.Module):
    """1D Convolution with causal padding to prevent future leakage."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out


class Plain1DCNN(nn.Module):
    """Baseline 1D CNN without multi-scale kernels or spatial attention."""
    def __init__(self, in_channels=18, n_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (B, 18, 640)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.pool1(h)
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool2(h)
        h = torch.mean(h, dim=-1) # Global average pooling
        logits = self.fc(h)
        return logits


class MSTMBranch(nn.Module):
    """
    Multi-Scale Temporal Model (MSTM):
    Parallel dilated causal convolutions at 3 scales (kernels 3, 5, 7; dilations 1, 2, 4).
    """
    def __init__(self, in_channels=18, out_features=64, use_dilation=True, multiscale=True):
        super().__init__()
        self.multiscale = multiscale
        
        d1 = 1
        d2 = 2 if use_dilation else 1
        d3 = 4 if use_dilation else 1
        
        self.branch1 = CausalConv1d(in_channels, 32, kernel_size=3, dilation=d1)
        if multiscale:
            self.branch2 = CausalConv1d(in_channels, 32, kernel_size=5, dilation=d2)
            self.branch3 = CausalConv1d(in_channels, 32, kernel_size=7, dilation=d3)
            concat_channels = 96
        else:
            self.branch2 = None
            self.branch3 = None
            concat_channels = 32
            
        self.bn = nn.BatchNorm1d(concat_channels)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(concat_channels, out_features)

    def forward(self, x, return_penultimate=False):
        # x: (B, 18, 640)
        b1 = F.relu(self.branch1(x))
        if self.multiscale:
            b2 = F.relu(self.branch2(x))
            b3 = F.relu(self.branch3(x))
            h = torch.cat([b1, b2, b3], dim=1)
        else:
            h = b1
            
        h = self.bn(h)
        h = self.dropout(h)
        pooled = torch.mean(h, dim=-1) # (B, concat_channels)
        penultimate = F.relu(self.fc(pooled))
        return penultimate


class SBAMBranch(nn.Module):
    """
    Synchronization-Based Attention Mechanism (SBAM):
    Permutes channels into tokens to compute multi-head spatial attention across electrode pairs.
    """
    def __init__(self, in_channels=18, seq_len=640, out_features=64, num_heads=4, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.num_heads = num_heads
        
        # Linear projection of temporal signal per channel to embedding dimension d_model=64
        self.proj = nn.Linear(seq_len, 64)
        
        if use_attention:
            self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, batch_first=True)
            self.norm = nn.LayerNorm(64)
            
        self.conv_sbm = nn.Conv1d(18, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32 * 64, out_features)

    def forward(self, x, return_weights=False):
        # x: (B, 18, 640) - 18 electrode channels
        tokens = self.proj(x) # (B, 18, 64)
        attn_weights = None
        
        if self.use_attention:
            attn_out, attn_weights = self.mha(tokens, tokens, tokens)
            tokens = self.norm(tokens + attn_out)
            
        h = F.relu(self.bn(self.conv_sbm(tokens))) # (B, 32, 64)
        h = self.dropout(h)
        penultimate = F.relu(self.fc(h.flatten(start_dim=1))) # (B, out_features)
        
        if return_weights:
            return penultimate, attn_weights
        return penultimate


class STFMFusionModel(nn.Module):
    """
    Spatio-Temporal Fusion Model (STFM):
    Jointly fuses penultimate representations from MSTM and SBAM branches.
    """
    def __init__(self, in_channels=18, seq_len=640, n_classes=1, mstm_dilation=True, mstm_multiscale=True, sbam_heads=4, sbam_attn=True):
        super().__init__()
        self.mstm = MSTMBranch(in_channels=in_channels, out_features=64, use_dilation=mstm_dilation, multiscale=mstm_multiscale)
        self.sbam = SBAMBranch(in_channels=in_channels, seq_len=seq_len, out_features=64, num_heads=sbam_heads, use_attention=sbam_attn)
        
        # Fusion head
        self.fusion_fc1 = nn.Linear(128, 64)
        self.fusion_bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x, return_explanations=False):
        # x: (B, 18, 640)
        feat_mstm = self.mstm(x)
        if return_explanations:
            feat_sbam, attn_weights = self.sbam(x, return_weights=True)
        else:
            feat_sbam = self.sbam(x)
            attn_weights = None
            
        combined = torch.cat([feat_mstm, feat_sbam], dim=1) # (B, 128)
        h = F.relu(self.fusion_bn(self.fusion_fc1(combined)))
        h = self.dropout(h)
        logits = self.classifier(h)
        
        if return_explanations:
            return logits, attn_weights
        return logits


class SingleBranchClassifier(nn.Module):
    """Wraps either MSTM or SBAM as a standalone classifier."""
    def __init__(self, branch, feat_dim=64, n_classes=1):
        super().__init__()
        self.branch = branch
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        h = self.branch(x)
        return self.classifier(h)
