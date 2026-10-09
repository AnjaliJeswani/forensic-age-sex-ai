import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_cat))

class DecoupledForensicHead(nn.Module):
    def __init__(self):
        super(DecoupledForensicHead, self).__init__()
        
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.spatial_attention = SpatialAttention()
        
        # 3. Dual Task Heads (Clamped Regression)
        self.age_branch = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Re-adding a small dropout to prevent dead neurons
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # 🚨 CRITICAL: Forces output strictly between 0 and 1
        )
        
        self.sex_branch = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, multi_view_scan):
        features = self.backbone.features(multi_view_scan)
        attended = self.spatial_attention(features)
        pooled = torch.nn.functional.adaptive_avg_pool2d(attended, 1).squeeze(-1).squeeze(-1)
        
        pred_norm_age = self.age_branch(pooled)
        raw_sex_logits = self.sex_branch(pooled)
        return pred_norm_age, raw_sex_logits