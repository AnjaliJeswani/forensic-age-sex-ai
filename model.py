import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ── 1. Define the CBAM Modules ────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Max and Avg pooling across the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)  # Focus on WHAT is important
        x = x * self.sa(x)  # Focus on WHERE it is important
        return x

# ── 2. Define the Upgraded Multi-Task Engine ──────────────────────────────────

class DentalMultiTaskNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DentalMultiTaskNet, self).__init__()
        
        # Load the EfficientNet-B3 backbone
        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
            self.backbone = efficientnet_b3(weights=weights)
        else:
            self.backbone = efficientnet_b3(weights=None)
            
        in_features = self.backbone.classifier[1].in_features # 1536
        
        # Isolate the convolutional features (bypassing the default pool/flatten)
        self.features = self.backbone.features
        
        # Inject our CBAM filter specifically mapped to EfficientNet's 1536 channels
        self.cbam = CBAM(in_planes=in_features)
        
        # Add our own pooling layer to compress the noise-suppressed features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Age Prediction Head (Same as Phase-1)
        self.age_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Gender Prediction Head (Same as Phase-1)
        self.gender_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, 4_images, 3_channels, 224_height, 224_width]
        batch_size, num_views, c, h, w = x.shape
        
        # Push all images through independently: [batch_size * 4, 3, 224, 224]
        x = x.view(-1, c, h, w)
        
        # 1. Extract 2D Spatial Features (Shape: [B*4, 1536, 7, 7])
        f_maps = self.features(x)
        
        # 2. Apply CBAM for Noise Suppression & Artifact Filtering
        f_maps_attended = self.cbam(f_maps)
        
        # 3. Pool and Flatten (Shape: [B*4, 1536])
        pooled = self.global_pool(f_maps_attended)
        features_1d = pooled.view(pooled.size(0), -1)
        
        # 4. Reshape back to group the 4 views per patient
        features_grouped = features_1d.view(batch_size, num_views, -1)
        
        # 5. Master Feature Fusion via Mean Pooling
        pooled_features = features_grouped.mean(dim=1)
        
        # Pass the noise-suppressed master vector to the heads
        age_pred = self.age_head(pooled_features)
        gender_pred = self.gender_head(pooled_features)
        
        return {
            'age': age_pred.squeeze(-1),      
            'gender': gender_pred.squeeze(-1) 
        }

# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initializing Phase-II Dual-Fusion EfficientNet-B3 with CBAM...")
    model = DentalMultiTaskNet(pretrained=True)
    
    print("Creating dummy patient data (Batch Size: 2, Images: 4, Channels: 3, 224x224)...")
    dummy_input = torch.randn(2, 4, 3, 224, 224) 
    
    print("Passing data through the model...")
    outputs = model(dummy_input)
    
    print("\n=== Phase-II Model Output ===")
    print(f"Age predictions shape: {outputs['age'].shape} -> Values: {outputs['age'].detach().numpy()}")
    print(f"Gender predictions shape: {outputs['gender'].shape} -> Values: {outputs['gender'].detach().numpy()}")
    print("Success! CBAM architecture is fully integrated and ready for training.")