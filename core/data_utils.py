import os
import torch
import pandas as pd
from torch.utils.data import Dataset, WeightedRandomSampler

class ForensicDataset(Dataset):
    def __init__(self, csv_file, scan_dir):
        self.data = pd.read_csv(csv_file, sep=';', encoding='latin-1')
        self.data.columns = self.data.columns.str.strip().str.lower()
        self.data.rename(columns={'age, years': 'age'}, inplace=True)
        self.data = self.data.dropna(subset=['age', 'sex'])
        self.scan_dir = scan_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        patient_id = str(row['id']).strip().zfill(3) 
        file_name = f"patient_{patient_id}_128.pt"
        file_path = os.path.join(self.scan_dir, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"\n❌ MISSING TENSOR: Could not find '{file_path}'\n")
            
        scan_tensor = torch.load(file_path, weights_only=True).to(torch.float32)
        if scan_tensor.dim() == 3:
            scan_tensor = scan_tensor.unsqueeze(0)
            
        # 🚨 RESTORED: HU THRESHOLDING (+300 to +400)
        # Acts as a digital eraser to isolate cranial bone from soft tissue/noise
        bone_mask = (scan_tensor >= 300) & (scan_tensor <= 400)
        scan_tensor = scan_tensor * bone_mask.float()
            
        # 3-View Orthogonal Projections (Axial, Coronal, Sagittal)
        vol = scan_tensor.squeeze(0) 
        axial_proj = torch.max(vol, dim=0)[0]    # Depth dimension
        coronal_proj = torch.max(vol, dim=1)[0]  # Height dimension
        sagittal_proj = torch.max(vol, dim=2)[0] # Width dimension
        
        multi_view_tensor = torch.stack([axial_proj, coronal_proj, sagittal_proj], dim=0)
        
        v_min, v_max = multi_view_tensor.min(), multi_view_tensor.max()
        multi_view_tensor = (multi_view_tensor - v_min) / (v_max - v_min + 1e-8)
        
        age = torch.tensor(float(row['age']) / 100.0, dtype=torch.float32) 
        sex_val = 1.0 if str(row['sex']).strip().upper() == 'M' else 0.0
        sex = torch.tensor(sex_val, dtype=torch.float32)
        
        return multi_view_tensor, age, sex

def get_balanced_sampler(csv_file):
    df = pd.read_csv(csv_file, sep=';', encoding='latin-1')
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={'age, years': 'age'}, inplace=True)
    df = df.dropna(subset=['age', 'sex'])
    
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    
    df['age_bin'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Y', 'M', 'S'])
    df['demo_class'] = df['age_bin'].astype(str) + "_" + df['sex'].astype(str).str.strip().str.upper()
    
    class_counts = df['demo_class'].value_counts().to_dict()
    weights = [1.0 / class_counts[demo] for demo in df['demo_class']]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )
    return sampler