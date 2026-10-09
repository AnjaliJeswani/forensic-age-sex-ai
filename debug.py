import os
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class IPCTIDataset(Dataset):
    """
    Loads all 4 images per patient and returns them as a stacked tensor.
    Labels: age (float, for regression) and gender (0=F, 1=M, for classification)
    """
    IMAGE_TYPES = [
        "coronal_11",
        "coronal_21",
        "sagittal_11",
        "sagittal_21",
    ]

    def __init__(self, csv_path, dataset_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        self.valid_patients = []
        missing = 0

        for _, row in self.df.iterrows():
            raw_id = str(row['id'])
            age = float(row['age'])
            
            # Check for folder without padding (e.g., "679")
            folder = self.dataset_root / raw_id
            patient_id = raw_id
            
            # If not found, check with 4-digit padding (e.g., "0679")
            if not folder.exists():
                folder = self.dataset_root / raw_id.zfill(4)
                patient_id = raw_id.zfill(4)

            if folder.exists():
                imgs = self._find_images(folder, patient_id)
                if imgs is not None:
                    self.valid_patients.append({
                        'id': patient_id,
                        'age': age,
                        'gender': 0 if row['sex'] == 'F' else 1,
                        'images': imgs
                    })
                else:
                    missing += 1
            else:
                missing += 1

        print(f"Loaded {len(self.valid_patients)} patients "
              f"({missing} skipped due to missing images or folders)")

    def _find_images(self, folder, patient_id):
        """Find all 4 images for a patient using wildcards to ignore the age string."""
        paths = []
        for img_type in self.IMAGE_TYPES:
            matches = list(folder.glob(f"{patient_id}_{img_type}_*.jpg")) + \
                      list(folder.glob(f"{patient_id}_{img_type}_*.JPG"))
            
            if not matches:
                return None
            paths.append(matches[0])
        return paths

    def __len__(self):
        return len(self.valid_patients)

    def __getitem__(self, idx):
        patient = self.valid_patients[idx]
        images = []
        for img_path in patient['images']:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images)

        return {
            'images': images,                              
            'age': torch.tensor(patient['age'], dtype=torch.float32), 
            'gender': torch.tensor(patient['gender'], dtype=torch.long), 
            'id': patient['id']
        }

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

if __name__ == "__main__":
    DATASET_ROOT = r"C:\Users\hp\Downloads\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI\IPCTI\dataset"
    CV_ROOT      = r"C:\Users\hp\Downloads\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI\IPCTI\cross-validation"
    
    train_csv = Path(CV_ROOT) / "cv_1" / "split_1" / "train.csv"
    
    dataset = IPCTIDataset(
        csv_path=train_csv,
        dataset_root=DATASET_ROOT,
        transform=get_transforms(train=True)
    )
    
    if len(dataset) > 0:
        print(f"\nDataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"Images shape: {sample['images'].shape}")
        print(f"Age: {sample['age']}")
        print(f"Gender: {sample['gender']} (0=F, 1=M)")
        print(f"Patient ID: {sample['id']}")