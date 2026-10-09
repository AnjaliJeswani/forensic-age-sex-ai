import pandas as pd
from pathlib import Path

# Your exact paths
DATASET_ROOT = r"C:\Users\hp\Downloads\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI\IPCTI\dataset"
CV_ROOT = r"C:\Users\hp\Downloads\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI\IPCTI\cross-validation"

dataset_path = Path(DATASET_ROOT)
df = pd.read_csv(Path(CV_ROOT) / 'cv_1' / 'split_1' / 'train.csv')

# 1. What's actually on your drive?
all_folders = [f.name for f in dataset_path.iterdir() if f.is_dir()]
print(f"=== DISK INFO ===")
print(f"Total patient folders on your computer: {len(all_folders)}")
print(f"First 5 folder names: {all_folders[:5]}")

# 2. What's missing?
missing_ids = []
for _, row in df.iterrows():
    raw_id = str(row['id'])
    padded_id = raw_id.zfill(4)
    
    if raw_id not in all_folders and padded_id not in all_folders:
        missing_ids.append(raw_id)

print(f"\n=== CSV vs DISK ===")
print(f"Number of CSV patients missing from your computer: {len(missing_ids)}")
if len(missing_ids) > 0:
    print(f"First 10 missing IDs from CSV: {missing_ids[:10]}")
else:
    print("All folders exist! The issue is missing images INSIDE the folders.")

    # 3. If folders exist, let's look inside one failing folder
    print("\n=== CHECKING INSIDE A FAILING FOLDER ===")
    IMAGE_TYPES = ["coronal_11", "coronal_21", "sagittal_11", "sagittal_21"]
    for _, row in df.iterrows():
        raw_id = str(row['id'])
        folder = dataset_path / raw_id if (dataset_path / raw_id).exists() else dataset_path / raw_id.zfill(4)
        
        # Check if all 4 images are there
        found_all = True
        for img_type in IMAGE_TYPES:
            matches = list(folder.glob(f"*{img_type}*.jpg")) + list(folder.glob(f"*{img_type}*.JPG"))
            if not matches:
                found_all = False
                break
        
        if not found_all:
            print(f"Patient {folder.name} is missing images. Here is what is ACTUALLY inside {folder.name}:")
            for f in folder.iterdir():
                print(f"  - {f.name}")
            break