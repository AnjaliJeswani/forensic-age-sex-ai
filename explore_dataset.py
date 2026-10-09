import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ── Change this to where your IPCTI dataset folder is ──────────────────────────
DATASET_ROOT = r"C:\Users\hp\Downloads\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI\IPCTI"  
# ───────────────────────────────────────────────────────────────────────────────

dataset_dir  = Path(DATASET_ROOT) / "dataset"
cv_dir       = Path(DATASET_ROOT) / "cross-validation"

# ── 1. Load one cross-validation split to get patient metadata ─────────────────
train_csv = cv_dir / "cv_1" / "split_1" / "train.csv"
test_csv  = cv_dir / "cv_1" / "split_1" / "test.csv"

train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)
full_df  = pd.concat([train_df, test_df], ignore_index=True)

print("=== Columns in CSV ===")
print(full_df.columns.tolist())
print("\n=== First 5 rows ===")
print(full_df.head())

print(f"\n=== Total patients: {len(full_df)} ===")
print(f"Gender distribution:\n{full_df['sex'].value_counts()}")
print(f"\nAge stats:\n{full_df['age'].describe().round(1)}")

# ── 2. Check actual image folders ──────────────────────────────────────────────
patient_folders = [f for f in dataset_dir.iterdir() if f.is_dir()]
print(f"\n=== Patient folders found: {len(patient_folders)} ===")

# Look inside one folder to understand structure
sample = patient_folders[0]
print(f"\nSample folder: {sample.name}")
for f in sorted(sample.rglob("*")):
    print(" ", f.relative_to(sample))

# ── 3. Count images per patient ────────────────────────────────────────────────
image_counts = {}
for folder in patient_folders:
    imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG"))
    image_counts[folder.name] = len(imgs)

counts = list(image_counts.values())
print(f"\n=== Images per patient ===")
print(f"Min: {min(counts)}, Max: {max(counts)}, Most common: {max(set(counts), key=counts.count)}")

patients_with_4 = sum(1 for c in counts if c == 4)
print(f"Patients with exactly 4 images: {patients_with_4} / {len(patient_folders)}")

# ── 4. Plot age and gender distribution ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(full_df['age'], bins=20, color='steelblue', edgecolor='white')
axes[0].axvline(full_df['age'].mean(), color='red', linestyle='--', label=f"Mean: {full_df['age'].mean():.1f}")
axes[0].set_title("Age distribution")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")
axes[0].legend()

gender_counts = full_df['sex'].value_counts()
axes[1].bar(gender_counts.index, gender_counts.values,
            color=['#E8866A', '#6A9FE8'], edgecolor='white')
axes[1].set_title("Gender distribution")
axes[1].set_ylabel("Count")
for i, v in enumerate(gender_counts.values):
    axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("dataset_overview.png", dpi=120)
plt.show()
print("\nSaved: dataset_overview.png")