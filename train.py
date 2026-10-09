import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
import time

from dataset import IPCTIDataset, get_transforms
from model import DentalMultiTaskNet

def main():
    # ── Paths ─────────────────────────────────────────────────────────────
    DATASET_ROOT = r"C:\Users\hp\Downloads\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI\IPCTI\dataset"
    CV_ROOT      = r"C:\Users\hp\Downloads\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI\IPCTI\cross-validation"
    
    train_csv = Path(CV_ROOT) / "cv_1" / "split_1" / "train.csv"
    
    # ── Load Dataset & Handle Imbalance ─────────────────────────────────────
    print("Loading Dataset...")
    train_dataset = IPCTIDataset(csv_path=train_csv, dataset_root=DATASET_ROOT, transform=get_transforms(train=True))
    
    genders = [p['gender'] for p in train_dataset.valid_patients]
    count_F, count_M = genders.count(0), genders.count(1)
    
    weight_F = 1.0 / count_F
    weight_M = 1.0 / count_M
    sample_weights = [weight_F if g == 0 else weight_M for g in genders]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Batch size 8 - Optimized for your RTX GPU!
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)

    # ── Initialize Model ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing Model on: {device.type.upper()}...")
    
    model = DentalMultiTaskNet(pretrained=True).to(device)
    
    age_criterion = nn.L1Loss() 
    gender_criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ── Full Training Loop ──────────────────────────────────────────────────
    NUM_EPOCHS = 15  
    
    print(f"\nStarting Full Training for {NUM_EPOCHS} Epochs...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to GPU
            images = batch['images'].to(device)
            true_age = batch['age'].to(device)
            true_gender = batch['gender'].to(device).float() 
            
            optimizer.zero_grad()
            
            outputs = model(images)

            loss_age = age_criterion(outputs['age'], true_age)
            loss_gender = gender_criterion(outputs['gender'], true_gender)

            total_loss = (loss_age * 0.2) + (loss_gender * 5.0)
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            # Print update every 20 batches
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] - Current Loss: {total_loss.item():.4f}")
        
        # Epoch Summary
        epoch_loss = running_loss / len(train_loader)
        print(f" Epoch {epoch+1} Completed | Average Loss: {epoch_loss:.4f}")

    # ── Save the Model ──────────────────────────────────────────────────────
    total_time = (time.time() - start_time) / 60
    save_path = "dental_multitask_model.pth"
    torch.save(model.state_dict(), save_path)
    
    print(f"\n Training Complete in {total_time:.1f} minutes!")
    print(f" Model saved successfully to your Desktop as: {save_path}")

if __name__ == "__main__":
    main()