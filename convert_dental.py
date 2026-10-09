import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import glob

def convert_to_tensors():
    INPUT_DIR = r"C:\Users\hp\Downloads\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI\IPCTI"
    OUTPUT_DIR = './dental_tensors'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Starting conversion from {INPUT_DIR} to {OUTPUT_DIR}...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Search for ALL common image formats, including inside subfolders
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    image_paths = []
    
    for ext in extensions:
        # Check main folder
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        # Check subfolders
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, '**', ext), recursive=True))
    
    # Remove any duplicates
    image_paths = list(set(image_paths))
    
    if len(image_paths) == 0:
        print("Still no images found! Open the folder manually and check the exact file extension.")
        return

    print(f"Found {len(image_paths)} images. Processing...")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        patient_id = filename.split('.')[0] 
        
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)
            
            output_path = os.path.join(OUTPUT_DIR, f"{patient_id}.pt")
            torch.save(tensor, output_path)
            print(f"Converted: {filename} -> {patient_id}.pt")
            
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")

    print("All dental tensors generated successfully!")

if __name__ == '__main__':
    convert_to_tensors()