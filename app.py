from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import math
import base64
import pandas as pd
import os
import cv2
import numpy as np
import tempfile
import nibabel as nib
import time

from model import DentalMultiTaskNet
from core.model import DecoupledForensicHead

app = Flask(__name__)

print("Loading AI Models into Memory...")
device = torch.device("cpu") 

dental_model = DentalMultiTaskNet(pretrained=False).to(device)
dental_model.load_state_dict(torch.load("dental_multitask_model.pth", map_location=device, weights_only=True), strict=False)
dental_model.eval() 

skull_model = DecoupledForensicHead().to(device)
skull_model.load_state_dict(torch.load("weights/age_expert_engine_2d.pth", map_location=device, weights_only=True), strict=False)
skull_model.eval()

dental_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_skull_scan(file_path):
    if file_path.endswith('.pt'):
        scan_tensor = torch.load(file_path, map_location='cpu', weights_only=True).to(torch.float32)
        if scan_tensor.dim() == 3:
            scan_tensor = scan_tensor.unsqueeze(0)
    elif file_path.endswith(('.nii', '.nii.gz', '.dcm')):
        if file_path.endswith('.dcm'):
            import pydicom
            dicom_data = pydicom.dcmread(file_path)
            scan_tensor = torch.tensor(dicom_data.pixel_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            nii_img = nib.load(file_path)
            scan_tensor = torch.tensor(nii_img.get_fdata(), dtype=torch.float32)
            if scan_tensor.dim() == 3:
                scan_tensor = scan_tensor.unsqueeze(0)
                
        if scan_tensor.shape[1:] != (128, 128, 128):
            scan_tensor = scan_tensor.unsqueeze(0)
            scan_tensor = F.interpolate(scan_tensor, size=(128, 128, 128), mode='trilinear', align_corners=False)
            scan_tensor = scan_tensor.squeeze(0)
    else:
        raise ValueError("Unsupported format.")

    bone_mask = (scan_tensor >= 300) & (scan_tensor <= 400)
    scan_tensor = scan_tensor * bone_mask.float()
        
    vol = scan_tensor.squeeze(0) 
    axial_proj = torch.max(vol, dim=0)[0]    
    coronal_proj = torch.max(vol, dim=1)[0]  
    sagittal_proj = torch.max(vol, dim=2)[0] 
    
    multi_view_tensor = torch.stack([axial_proj, coronal_proj, sagittal_proj], dim=0)
    v_min, v_max = multi_view_tensor.min(), multi_view_tensor.max()
    multi_view_tensor = (multi_view_tensor - v_min) / (v_max - v_min + 1e-8)
    
    return multi_view_tensor.unsqueeze(0), [axial_proj, coronal_proj, sagittal_proj]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            start_time = time.time()
            
            dental_files = {
                'coronal_11': request.files.get('coronal_11'),
                'coronal_21': request.files.get('coronal_21'),
                'sagittal_11': request.files.get('sagittal_11'),
                'sagittal_21': request.files.get('sagittal_21')
            }
            skull_file = request.files.get('skull_scan')
            
            has_dental = any(f and f.filename != '' for f in dental_files.values())
            has_skull = skull_file and skull_file.filename != ''
            
            if not has_dental and not has_skull:
                return render_template('index.html', error="Please upload Dental scans, a Skull scan, or both.")

            results = {'has_dental': False, 'has_skull': False, 'high_variance': False}

            # ================= DENTAL PIPELINE =================
            if has_dental:
                processed_images = []
                display_images = {}
                
                for key, file in dental_files.items():
                    if file and file.filename != '':
                        img_bytes = file.read()
                        display_images[key] = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
                        file.seek(0)
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        processed_images.append(dental_transform(img))
                
                if len(processed_images) > 0:
                    dental_batch = torch.stack(processed_images).unsqueeze(0).to(device)
                    dental_batch.requires_grad = True 
                    
                    outputs = dental_model(dental_batch)
                    d_age = outputs['age'].item()
                    d_sex_logit = outputs['gender'].item()
                    d_prob = 1.0 / (1.0 + math.exp(-d_sex_logit))
                    
                    results['dental_age'] = round(d_age, 1)
                    results['dental_sex'] = "Male" if d_prob > 0.5 else "Female"
                    results['dental_conf'] = round((d_prob if d_prob > 0.5 else (1 - d_prob)) * 100, 1)
                    results['has_dental'] = True
                    results['dental_originals'] = display_images 

                    heatmap_display = {}
                    dental_model.zero_grad()
                    outputs['age'].backward()
                    raw_saliency = dental_batch.grad.abs().squeeze(0).cpu().numpy()
                    raw_saliency = np.max(raw_saliency, axis=1)
                    
                    img_idx = 0
                    for key, file in dental_files.items():
                        if file and file.filename != '':
                            file.seek(0)
                            img_cv = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                            img_cv = cv2.resize(img_cv, (224, 224))
                            saliency = raw_saliency[img_idx]
                            s_max = np.max(saliency)
                            if s_max > 0: saliency = saliency / s_max
                            saliency = cv2.GaussianBlur(saliency, (15, 15), 0)
                            s_max = np.max(saliency)
                            if s_max > 0: saliency = saliency / s_max
                            
                            # Create heatmap WITH black background masked out for the slider
                            heatmap_color = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
                            blended = cv2.addWeighted(img_cv, 0.4, heatmap_color, 0.6, 0)
                            
                            _, buffer = cv2.imencode('.jpg', blended)
                            heatmap_display[key] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                            img_idx += 1
                    results['dental_heatmaps'] = heatmap_display

            # ================= SKULL PIPELINE =================
            if has_skull:
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, skull_file.filename)
                skull_file.save(temp_path)
                
                skull_tensor, projections = process_skull_scan(temp_path)
                skull_tensor = skull_tensor.to(device)
                
                with torch.no_grad():
                    s_age_norm, s_sex_logits = skull_model(skull_tensor)
                    s_age = s_age_norm.item() * 100.0
                    s_prob = torch.sigmoid(s_sex_logits).item()
                
                os.remove(temp_path) 
                
                results['skull_age'] = round(s_age, 1)
                results['skull_sex'] = "Male" if s_prob >= 0.5 else "Female"
                results['skull_conf'] = round((s_prob if s_prob >= 0.5 else (1 - s_prob)) * 100, 1)
                results['has_skull'] = True

                skull_views = {}
                view_names = ['Axial', 'Coronal', 'Sagittal']
                for idx, proj in enumerate(projections):
                    proj_img = proj.cpu().numpy()
                    proj_norm = cv2.normalize(proj_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    proj_colored = cv2.applyColorMap(proj_norm, cv2.COLORMAP_BONE)
                    _, buffer = cv2.imencode('.jpg', proj_colored)
                    skull_views[view_names[idx]] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                results['skull_views'] = skull_views

            # ================= FUSION & METRICS =================
            if has_dental and has_skull:
                results['consensus_age'] = round((results['dental_age'] + results['skull_age']) / 2, 1)
                results['consensus_sex'] = results['skull_sex'] if results['skull_conf'] > results['dental_conf'] else results['dental_sex']
                # Check for high variance (> 3.0 years difference)
                if abs(results['dental_age'] - results['skull_age']) > 3.0:
                    results['high_variance'] = True
            elif has_dental:
                results['consensus_age'] = results['dental_age']
                results['consensus_sex'] = results['dental_sex']
            else:
                results['consensus_age'] = results['skull_age']
                results['consensus_sex'] = results['skull_sex']

            results['exec_time'] = round((time.time() - start_time) * 1000)

            return render_template('index.html', results=results)
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")
            
    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)