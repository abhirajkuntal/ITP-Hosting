# run_model.py

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# Load YOLOv8 model from models folder
yolo_model = YOLO('models/yolo11s.pt')

# Load HybridNets model for segmentation
hybrid_model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=False)
hybrid_model.load_state_dict(torch.load('models/hybridnets_weights.pth', map_location='cpu'))
hybrid_model.eval()

# Transformation for HybridNets input
transform = transforms.Compose([
    transforms.Resize((384, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def process_image(input_path, output_path):
    print(f"[INFO] Starting image processing: {input_path}")

    try:
        # Load image
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"[ERROR] Could not read image: {input_path}")
            return

        # Convert to RGB for model
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        # Transform for HybridNets
        input_tensor = transform(pil_img).unsqueeze(0)

        # Run HybridNets for segmentation
        with torch.no_grad():
            seg_output = hybrid_model(input_tensor)[1][0].sigmoid().cpu().numpy()

        # Prepare mask
        seg_mask = (seg_output > 0.3).astype(np.uint8)[0] * 255  # Binary mask
        seg_mask_color = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)

        # Run YOLO detection
        results = yolo_model(frame)
        annotated_frame = results[0].plot()

        # Resize seg_mask_color to match YOLO output frame size
        seg_mask_color = cv2.resize(seg_mask_color, (annotated_frame.shape[1], annotated_frame.shape[0]))
        
        if len(seg_mask_color.shape) == 2:
            seg_mask_color = cv2.cvtColor(seg_mask_color, cv2.COLOR_GRAY2BGR)        

        # Combine YOLO + Segmentation
        final_frame = cv2.addWeighted(annotated_frame, 0.7, seg_mask_color, 0.3, 0)

        # Save result
        cv2.imwrite(output_path, final_frame)
        print(f"Image processed and saved: {output_path}")
    except Exception as e:
        print(f"[ERROR] Exception during image processing: {str(e)}")

def process_video(input_path, output_path):
    print(f"[INFO] Starting video processing: {input_path}")
    try:
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0 
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"[INFO] Processing frame {frame_count}")

            # Convert to RGB for HybridNets
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)

            input_tensor = transform(pil_img).unsqueeze(0)

            # HybridNets segmentation
            with torch.no_grad():
                seg_output = hybrid_model(input_tensor)[1][0].sigmoid().cpu().numpy()

            seg_mask = (seg_output > 0.3).astype(np.uint8)[0] * 255
            seg_mask_color = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)

            # YOLO detection
            results = yolo_model(frame)
            annotated_frame = results[0].plot()

            # Resize both to match VideoWriter size (width, height)
            annotated_frame = cv2.resize(annotated_frame, (width, height))
            seg_mask_color = cv2.resize(seg_mask_color, (width, height))

            # Blend
            final_frame = cv2.addWeighted(annotated_frame, 0.7, seg_mask_color, 0.3, 0)

            # Write frame to output video
            out.write(final_frame)

        cap.release()
        out.release()
        print(f"Video processed and saved: {output_path}")

    except Exception as e:
        print(f"[ERROR] Exception during video processing: {str(e)}")
       