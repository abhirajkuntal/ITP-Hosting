# run_model.py

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import gdown

# === Ensure models directory exists ===
os.makedirs('models', exist_ok=True)

# === Define model paths and Google Drive links ===
yolo_path = 'models/yolo11s.pt'
yolo_gdrive_id = '1o53aVuum-ND13jj3XlAQE8tF4Hus9lhP'

hybrid_path = 'models/hybridnets_weights.pth'
hybrid_gdrive_id = '1muXT6Z1dzRZFw57lHgPtg9iDw-t3Fhem'

# === Download models if not present ===
if not os.path.exists(yolo_path):
    print("[INFO] Downloading YOLOv8 model...")
    gdown.download(f'https://drive.google.com/uc?id={yolo_gdrive_id}', yolo_path, quiet=False)

if not os.path.exists(hybrid_path):
    print("[INFO] Downloading HybridNets model...")
    gdown.download(f'https://drive.google.com/uc?id={hybrid_gdrive_id}', hybrid_path, quiet=False)

# === Load YOLOv8 ===
yolo_model = YOLO(yolo_path)

# === Load HybridNets ===
hybrid_model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=False)
hybrid_model.load_state_dict(torch.load(hybrid_path, map_location='cpu'))
hybrid_model.eval()

# === Image transformation for HybridNets ===
transform = transforms.Compose([
    transforms.Resize((384, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Image Processing ===
def process_image(input_path, output_path):
    print(f"[INFO] Starting image processing: {input_path}")

    try:
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"[ERROR] Could not read image: {input_path}")
            return

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        input_tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            seg_output = hybrid_model(input_tensor)[1][0].sigmoid().cpu().numpy()

        seg_mask = (seg_output > 0.3).astype(np.uint8)[0] * 255
        seg_mask_color = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)

        results = yolo_model(frame)
        annotated_frame = results[0].plot()

        seg_mask_color = cv2.resize(seg_mask_color, (annotated_frame.shape[1], annotated_frame.shape[0]))
        if len(seg_mask_color.shape) == 2:
            seg_mask_color = cv2.cvtColor(seg_mask_color, cv2.COLOR_GRAY2BGR)

        final_frame = cv2.addWeighted(annotated_frame, 0.7, seg_mask_color, 0.3, 0)
        cv2.imwrite(output_path, final_frame)

        print(f"Image processed and saved: {output_path}")

    except Exception as e:
        print(f"[ERROR] Exception during image processing: {str(e)}")

# === Video Processing ===
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

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
            input_tensor = transform(pil_img).unsqueeze(0)

            with torch.no_grad():
                seg_output = hybrid_model(input_tensor)[1][0].sigmoid().cpu().numpy()

            seg_mask = (seg_output > 0.3).astype(np.uint8)[0] * 255
            seg_mask_color = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)

            results = yolo_model(frame)
            annotated_frame = results[0].plot()

            annotated_frame = cv2.resize(annotated_frame, (width, height))
            seg_mask_color = cv2.resize(seg_mask_color, (width, height))

            final_frame = cv2.addWeighted(annotated_frame, 0.7, seg_mask_color, 0.3, 0)
            out.write(final_frame)

        cap.release()
        out.release()
        print(f"Video processed and saved: {output_path}")

    except Exception as e:
        print(f"[ERROR] Exception during video processing: {str(e)}")
