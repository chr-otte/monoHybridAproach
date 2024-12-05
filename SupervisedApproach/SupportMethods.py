import torch
from PIL import Image
import numpy as np

def load_kitti_image(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = np.asarray(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, H, W]
    return img_tensor