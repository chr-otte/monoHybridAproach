import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from SupportMethods import load_kitti_image
import cv2
import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np


class StereoModel(nn.Module):
    def __init__(self, baseline=0.54, focal_length=721.5377):
        super().__init__()
        self.baseline = baseline
        self.focal_length = focal_length

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=256,
            blockSize=11,
            P1=8 * 3 * 11 ** 2,
            P2=32 * 3 * 11 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def preprocess_image(self, img_np):
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        # Apply bilateral filter
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        return gray

    def compute_disparity_opencv(self, left_img, right_img):
        """
        Handle both single and batched inputs
        """
        # Store original batch size
        batch_size = left_img.size(0)

        # Process each image in the batch
        all_disparities = []
        for b in range(batch_size):
            # Extract single image from batch
            left_single = left_img[b]
            right_single = right_img[b]

            # Convert to numpy and ensure proper range [0, 255]
            left_np = (left_single.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')
            right_np = (right_single.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')

            # Preprocess images
            left_processed = self.preprocess_image(left_np)
            right_processed = self.preprocess_image(right_np)

            # Compute disparity map
            disparity = self.stereo.compute(left_processed, right_processed).astype('float32') / 16.0

            # Post-processing
            mask = (disparity <= 0).astype(np.uint8)
            disparity = cv2.inpaint(disparity, mask, 3, cv2.INPAINT_TELEA)
            disparity = cv2.fastNlMeansDenoising(disparity.astype(np.uint8)).astype(np.float32)

            # Convert to tensor and add to list
            disparity_tensor = torch.from_numpy(disparity).unsqueeze(0)
            all_disparities.append(disparity_tensor)

        # Stack all disparities into a batch
        batched_disparity = torch.stack(all_disparities, dim=0).to(left_img.device)
        return batched_disparity

    def compute_confidence(self, disparity):
        """
        Compute confidence map for disparity estimates
        Args:
            disparity: tensor of shape [B, 1, H, W]
        Returns:
            confidence: tensor of shape [B, 1, H, W]
        """
        # Process each batch item separately
        batch_size = disparity.size(0)
        all_confidences = []

        for b in range(batch_size):
            # Extract single disparity map
            disp_single = disparity[b, 0].cpu().numpy()  # Remove batch and channel dims

            # Compute gradients
            grad_x = cv2.Sobel(disp_single, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(disp_single, cv2.CV_32F, 0, 1, ksize=3)

            # Compute gradient magnitude
            gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Create confidence
            conf = 1.0 / (1.0 + gradient_mag)
            conf_tensor = torch.from_numpy(conf).unsqueeze(0)  # Add channel dimension
            all_confidences.append(conf_tensor)

        # Stack all confidences into a batch
        batched_confidence = torch.stack(all_confidences, dim=0).to(disparity.device)

        # Apply valid disparity mask
        batched_confidence = batched_confidence * (disparity > 0).float()

        return batched_confidence


    def compute_depth_from_disparity(self, disparity):
        valid_mask = disparity > 0
        depth = torch.zeros_like(disparity)
        depth[valid_mask] = (self.baseline * self.focal_length) / (disparity[valid_mask])
        depth = torch.clamp(depth, min=1.0, max=100.0)
        return depth

    def forward(self, left_img, right_img):
        # Compute disparity
        disparity = self.compute_disparity_opencv(left_img, right_img)

        # Convert disparity to depth
        depth = self.compute_depth_from_disparity(disparity)

        # Compute confidence
        confidence = self.compute_confidence(disparity)

        return {
            'depth': depth,
            'confidence': confidence,
            'disparity': disparity
        }

    def verify_depth_map(self, depth):
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            print(f"Depth statistics:")
            print(f"Min depth: {valid_depth.min().item():.2f}m")
            print(f"Max depth: {valid_depth.max().item():.2f}m")
            print(f"Mean depth: {valid_depth.mean().item():.2f}m")
            print(f"Valid depth points: {len(valid_depth)}/{depth.numel()}")

def load_image(path):
    img = Image.open(path).convert('RGB')
    # Convert to numpy array and normalize to [0,1]
    img_np = np.array(img, dtype=np.float32) / 255.0
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()
    return img_tensor

if __name__ == "__main__":
    # Initialize model
    stereo_model = StereoModel().cuda()

    # Load images
    left_img = load_image(r'C:\Projekter\pythonProject\HybridDepthEstimation\images\left\0000000000.jpg')
    right_img = load_image(r'C:\Projekter\pythonProject\HybridDepthEstimation\images\right\0000000000.jpg')

    # Process images
    outputs = stereo_model(left_img, right_img)

    # Visualize results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(left_img[0].permute(1, 2, 0).cpu().numpy())
    plt.title('Left Image')

    plt.subplot(132)
    plt.imshow(outputs['depth'][0, 0].cpu().numpy(), cmap='magma')
    plt.colorbar()
    plt.title('Depth Map')

    plt.subplot(133)
    plt.imshow(outputs['disparity'][0, 0].cpu().numpy(), cmap='jet')
    plt.colorbar()
    plt.title('Disparity Map')
    plt.show()

    # Verify depth results
    stereo_model.verify_depth_map(outputs['depth'])