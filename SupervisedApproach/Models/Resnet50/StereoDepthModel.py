import os

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
    def __init__(self, baseline=0.54, focal_length=721.5377, save_dir="precomputed"):
        super().__init__()
        self.baseline = baseline
        self.focal_length = focal_length
        self.save_dir = save_dir

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

    def compute_disparity_opencv(self, left_imgs, right_imgs):
        """
        Computes or loads disparity for a batch of stereo images.
        Args:
            left_imgs: Tensor of shape [B, C, H, W]
            right_imgs: Tensor of shape [B, C, H, W]
            save_dir: Directory to save/load precomputed disparity maps (optional)
        Returns:
            batched_disparity: Tensor of shape [B, 1, H, W]
        """
        batch_size = left_imgs.size(0)
        all_disparities = []

        for b in range(batch_size):
            # Define unique file name for this sample
            save_path = None

            # If precomputed file exists, load it
            if save_path and os.path.exists(save_path):
                disparity = torch.load(save_path)['disparity']
            else:
                # Extract individual images from the batch
                left_single = left_imgs[b]
                right_single = right_imgs[b]

                # Convert to numpy and preprocess
                left_np = (left_single.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                right_np = (right_single.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                left_processed = self.preprocess_image(left_np)
                right_processed = self.preprocess_image(right_np)

                # Compute disparity
                disparity = self.stereo.compute(left_processed, right_processed).astype('float32') / 16.0
                mask = (disparity <= 0).astype(np.uint8)
                disparity = cv2.inpaint(disparity, mask, 3, cv2.INPAINT_TELEA)
                disparity = cv2.fastNlMeansDenoising(disparity.astype(np.uint8)).astype(np.float32)

                # Convert to tensor
                disparity = torch.from_numpy(disparity).unsqueeze(0)  # Add channel dimension

                # Save to disk if save_path is provided
                if save_path:
                    torch.save({'disparity': disparity}, save_path)

            all_disparities.append(disparity)

        # Stack all disparities into a batched tensor
        batched_disparity = torch.stack(all_disparities, dim=0).to(left_imgs.device)
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

    def forward(self, left_imgs, right_imgs):
        """
        Forward pass of the stereo model.
        Args:
            left_imgs: Tensor of shape [B, C, H, W] (batch of left images)
            right_imgs: Tensor of shape [B, C, H, W] (batch of right images)
        Returns:
            dict with:
                - depth: Tensor of shape [B, 1, H, W]
                - confidence: Tensor of shape [B, 1, H, W]
                - disparity: Tensor of shape [B, 1, H, W]
        """
        # Compute or load disparity maps
        disparity = self.compute_disparity_opencv(left_imgs, right_imgs)

        # Compute depth and confidence maps
        depth = self.compute_depth_from_disparity(disparity)
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
    left_img = load_image(r'C:\Github\monoHybridAproach\images\left\0000000000.jpg')
    right_img = load_image(r'C:\Github\monoHybridAproach\images\right\0000000000.jpg')

    # Process images
    outputs = stereo_model(left_img, right_img)

    # Visualize results
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.figure(figsize=(20, 5))

    # Left Image
    plt.subplot(1, 4, 1)
    plt.imshow(left_img[0].permute(1, 2, 0).cpu().numpy())
    plt.title('Left Image')
    plt.axis('off')

    # Right Image
    plt.subplot(1, 4, 2)
    plt.imshow(right_img[0].permute(1, 2, 0).cpu().numpy())
    plt.title('Right Image')
    plt.axis('off')


    # Disparity Map with colorbar
    ax4 = plt.subplot(1, 4, 3)
    im4 = ax4.imshow(outputs['disparity'][0, 0].cpu().numpy(), cmap='jet')
    ax4.set_title('Disparity Map')
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im4, cax=cax4)
    # Depth Map with colorbar
    ax3 = plt.subplot(1, 4, 4)
    im3 = ax3.imshow(outputs['depth'][0, 0].cpu().numpy(), cmap='magma')
    ax3.set_title('Depth Map')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)

    plt.tight_layout()
    plt.show()

    # Verify depth results
    stereo_model.verify_depth_map(outputs['depth'])
