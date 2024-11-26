import torch
import torch.nn as nn
from torchvision import models
from encoder import SharedResNet18Encoder
import torch
import torchvision.transforms as transforms
from PIL import Image
from KITTIDataset import KITTIRawDataset
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from PIL import Image
from MonocularDepthNet import MonocularDepthNet
from StereoDepthModel import StereoModel
import numpy as np
import cv2
import kornia


def train_hybrid_depth(mono_model, stereo_model, optimizer,
                       frame_t_minus, frame_t, frame_t_plus,
                       stereo_left, stereo_right, debug = False):
    # Stack adjacent frames
    input_frames = torch.cat([frame_t_minus, frame_t, frame_t_plus], dim=1)

    # Get depth prediction from monocular model
    depth_outputs = mono_model(input_frames)

    # Calculate stereo depth at full resolution
    with torch.no_grad():
        stereo_outputs = stereo_model(stereo_left, stereo_right)  # Stereo images are full resolution
        full_res_depth = stereo_outputs['depth']

    # Calculate loss at multiple scales
    total_loss = 0
    for scale in range(4):
        if ("depth", scale) in depth_outputs:
            pred_depth = depth_outputs[("depth", scale)]
            # Resize full resolution stereo depth to match the current scale
            target_depth = F.interpolate(
                full_res_depth,
                size=pred_depth.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            loss = torch.abs(pred_depth - target_depth).mean()
            total_loss += loss


    return total_loss, depth_outputs


def train(mono_model, stereo_model, train_loader, num_epochs=10, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mono_model = mono_model.to(device)
    stereo_model = stereo_model.to(device)
    stereo_model.eval()  # Set stereo model to eval mode
    total_loss = 0
    optimizer = torch.optim.Adam(mono_model.parameters(), lr=learning_rate)
    visualization_freq = 50  # Visualize every 50 batches

    for epoch in range(num_epochs):
        mono_model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Move left to device
            src_images = [img.to(device) for img in batch['src_images']]
            stereo_images = [img.to(device) for img in batch['stereo_images']]

            optimizer.zero_grad()

            loss, depth_outputs = train_hybrid_depth(
                mono_model, stereo_model, optimizer,
                src_images[0],  # t-1 frame
                src_images[1],  # t frame
                src_images[2],  # t+1 frame
                stereo_images[0],  # left image
                stereo_images[1]   # right image
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_loss += loss.item()

            if batch_idx % visualization_freq == 0:
                with torch.no_grad():
                    # Get ground truth depth from stereo model
                    stereo_output = stereo_model(stereo_images[0], stereo_images[1])
                    gt_depth = stereo_output['depth']

                    # Get predicted depth at finest scale
                    pred_depth = depth_outputs[("depth", 0)]

                    # Resize ground truth to match prediction if needed
                    if gt_depth.shape != pred_depth.shape:
                        gt_depth = F.interpolate(
                            gt_depth,
                            size=pred_depth.shape[-2:],
                            mode='bilinear',
                            align_corners=True
                        )

                    # Visualize
                    # Visualize
                    visualize_results(
                        epoch,
                        batch_idx,
                        batch['curr_image'],  # Pass the full tensor
                        pred_depth,
                        gt_depth,
                        save_dir=f"training_visualization"
                    )

                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} complete, Average Loss: {avg_loss:.4f}')

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': mono_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'mono_depth_checkpoint_epoch_{epoch + 1}.pth')


def visualize_results(epoch, batch_idx, input_image, pred_depth, gt_depth, save_dir="visualization"):
    """
    Visualize and save comparison between predicted and ground truth depth
    Args:
        epoch: Current epoch number
        batch_idx: Current batch index
        input_image: Input image tensor
        pred_depth: Predicted depth tensor [B, 1, H, W]
        gt_depth: Ground truth depth tensor [B, 1, H, W]
        save_dir: Directory to save visualizations
    """
    import matplotlib.pyplot as plt
    import os

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Handle different input tensor shapes
    if input_image.dim() == 4:  # [B, C, H, W]
        img = input_image[0].permute(1, 2, 0).cpu().numpy()
    elif input_image.dim() == 3:  # [C, H, W]
        img = input_image.permute(1, 2, 0).cpu().numpy()
    elif input_image.dim() == 2:  # [H, W]
        img = input_image.cpu().numpy()
    else:
        raise ValueError(f"Unexpected input image dimensions: {input_image.shape}")

    # Convert depth maps to numpy
    pred = pred_depth[0, 0].cpu().numpy() if pred_depth.dim() == 4 else pred_depth[0].cpu().numpy()
    gt = gt_depth[0, 0].cpu().numpy() if gt_depth.dim() == 4 else gt_depth[0].cpu().numpy()

    # Create figure
    plt.figure(figsize=(15, 5))

    # Plot input image
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Input Image')
    plt.axis('off')

    # Plot predicted depth
    plt.subplot(132)
    plt.imshow(pred, cmap='magma')
    plt.colorbar(label='Depth')
    plt.title('Predicted Depth')
    plt.axis('off')

    # Plot ground truth depth
    plt.subplot(133)
    plt.imshow(gt, cmap='magma')
    plt.colorbar(label='Depth')
    plt.title('Ground Truth Depth')
    plt.axis('off')

    # Save figure
    save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_batch_{batch_idx:04d}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


    # Main execution
if __name__ == "__main__":
    # Sequences for training and validation
    train_sequences = ['2011_09_26']
    val_sequences = ['2011_09_30', '2011_10_03']

    root_dir = 'C:/Github/monodepth2/kitti_data'

    # Create datasets
    train_dataset = KITTIRawDataset(root_dir=root_dir, sequences=train_sequences)
    #val_dataset = KITTIRawDataset(root_dir=root_dir, sequences=val_sequences)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, drop_last=True)
    #val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=False)

    # Initialize models
    mono_model = MonocularDepthNet(pretrained=True)
    stereo_model = StereoModel()

    # Train
    train(mono_model, stereo_model, train_loader)
