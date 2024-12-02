import torch
import torch.nn as nn
from torchvision import models
from Models.TemporalImages.encoder import SharedResNet18Encoder as Base_SharedRedNet18Encoder
import torch
import torchvision.transforms as transforms
from PIL import Image
from KITTIDataset import KITTIRawDataset
from AugmentedKITTIDataset import AugmentedKITTIDataset
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from PIL import Image
from Models.TemporalImages.StereoDepthModel import StereoModel as BaseSteroModel
from Models.TemporalImages.MonocularDepthNet import MonocularDepthNet as BaseMonocularDepthNet
from Models.FullScaleImages.MonocularDepthNet import MonocularDepthNet as FullScaleMonocularDepthNet
from Models.Resnet50.MonocularDepthNet import MonocularDepthNet as Resnet50DetphNet
from Models.SingleImage.MonocularDepthNet import MonocularDepthNet as SingleFrameDepthNet
from Models.DenseConnections.MonocularDepthNet import MonocularDepthNet as DenseMonocularDepthNet


import numpy as np
import cv2
import kornia
import random
import pandas as pd

def train_hybrid_depth(mono_model, optimizer,
                       frame_t_minus, frame_t, frame_t_plus,
                       stereo_left, stereo_right, ground_truth, debug = False):
    # Stack adjacent frames
    input_frames = torch.cat([frame_t_minus, frame_t, frame_t_plus], dim=1)

    # Get depth prediction from monocular model
    depth_outputs = mono_model(input_frames)

    # Calculate stereo depth at full resolution
    with torch.no_grad():
        full_res_depth = ground_truth.to('cuda')

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


def train(mono_model, train_loader, validation_loader, num_epochs=10, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mono_model = mono_model.to(device)
    total_loss = 0
    optimizer = torch.optim.Adam(mono_model.parameters(), lr=learning_rate)
    visualization_freq = 10  # Visualize every 50 batches

    batch_loss = []
    validation_loss = []

    for epoch in range(num_epochs):
        mono_model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Move left to device
            src_images = [img.to(device) for img in batch['src_images']]
            stereo_images = [img.to(device) for img in batch['stereo_images']]

            optimizer.zero_grad()

            loss, depth_outputs = train_hybrid_depth(
                mono_model,  optimizer,
                src_images[0],  # t-1 frame
                src_images[1],  # t frame
                src_images[2],  # t+1 frame
                stereo_images[0],  # left image
                stereo_images[1],   # right image
                batch["ground_truth"]
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_loss += loss.item()

            if batch_idx % visualization_freq == 0:
                with torch.no_grad():
                    # Get ground truth depth from stereo model
                    gt_depth = batch["ground_truth"]

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
                    batch_loss.append(loss.item())
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
        # Validation phase
        mono_model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(validation_loader):
                src_images = [img.to(device) for img in batch['src_images']]
                stereo_images = [img.to(device) for img in batch['stereo_images']]
                ground_truth = batch['ground_truth'].to(device)

                # Compute validation loss
                loss, _ = train_hybrid_depth(
                    mono_model, optimizer,
                    src_images[0],  # t-1 frame
                    src_images[1],  # t frame
                    src_images[2],  # t+1 frame
                    stereo_images[0],  # left image
                    stereo_images[1],  # right image
                    ground_truth
                )

                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(validation_loader)
        validation_loss.append(avg_val_loss)
        print(f'Epoch {epoch} Validation Complete, Average Validation Loss: {avg_val_loss:.4f}')
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
    return batch_loss, validation_loss



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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data(i):
    start_train = 0
    end_train = 100
    start_val = 100
    end_val = 102
    if(i==0):
        train_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_train, end=end_train, camera_left="image_02", camera_right="image_03")
        val_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_val, end=end_val, camera_left="image_02", camera_right="image_03")
        name = "Basemodel"
        mono_model = BaseMonocularDepthNet(pretrained=True)

    elif(i==1):
        train_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_train, end=end_train, contrast=0.3, camera_left="image_02", camera_right="image_03")
        val_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_val, end=end_val, contrast=0.3, camera_left="image_02", camera_right="image_03")
        name = "Basemodel - Contrast 0.3"
        mono_model = BaseMonocularDepthNet(pretrained=True)

    elif(i==2):
        train_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_train, end=end_train, contrast=0.3, camera_left="image_00", camera_right="image_01")
        val_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_val, end=end_val, contrast=0.3, camera_left="image_00", camera_right="image_01")
        name = "Basemodel - Contrast 0.3 GreayScale"
        mono_model = BaseMonocularDepthNet(pretrained=True)

    elif(i==3):
        train_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_train, end=end_train, camera_left="image_00", camera_right="image_01")
        val_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_val, end=end_val, camera_left="image_00", camera_right="image_01")
        name = "Basemodel - GreayScale"
        mono_model = BaseMonocularDepthNet(pretrained=True)

    elif(i==4):
        train_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_train, end=end_train, camera_left="image_02", camera_right="image_03", full_scale=True)
        val_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences , start=start_val, end=end_val, camera_left="image_02", camera_right="image_03", full_scale=True)
        name = "Fullscale"
        mono_model = FullScaleMonocularDepthNet(pretrained=True)

    elif(i==5):
        train_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, camera_left="image_02", camera_right="image_03")
        val_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences , start=start_val, end=end_val, camera_left="image_02", camera_right="image_03")
        name = "Resnet50"
        mono_model = Resnet50DetphNet(pretrained=True)

    elif(i==6):
        train_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_train, end=end_train, camera_left="image_02", camera_right="image_03")
        val_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_val, end=end_val, camera_left="image_02", camera_right="image_03")
        name = "SingleFrame"
        mono_model = SingleFrameDepthNet(pretrained=True)

    elif(i==7):
        train_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_train, end=end_train, camera_left="image_02", camera_right="image_03")
        val_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=start_val, end=end_val, camera_left="image_02", camera_right="image_03")
        name = "DenseConnections"
        mono_model = DenseMonocularDepthNet(pretrained=True)

    return mono_model, train_dataset, val_dataset, name



if __name__ == "__main__":
    # Sequences for training and validation
    train_sequences = ['2011_09_26']
    val_sequences = ['2011_09_30', '2011_10_03']

    # Set manual seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # If using multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)

    # Create a PyTorch generator with a fixed seed
    generator = torch.Generator()
    generator.manual_seed(42)

    root_dir = 'C:/Github/monodepth2/kitti_data'

    for i in range(8):
        mono_model, train_dataset, val_dataset, name = get_data(i)

        # Create DataLoader with consistent shuffling
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,  # Shuffling is deterministic with the fixed seed
            num_workers=4,
            worker_init_fn=seed_worker,
            pin_memory=True,
            generator=generator
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=True,  # Shuffling is deterministic with the fixed seed
            num_workers=4,
            worker_init_fn=seed_worker,
            pin_memory=True,
            generator=generator
        )


        # Train
        batch_loss, validation_loss = train(mono_model, train_loader, val_loader, num_epochs=2)
        batch_loss_df = pd.DataFrame({
            "index": range(len(batch_loss)),
            "Batch_Loss": batch_loss,
        })
        batch_loss_df.to_csv(f"batch_loss_{name}.csv", index=False)

        val_loss_df = pd.DataFrame({
            "index": range(len(validation_loss)),
            "Validation_Loss": validation_loss
        })
        val_loss_df.to_csv(f"validation_loss_{name}.csv", index=False)

