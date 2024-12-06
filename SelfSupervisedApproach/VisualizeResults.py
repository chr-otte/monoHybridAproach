import os
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import random
from SelfSupervisedApproach.KittiDataset import KITTIRawDataset
from SelfSupervisedApproach.main_final import get_experiment_config, save_visualization_improved
from DepthNet import DepthNetbase, DepthNet18, DepthNet50
from SelfSupervisedApproach.PoseNet import PoseNet
# Set manual seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using multi-GPU setups
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)
generator = torch.Generator()
generator.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_vis_path(i : int) -> str:
    if i == 0:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\VisualizedResults\depth_raw_epoch_0_batch_0_exp_0.npy"
    if i == 1:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\VisualizedResults\depth_raw_epoch_0_batch_0_exp_1.npy"
    if i == 2:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\VisualizedResults\depth_raw_epoch_0_batch_0_exp_2.npy"
    if i == 3:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\VisualizedResults\depth_raw_epoch_0_batch_0_exp_3.npy"
    if i == 4:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\VisualizedResults\depth_raw_epoch_0_batch_0_exp_4.npy"
    if i == 5:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\VisualizedResults\depth_raw_epoch_0_batch_0_exp_5.npy"
    if i == 6:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\Models\ResNet50 Architecture_checkpoint_epoch_10.pth"

def load_model_path(i : int) -> str:
    if i == 0:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\Models\Baseline Model_checkpoint_epoch_10.pth"
    if i == 1:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\Models\Clamp Depth Calculation_checkpoint_epoch_10.pth"
    if i == 2:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\Models\Reduced Target Mean Loss_checkpoint_epoch_10.pth"
    if i == 3:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\Models\Data Augmentation_checkpoint_epoch_10.pth"
    if i == 4:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\Models\Black & White Images_checkpoint_epoch_10.pth"
    if i == 5:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\Models\Increased Weight Decay_checkpoint_epoch_10.pth"
    if i == 6:
        return rf"C:\Github\monoHybridAproach\SelfSupervisedApproach\Models\ResNet50 Architecture_checkpoint_epoch_10.pth"

root_dir =  rf'C:\Github\monodepth2\kitti_data\2011_09_26'
train_sequences = [rf'2011_09_26']

def generate_input_images():
    for i in range(7):
        depth_net, target_mean_loss_weight, KITTI_Raw_Data_set, pose_net, weight_decay, name = get_experiment_config(i)
        depth_net.to(device)

        # Create datasets
        train_dataset = KITTI_Raw_Data_set(root_dir=root_dir, sequences=train_sequences, start=0, end=5)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=8, drop_last=True)

       # Set the model to evaluation mode
        depth_net.eval()

        for batch_idx, batch in enumerate(train_loader):
            curr_image = batch['curr_image'].to(device)
            src_images = batch['src_images'].to(device)
            disp, depth = depth_net(curr_image)


            save_visualization_improved(
                curr_image[0],
                depth[0, 0],
                0,
                batch_idx,
                i
            )
            break



import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for i in range(6):
        path = get_vis_path(i)
        depth_array = np.load(path)
        _,_,_,_,_, name = get_experiment_config(i)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the depth array
        img = ax.imshow(depth_array, cmap='magma')

        # Create a divider to match the colorbar height to the image height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and padding as needed

        # Add the colorbar to the custom axis
        cbar = plt.colorbar(img, cax=cax)

        # Set title for the subplot
        ax.set_title(name)

        # Adjust layout for proper fitting
        plt.tight_layout()

        # Save the plot to a file
        output_path = f"VisualizedResults/FinalImages/depth_map_{name}.png"  # Set your desired output path and filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        # Close the figure to free up memory
        plt.close(fig)



