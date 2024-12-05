import torch
import torch.nn as nn
from torchvision import models
from SupervisedApproach.Models.TemporalImages.encoder import SharedResNet18Encoder as Base_SharedRedNet18Encoder
import torch
import torchvision.transforms as transforms
from PIL import Image
from SupervisedApproach.KITTIDataset import KITTIRawDataset
from SupervisedApproach.AugmentedKITTIDataset import AugmentedKITTIDataset
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from PIL import Image
from SupervisedApproach.Models.TemporalImages.StereoDepthModel import StereoModel as BaseSteroModel
from SupervisedApproach.Models.TemporalImages.MonocularDepthNet import MonocularDepthNet as BaseMonocularDepthNet
from SupervisedApproach.Models.FullScaleImages.MonocularDepthNet import MonocularDepthNet as FullScaleMonocularDepthNet
from SupervisedApproach.Models.Resnet50.MonocularDepthNet import MonocularDepthNet as Resnet50DetphNet
from SupervisedApproach.Models.SingleImage.MonocularDepthNet import MonocularDepthNet as SingleFrameDepthNet
from SupervisedApproach.Models.DenseConnections.MonocularDepthNet import MonocularDepthNet as DenseMonocularDepthNet
import asyncio
import warnings
import matplotlib
import numpy as np
import random
from SupervisedApproach.main  import visualize_results

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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


mono_model = DenseMonocularDepthNet()
mono_model.load_state_dict(torch.load('C:\Github\monoHybridAproach\SupervisedApproach\TrainedModels\DenseConnections_mono_depth_checkpoint_epoch_20.pth')['model_state_dict'])

# Set the model to evaluation mode
mono_model.eval()

# Example input data (replace with your actual data)
# Ensure the data matches the input format of your model (e.g., shape, data type)
train_sequences = ['2011_09_26']
root_dir = 'C:/Github/monodepth2/kitti_data'

train_dataset = AugmentedKITTIDataset(root_dir=root_dir, sequences=train_sequences, start=0, end=5,
                                      camera_left="image_02", camera_right="image_03")

input_data = train_dataset.__getitem__(0)
src_images = [img.unsqueeze(0).to(device) for img in input_data['src_images']]
input_frames = torch.cat([src_images[0], src_images[1], src_images[2]], dim=1)


# Move to the same device as your model
mono_model = mono_model.to(device)
ground_truth = input_data['ground_truth'].unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():  # Disables gradient computation for faster inference
    output = mono_model(input_frames)
    center_frame_path = r'C:\Github\monodepth2\kitti_data\2011_09_26\2011_09_26_drive_0001_sync\image_02\data\0000000000.jpg'  # Replace with your absolute path
    center_frame_img = Image.open(center_frame_path).convert('RGB')  # Load image as RGB
    transform = transforms.ToTensor()
    center_frame_tensor = transform(center_frame_img).unsqueeze(0)  # Add batch dimension to match tensor format

    pred_depth = output[('depth', 0)]


    visualize_results(0, 0, center_frame_tensor, pred_depth, ground_truth, 'VisualizationResults','Result')




