import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class KITTIRawDataset(Dataset):
    def __init__(self, root_dir, sequences, camera_left='image_02', camera_right ='image_03'):

        self.root_dir = root_dir
        self.sequences = sequences  # sequences of directories for data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3706, 0.3861, 0.3680],  # KITTI calculated mean
            std=[0.3059, 0.3125, 0.3136])                        # KITTI calculated std
        ])
        self.samples = []

        # Loop over sequences to collect file paths
        for seq in sequences:
            seq_dir = os.path.join(root_dir, seq)
            # List all drives within the date
            drives = [d for d in os.listdir(seq_dir) if 'sync' in d]
            for drive in drives:
                drive_dir = os.path.join(seq_dir, drive)
                image_dir_left = os.path.join(drive_dir, camera_left, 'data')
                image_dir_right = os.path.join(drive_dir, camera_right, 'data')


                if not os.path.exists(image_dir_left) or  not os.path.exists(image_dir_right):
                    continue  # Skip if images are not available

                # Get sorted list of image files
                image_files_left = sorted([
                    os.path.join(image_dir_left, f) for f in os.listdir(image_dir_left)
                    if f.endswith('.JPG') or f.endswith('.jpg')
                ])

                image_files_right = sorted([
                    os.path.join(image_dir_right, f) for f in os.listdir(image_dir_right)
                    if f.endswith('.JPG') or f.endswith('.jpg')
                ])

                # Collect samples as dictionaries containing current and source images

                # Two images to simulate multiple cameras, needs to be consecutive images.
                # avoiding the use of first and last image, as no consecutive images exists.
                for i in range(1, len(image_files_left) - 1):
                    frame = {}
                    frame['curr_img'] = image_files_left[i]
                    frame['src_imgs'] = [
                        image_files_left[i - 1],
                        image_files_left[i],
                        image_files_left[i + 1]
                    ]
                    frame['stero_imgs'] = [image_files_left[i],image_files_right[i]]

                    self.samples.append(frame)

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load stereo images
        stereo = [Image.open(img_path).convert('RGB') for img_path in sample['stero_imgs']]
        frames = [Image.open(img_path).convert('RGB') for img_path in sample['src_imgs']]

        def to_tensor(img):
            img_np = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            return tensor  # Shape: [3, H, W]

        # Convert stereo images to tensors
        stereo = [to_tensor(f) for f in stereo]

        # Apply transforms to temporal frames if needed
        if self.transform:
            frames = [self.transform(f) for f in frames]

        return {
            'curr_image': frames[1],
            'src_images': [frames[0], frames[1], frames[2]],
            'stereo_images': stereo  # List of two tensors, each with shape [3, H, W]
        }