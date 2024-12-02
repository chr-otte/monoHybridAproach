import os
from PIL import Image
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import models
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from Models.TemporalImages.StereoDepthModel  import StereoModel
import random
import hashlib
def get_hashed_filename(left_img_path, right_img_path):
    unique_string = left_img_path + right_img_path
    hashed = hashlib.md5(unique_string.encode()).hexdigest()
    return f"disparity_{hashed}.pth"

class AugmentedKITTIDataset(Dataset):
    def __init__(self, root_dir, sequences, camera_left='image_00', camera_right='image_01', brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, seed=42, start=0, end = 100, full_scale = False):
        self.root_dir = root_dir
        self.sequences = sequences

        # Store augmentation parameters
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        # Store seed for reproducibility
        self.seed = seed

        # TemporalImages transform

        if full_scale:
            self.base_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3706, 0.3861, 0.3680],
                    std=[0.3059, 0.3125, 0.3136]
                )
            ])
        else:
            self.base_transform = transforms.Compose([
                transforms.Resize((192, 640)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3706, 0.3861, 0.3680],
                    std=[0.3059, 0.3125, 0.3136]
                )
            ])

        self.samples = []

        # Collect samples with deterministic ordering
        for seq in sorted(sequences):  # Use sorted to ensure consistent order
            seq_dir = os.path.join(root_dir, seq)
            drives = sorted([d for d in os.listdir(seq_dir) if 'sync' in d])

            for drive in drives:
                drive_dir = os.path.join(seq_dir, drive)
                image_dir_left = os.path.join(drive_dir, camera_left, 'data')
                image_dir_right = os.path.join(drive_dir, camera_right, 'data')

                if not os.path.exists(image_dir_left) or not os.path.exists(image_dir_right):
                    continue

                # Get sorted list of image files
                image_files_left = sorted([
                    os.path.join(image_dir_left, f) for f in os.listdir(image_dir_left)
                    if f.endswith('.JPG') or f.endswith('.jpg')
                ])

                image_files_right = sorted([
                    os.path.join(image_dir_right, f) for f in os.listdir(image_dir_right)
                    if f.endswith('.JPG') or f.endswith('.jpg')
                ])

                # Collect samples
                for i in range(1, len(image_files_left) - 1):
                    frame = {}
                    frame['curr_img'] = image_files_left[i]
                    frame['src_imgs'] = [
                        image_files_left[i - 1],
                        image_files_left[i],
                        image_files_left[i + 1]
                    ]
                    frame['stero_imgs'] = [image_files_left[i], image_files_right[i]]
                    self.samples.append(frame)

        self.samples = self.samples[start:end]

        precomputed_dir = "PrecomputedGroundtruths"
        # Precompute or load depth maps
        model = StereoModel()
        os.makedirs(precomputed_dir, exist_ok=True)
        for i, sample in enumerate(self.samples):
            leftpath = sample['stero_imgs'][0]
            rightpath = sample['stero_imgs'][1]

            # Generate a unique file name for the stereo pair
            name = get_hashed_filename(leftpath, rightpath)
            precomputed_path = os.path.join(precomputed_dir, f"{name}.pth")

            if os.path.exists(precomputed_path):
                # Load precomputed depth map
                sample['depth_map'] = torch.load(precomputed_path)
            else:
                # Compute depth map using the stereo model
                if model is not None:
                    left_img = Image.open(leftpath).convert('RGB')
                    right_img = Image.open(rightpath).convert('RGB')

                    left_tensor = self.base_transform(left_img).unsqueeze(0).cuda()
                    right_tensor = self.base_transform(right_img).unsqueeze(0).cuda()

                    with torch.no_grad():

                        result = model(left_tensor, right_tensor)
                        depth_map = result['depth'].squeeze(0).cpu()

                    # Save depth map
                    torch.save(depth_map, precomputed_path)
                    sample['depth_map'] = depth_map
                else:
                    raise ValueError("Stereo model is not provided, and precomputed depth map is missing.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        stereo = [Image.open(img_path).convert('RGB') for img_path in sample['stero_imgs']]
        frames = [Image.open(img_path).convert('RGB') for img_path in sample['src_imgs']]

        def augment_image(img):
            # Apply specified color adjustments
            if self.brightness != 0:
                img = transforms.functional.adjust_brightness(img, 1.0 + self.brightness)
            if self.contrast != 0:
                img = transforms.functional.adjust_contrast(img, 1.0 + self.contrast)
            if self.saturation != 0:
                img = transforms.functional.adjust_saturation(img, 1.0 + self.saturation)
            if self.hue != 0:
                img = transforms.functional.adjust_hue(img, self.hue)

            # TemporalImages transforms
            img = transforms.functional.resize(img, (192, 640))
            img_np = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

            return transforms.functional.normalize(
                img_tensor,
                mean=[0.3706, 0.3861, 0.3680],
                std=[0.3059, 0.3125, 0.3136]
            )

        # Apply augmentations to all images
        stereo = [augment_image(img) for img in stereo]
        frames = [augment_image(img) for img in frames]

        return {
            'curr_image': frames[1],
            'src_images': [frames[0], frames[1], frames[2]],
            'stereo_images': stereo,  # List of two tensors, each with shape [3, H, W]
            'ground_truth' : sample['depth_map']
        }


def verify_specific_augmentation(dataset, num_samples=1):
    """
    Verify augmentations by showing original and augmented images side by side
    Args:
        dataset: The augmented dataset
        num_samples: Number of random samples to show
    """
    plt.figure(figsize=(15, 5 * num_samples))

    for sample_idx in range(num_samples):
        # Get random sample index
        idx = 100

        # Load original image directly
        sample = dataset.samples[idx]
        original_img = Image.open(sample['curr_img']).convert('RGB')

        # Resize original to match dataset size for fair comparison
        original_img = transforms.Resize((192, 640))(original_img)
        original_tensor = transforms.ToTensor()(original_img)

        # Get augmented version
        augmented_sample = dataset[idx]
        augmented_img = augmented_sample['curr_image']

        # Plot original
        plt.subplot(num_samples, 2, 2 * sample_idx + 1)
        plt.imshow(original_tensor.permute(1, 2, 0))
        plt.title('Original Image')
        plt.axis('off')

        # Plot augmented
        plt.subplot(num_samples, 2, 2 * sample_idx + 2)
        plt.imshow(augmented_img.permute(1, 2, 0) * 0.5 + 0.5)  # Denormalize
        plt.title(f'Augmented Image\nBrightness: {dataset.brightness:.2f}, '
                  f'Contrast: {dataset.contrast:.2f}\n'
                  f'Saturation: {dataset.saturation:.2f}, '
                  f'Hue: {dataset.hue:.2f}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Usage example:
if __name__ == "__main__":
    # Create dataset with color augmentation
    root_dir = 'C:/Github/monodepth2/kitti_data'
    sequences = ['2011_09_26']
    brightness_test = AugmentedKITTIDataset(root_dir, sequences, brightness=0.3)
    contrast_test = AugmentedKITTIDataset(root_dir, sequences, contrast=0.3)
    saturation = AugmentedKITTIDataset(root_dir, sequences, saturation=0.3)
    combined_test = AugmentedKITTIDataset(root_dir, sequences,brightness=0.3, contrast=0.3, saturation=0.3)

    verify_specific_augmentation(brightness_test)
    verify_specific_augmentation(contrast_test)
    verify_specific_augmentation(saturation)
    verify_specific_augmentation(combined_test)
