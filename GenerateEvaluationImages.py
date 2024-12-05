import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


class KITTIDepthProcessor:
    def __init__(self, data_dir, baseline=0.54, focal_length=721.5377):
        """
        Initialize the KITTI depth processor
        Args:
            data_dir: Path to KITTI dataset directory containing training/testing folders
            baseline: Distance between stereo cameras (in meters)
            focal_length: Focal length of the camera
        """
        self.data_dir = data_dir
        self.baseline = baseline
        self.focal_length = focal_length

        # Paths for different data types
        self.disp_dir = os.path.join(data_dir, 'training', 'disp_occ_0')  # Left image disparities
        self.image_dir = os.path.join(data_dir, 'training', 'image_2')  # Left images

    def load_disparity(self, index):
        """
        Load a disparity map from KITTI dataset
        Args:
            index: Image index
        Returns:
            disparity: Disparity map
            valid_mask: Boolean mask of valid pixels
        """
        disp_path = os.path.join(self.disp_dir, f'{index:06d}_10.png')

        # Load 16-bit PNG and convert to float32
        disp = np.array(Image.open(disp_path), dtype=np.float32)

        # Convert from stored integer disparity to float disparity
        disp = disp / 256.0

        # Create validity mask (0 values are invalid)
        valid_mask = disp > 0

        return disp, valid_mask

    def disparity_to_depth(self, disparity):
        """
        Convert disparity map to depth map using KITTI stereo parameters
        Args:
            disparity: Disparity map
        Returns:
            depth: Depth map
        """
        depth = np.zeros_like(disparity)
        valid_mask = disparity > 0

        # Apply depth = baseline * focal_length / disparity
        depth[valid_mask] = (self.baseline * self.focal_length) / disparity[valid_mask]

        return depth

    def process_all_images(self, output_dir, visualize=True):
        """
        Process all images in the dataset and save depth maps
        Args:
            output_dir: Directory to save processed depth maps
            visualize: Whether to create visualization plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        if visualize:
            os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

        # Get list of all disparity files
        disp_files = sorted(os.listdir(self.disp_dir))

        for filename in tqdm(disp_files, desc="Processing depth maps"):
            # Get index from filename
            index = int(filename.split('_')[0])

            # Load disparity
            disparity, valid_mask = self.load_disparity(index)

            # Convert to depth
            depth = self.disparity_to_depth(disparity)

            # Save depth map as numpy array
            depth_path = os.path.join(output_dir, f'{index:06d}_depth.npy')
            np.save(depth_path, depth)

            # Save validity mask
            mask_path = os.path.join(output_dir, f'{index:06d}_mask.npy')
            np.save(mask_path, valid_mask)

            if visualize:
                self.visualize_depth(depth, valid_mask, index, output_dir)

    def visualize_depth(self, depth, valid_mask, index, output_dir):
        """
        Create visualization of depth map
        Args:
            depth: Depth map
            valid_mask: Validity mask
            index: Image index
            output_dir: Output directory
        """
        # Create figure
        plt.figure(figsize=(12, 5))

        # Plot depth map
        plt.subplot(121)
        valid_depth = depth[valid_mask]
        vmin, vmax = np.percentile(valid_depth, [5, 95])
        plt.imshow(depth, cmap='magma', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Depth (m)')
        plt.title('Depth Map')

        # Plot validity mask
        plt.subplot(122)
        plt.imshow(valid_mask, cmap='gray')
        plt.title('Validity Mask')

        # Save figure
        vis_path = os.path.join(output_dir, 'visualizations', f'{index:06d}_depth_vis.png')
        plt.savefig(vis_path)
        plt.close()


def main():
    # Example usage
    data_dir = r"C:\Users\OtteC\Downloads\data_scene_flow"  # Replace with your KITTI dataset path
    output_dir = r"C:\Github\monoHybridAproach\EvaluationImagesDepth"  # Replace with desired output path

    # Initialize processor
    processor = KITTIDepthProcessor(data_dir)

    # Process all images
    processor.process_all_images(output_dir)

    print("Processing complete!")
    print(f"Depth maps and visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()