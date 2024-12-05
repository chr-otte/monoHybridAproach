import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_depth_visualization(img_path, depth_map, validity_mask, save_path=None):
    """
    Create a visualization with original image, depth map, and validity mask
    Args:
        img_path: Path to original RGB image
        depth_map: Numpy array of depth values
        validity_mask: Boolean mask of valid depth pixels
        save_path: Optional path to save the visualization
    """
    # Load original image
    rgb_img = np.array(Image.open(img_path))

    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Plot original image
    ax1.imshow(rgb_img)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Plot depth map with colorbar
    im2 = ax2.imshow(depth_map, cmap='magma')
    ax2.set_title('Depth Map')
    ax2.axis('off')

    # Create colorbar with same height as image
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax, label='Depth (m)')

    # Plot validity mask
    ax3.imshow(validity_mask, cmap='gray')
    ax3.set_title('Validity Mask')
    ax3.axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    img_path = r"C:\Users\OtteC\Downloads\data_scene_flow\training\image_2\000000_10.png"
    depth_path = r"C:\Github\monoHybridAproach\EvaluationImagesDepth\000000_depth.npy"
    mask_path = r"C:\Github\monoHybridAproach\EvaluationImagesDepth\000000_mask.npy"
    save_path = "depth_visualization.png"

    # Load data
    depth_map = np.load(depth_path)
    validity_mask = np.load(mask_path)

    # Create visualization
    create_depth_visualization(
        img_path,
        depth_map,
        validity_mask,
        save_path
    )