import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from SupervisedApproach.Models.DenseConnections.MonocularDepthNet import MonocularDepthNet
class KITTIEvalDataset(Dataset):
    def __init__(self, image_dir, gt_depth_dir, gt_mask_dir):
        """
        Dataset for KITTI evaluation that loads consecutive frames
        Args:
            image_dir: Directory containing test images
            gt_depth_dir: Directory containing ground truth depth maps
            gt_mask_dir: Directory containing validity masks
        """
        self.image_dir = image_dir
        self.gt_depth_dir = gt_depth_dir
        self.gt_mask_dir = gt_mask_dir

        # Get sorted list of all image files
        self.filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

        # Image preprocessing - matching your training setup
        self.transform = transforms.Compose([
            transforms.Resize((192, 640)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3706, 0.3861, 0.3680],
                std=[0.3059, 0.3125, 0.3136]
            )
        ])
    def __len__(self):
        return len(self.filenames) - 2

    def __getitem__(self, idx):
        frames = []
        for i in range(3):
            img_path = os.path.join(self.image_dir, self.filenames[idx + i])
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            frames.append(img_tensor)

        basename = os.path.splitext(self.filenames[idx + 1])[0]  # middle frame
        depth_path = os.path.join(self.gt_depth_dir, f"{basename}_depth.npy")
        mask_path = os.path.join(self.gt_mask_dir, f"{basename}_mask.npy")

        gt_depth = np.load(depth_path)
        gt_mask = np.load(mask_path)

        return {
            'curr_image': frames[1],
            'src_images': frames,
            'gt_depth': torch.from_numpy(gt_depth).float(),
            'gt_mask': torch.from_numpy(gt_mask).bool(),
            'filename': self.filenames[idx + 1]
        }


class ModelEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def evaluate(self, dataloader):
        metrics_sum = None
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                src_images = torch.cat(batch['src_images'], dim=1).to(self.device)
                prediction = self.model(src_images)
                pred_depth = prediction[("depth", 0)].cpu().numpy()

                gt_depth = batch['gt_depth'].cpu().numpy()
                gt_mask = batch['gt_mask'].cpu().numpy()

                batch_metrics = self.compute_metrics(pred_depth, gt_depth, gt_mask)
                if metrics_sum is None:
                    metrics_sum = {k: v for k, v in batch_metrics.items()}
                else:
                    for k, v in batch_metrics.items():
                        metrics_sum[k] += v
                total_samples += 1

        metrics_avg = {k: v / total_samples for k, v in metrics_sum.items()}
        return metrics_avg

    def compute_metrics(self, pred, gt, mask):

        if len(pred.shape) == 4:
            pred = pred.squeeze()
        elif len(pred.shape) == 3:
            pred = pred.squeeze()

        if len(gt.shape) > 2:
            gt = gt.squeeze()

        if len(mask.shape) > 2:
            mask = mask.squeeze()

        print(f"Before resize - Prediction shape: {pred.shape}")
        print(f"Ground truth shape: {gt.shape}")
        print(f"Mask shape: {mask.shape}")

        pred_resized = torch.nn.functional.interpolate(
            torch.from_numpy(pred).unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
            size=(gt.shape[0], gt.shape[1]),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()

        print(f"After resize - Prediction shape: {pred_resized.shape}")

        depth_mask = (gt > 1e-3) & (gt < 80)

        combined_mask = mask & depth_mask
        pred_valid = pred_resized[combined_mask]
        gt_valid = gt[combined_mask]
        thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))

        metrics = {
            'abs_rel': np.mean(np.abs(pred_valid - gt_valid) / gt_valid),
            'sq_rel': np.mean(((pred_valid - gt_valid) ** 2) / gt_valid),
            'rmse': np.sqrt(np.mean((pred_valid - gt_valid) ** 2)),
            'rmse_log': np.sqrt(np.mean((np.log(pred_valid) - np.log(gt_valid)) ** 2)),
            'delta1': (thresh < 1.25).mean(),
            'delta2': (thresh < 1.25 ** 2).mean(),
            'delta3': (thresh < 1.25 ** 3).mean(),
            'd1_all': (np.abs(pred_valid - gt_valid) > 3).mean()
        }

        return metrics
def main():
    # Set paths
    image_dir   = r"C:\Github\monoHybridAproach\EvaluationImagesDepth\image_2"
    gt_depth_dir= r"C:\Github\monoHybridAproach\EvaluationImagesDepth\ground_truth"
    gt_mask_dir = r"C:\Github\monoHybridAproach\EvaluationImagesDepth\masks"
    model_path  = r"C:\Github\monoHybridAproach\TrainedModels\DenseConnections_mono_depth_checkpoint_epoch_20.pth"

    model = MonocularDepthNet()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = KITTIEvalDataset(image_dir, gt_depth_dir, gt_mask_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(dataloader)

    print("\nEvaluation Results:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric:>10s}: {value:.3f}")


if __name__ == "__main__":
    main()