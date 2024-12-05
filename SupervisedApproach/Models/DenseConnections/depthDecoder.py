import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc=[64, 64, 128, 256, 512], scales=range(4), min_depth=0.1, max_depth=100.0):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_output_channels = 1
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.convs = nn.ModuleDict()

        # Initialize lists to track feature dimensions
        self.dense_channels = []
        current_channels = self.num_ch_enc[4]  # Start with bottleneck features
        self.dense_channels.append(current_channels)

        # Create decoder layers with dense connections
        for i in range(4, -1, -1):
            # Calculate input channels for dense connections
            if i < 4:
                # Sum all previous dense features plus skip connection if available
                num_ch_in = sum(self.dense_channels)
                if i > 0:  # Add encoder skip connection channels
                    num_ch_in += self.num_ch_enc[i - 1]
            else:
                num_ch_in = self.num_ch_enc[4]  # Initial bottleneck features

            # Define fixed output channels for better memory management
            num_ch_out = 64  # Consistent channel size across layers

            # Create conv blocks with batch norm and optional skip connections
            self.convs[f"upconv_{i}_0"] = nn.Conv2d(num_ch_in, num_ch_out, 3, padding=1)
            self.convs[f"bn_{i}_0"] = nn.BatchNorm2d(num_ch_out)
            self.convs[f"upconv_{i}_1"] = nn.Conv2d(num_ch_out, num_ch_out, 3, padding=1)
            self.convs[f"bn_{i}_1"] = nn.BatchNorm2d(num_ch_out)

            # Track channels for dense connections
            self.dense_channels.append(num_ch_out)

        # Create depth prediction layers for each scale
        for i in self.scales:
            # Calculate total input channels for depth prediction
            total_channels = sum(self.dense_channels[:(5 - i)])
            self.convs[f"dispconv_{i}"] = nn.Sequential(
                nn.Conv2d(total_channels, 128, 3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(128, self.num_output_channels, 3, padding=1)
            )

    def forward(self, input_features, debug=False):
        outputs = {}
        dense_features = []
        x = input_features[-1]
        dense_features.append(x)

        for i in range(4, -1, -1):
            if debug:
                print(f"\nProcessing scale {i}")
                print(f"Current feature shape: {x.shape}")

            # Handle dense connections
            if i < 4:
                # Get target size for this level
                target_size = input_features[i].shape[-2:] if i > 0 else x.shape[-2:]

                # Scale all previous features to current resolution
                scaled_features = []
                for feat in dense_features:
                    scaled_feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
                    scaled_features.append(scaled_feat)
                    if debug:
                        print(f"Scaled feature shape: {scaled_feat.shape}")

                # Concatenate all scaled features
                x = torch.cat(scaled_features, dim=1)

                # Add skip connection if available
                if i > 0:
                    skip_x = input_features[i - 1]
                    # Ensure skip connection has the same spatial dimensions
                    skip_x = F.interpolate(skip_x, size=target_size, mode='bilinear', align_corners=True)
                    x = torch.cat([x, skip_x], dim=1)

                if debug:
                    print(f"After all concatenations: {x.shape}")

            # Process through double convolution blocks
            x = self.convs[f"upconv_{i}_0"](x)
            x = F.relu(self.convs[f"bn_{i}_0"](x))
            x = self.convs[f"upconv_{i}_1"](x)
            x = F.relu(self.convs[f"bn_{i}_1"](x))

            if debug:
                print(f"After convolutions: {x.shape}")

            # Store features for dense connections
            dense_features.append(x)

            # For next iteration, upsample if not last layer
            if i > 0:
                next_size = input_features[i - 1].shape[-2:]
                x = F.interpolate(x, size=next_size, mode='bilinear', align_corners=True)
                if debug:
                    print(f"After upsampling for next iteration: {x.shape}")

            # Generate depth predictions for requested scales
            if i in self.scales:
                # Get target size for depth prediction
                output_size = input_features[i].shape[-2:] if i > 0 else x.shape[-2:]

                # Scale and combine all dense features for depth prediction
                scaled_dense_features = []
                for feat in dense_features[:(5 - i)]:
                    scaled_feat = F.interpolate(feat, size=output_size, mode='bilinear', align_corners=True)
                    scaled_dense_features.append(scaled_feat)

                combined_features = torch.cat(scaled_dense_features, dim=1)
                depth = self.convs[f"dispconv_{i}"](combined_features)
                depth = F.sigmoid(depth)

                # Convert to metric depth
                depth = self.min_depth + (self.max_depth - self.min_depth) * depth
                outputs[("depth", i)] = depth

        return outputs
def verify_depth_decoder(encoder_features, decoder):
    """
    Verify depth decoder outputs
    Args:
        encoder_features: List of feature maps from encoder
        decoder: DepthDecoder instance
    """
    outputs = decoder(encoder_features)

    print("\nDepth Decoder Verification:")
    for scale, depth in outputs.items():
        #print(f"Scale {scale[1]}: Shape={depth.shape}, Range=[{depth.min():.3f}, {depth.max():.3f}]")

        # Visualize depth map
        plt.figure(figsize=(5, 5))
        plt.imshow(depth[0, 0].cpu().detach().numpy(), cmap='magma')
        plt.colorbar()
        plt.title(f'Depth Map - Scale {scale[1]}')
        plt.show()