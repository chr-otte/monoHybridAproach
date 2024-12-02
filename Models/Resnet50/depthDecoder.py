import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc=[64, 256, 512, 1024, 2048], scales=range(4), min_depth=0.1, max_depth=100.0):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_output_channels = 1
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.convs = nn.ModuleDict()

        num_ch_out_prev = self.num_ch_enc[4]

        # Define upsampling and decoding layers
        for i in range(4, -1, -1):
            if i == 4:
                num_ch_in = self.num_ch_enc[4]
            else:
                num_ch_in = num_ch_out_prev

            num_ch_out = self.num_ch_enc[i]
            num_ch_out_prev = num_ch_out

            # Upsampling layer
            self.convs[f"upconv_{i}"] = nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, stride=1, padding=1)
            self.convs[f"bn_{i}"] = nn.BatchNorm2d(num_ch_out)

            if i > 0:
                # Combine upsampled and skip connection features
                combined_ch = num_ch_out + self.num_ch_enc[i - 1]
                self.convs[f"conv_{i}_1"] = nn.Conv2d(combined_ch, num_ch_out, kernel_size=3, stride=1, padding=1)

        # Final depth prediction layers
        for i in self.scales:
            self.convs[f"dispconv_{i}"] = nn.Conv2d(self.num_ch_enc[i], self.num_output_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, input_features, debug=False):
        for i, f in enumerate(input_features):
            if debug:
                print(f"Scale {i}: {f.shape}")

        outputs = {}
        x = input_features[-1]  # Start with the deepest feature

        for i in range(4, -1, -1):
            # Upsample the current feature map
            x = F.relu(self.convs[f"bn_{i}"](self.convs[f"upconv_{i}"](x)))
            if debug:
                print(f"After upconv_{i}: {x.shape}")

            if i > 0:
                # Upsample to match the spatial resolution of the skip connection
                x = F.interpolate(x, scale_factor=2, mode='nearest')
                if debug:
                    print(f"Upsampled feature at scale {i}: {x.shape}")

                # Concatenate with the corresponding encoder feature
                x = torch.cat([x, input_features[i - 1]], dim=1)
                if debug:
                    print(f"Concatenated feature at scale {i}: {x.shape}")

                # Process concatenated features to reduce channels
                x = F.relu(self.convs[f"conv_{i}_1"](x))
                if debug:
                    print(f"Processed feature at scale {i}: {x.shape}")

            # Generate depth output for the requested scales
            if i in self.scales:
                disp = torch.sigmoid(self.convs[f"dispconv_{i}"](x))
                depth = self.min_depth + (self.max_depth - self.min_depth) * disp
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