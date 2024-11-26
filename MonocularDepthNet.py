import torch
import torch.nn as nn
from torchvision import models
from encoder import SharedResNet18Encoder
from depthDecoder import DepthDecoder

class MonocularDepthNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = SharedResNet18Encoder(pretrained)
        self.decoder = DepthDecoder(num_ch_enc=[64, 64, 128, 256, 512])

    def forward(self, x):
        features, skip_connections = self.encoder(x)
        encoder_features = [
            skip_connections['conv1'],
            skip_connections['layer1'],
            skip_connections['layer2'],
            skip_connections['layer3'],
            skip_connections['layer4']
        ]
        return self.decoder(encoder_features)
