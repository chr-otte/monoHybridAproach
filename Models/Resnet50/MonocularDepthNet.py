import torch
import torch.nn as nn
from torchvision import models
from Models.Resnet50.encoder import  SharedResNet50Encoder
from Models.Resnet50.depthDecoder import DepthDecoder

class MonocularDepthNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = SharedResNet50Encoder(pretrained)
        self.decoder = DepthDecoder(num_ch_enc=[64, 256, 512, 1024, 2048])

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
