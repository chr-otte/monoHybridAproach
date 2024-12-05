import torch
import torch.nn as nn
from torchvision import models

class SharedResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(SharedResNet50Encoder, self).__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Create new first conv layer with 9 input channels
        self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # If using pretrained weights, adapt the first conv layer
        if pretrained:
            # Expand pretrained weights to 9 channels
            pretrained_weight = resnet50.conv1.weight.data
            # Initialize new conv weights by repeating the pretrained weights 3 times
            self.conv1.weight.data = pretrained_weight.repeat(1, 3, 1, 1)
            # Optional: normalize the weights
            self.conv1.weight.data = self.conv1.weight.data / 3.0

        # Copy the rest of the layers from pretrained model
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

    def forward(self, x):
        skip_connections = {}

        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_connections['conv1'] = x

        x = self.maxpool(x)
        x = self.layer1(x)
        skip_connections['layer1'] = x

        x = self.layer2(x)
        skip_connections['layer2'] = x

        x = self.layer3(x)
        skip_connections['layer3'] = x

        x = self.layer4(x)
        skip_connections['layer4'] = x

        return x, skip_connections
