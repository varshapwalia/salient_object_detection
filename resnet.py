import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

is_affine = True

def create_3x3_conv(in_channels, out_channels, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion_factor = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = create_3x3_conv(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=is_affine)
        self.relu = nn.ReLU(inplace=True)
        # Second convolutional layer
        self.conv2 = create_3x3_conv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=is_affine)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion_factor = 4

    def __init__(self, in_channels, out_channels, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        # First bottleneck convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=is_affine)
        for param in self.bn1.parameters():
            param.requires_grad = False
        padding = 1
        if dilation_ in [2, 4]:
            padding = dilation_
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=is_affine)
        for param in self.bn2.parameters():
            param.requires_grad = False
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4, affine=is_affine)
        for param in self.bn3.parameters():
            param.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.initial_channels = 64
        super(ResNet, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=is_affine)
        for param in self.bn1.parameters():
            param.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation_ = 2)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0, 0.01)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)
        
    def _make_layer(self, block, channels, num_blocks, stride=1, dilation_=1):
        downsample = None
        if stride != 1 or self.initial_channels != channels * block.expansion_factor or dilation_ in [2, 4]:
            downsample = nn.Sequential(
                nn.Conv2d(self.initial_channels, channels * block.expansion_factor,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion_factor, affine=is_affine),
            )
            for param in downsample._modules['1'].parameters():
                param.requires_grad = False

        layers = []
        layers.append(block(self.initial_channels, channels, stride, dilation_=dilation_, downsample=downsample))
        self.initial_channels = channels * block.expansion_factor
        for _ in range(1, num_blocks):
            layers.append(block(self.initial_channels, channels, dilation_=dilation_))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        tmp_x = []
        # Initial convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_x.append(x)
        # Max pooling layer
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)
            tmp_x.append(x)

        return tmp_x

def create_resnet50(pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
        pass
    return model
