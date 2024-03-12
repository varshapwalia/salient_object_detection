import torch
from torch import nn
import torch.nn.functional as F
from resnet import create_resnet50 as resnet50  # Assumes resnet50 is defined in a separate module

from torch.autograd import Variable


config_resnet = {
    'convert': [[64, 256, 512, 1024, 2048], [128, 256, 512, 512, 512]],
    'merge1': [
        [128, 256, 128, 3, 1], [256, 512, 256, 3, 1],
        [512, 0, 512, 5, 2], [512, 0, 512, 5, 2], [512, 0, 512, 7, 3]
    ]
}


class ConvertLayer(nn.Module):
    """
    Converts feature maps from ResNet layers to specified channel sizes for further processing.
    """
    def __init__(self, config):
        super().__init__()
        self.conversion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            ) for in_channels, out_channels in zip(*config)
        ])

    def forward(self, feature_maps):
        # Convert each feature map using the corresponding conversion layer
        return [layer(fmap) for layer, fmap in zip(self.conversion_layers, feature_maps)]

        
        
class MergeLayer1(nn.Module):
    """
    Merges features from different layers. Supports transformations, upsampling, and integrates edge detection.
    """
    def __init__(self, config):
        super().__init__()
        self.transform_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.score_layers = nn.ModuleList()

        for in_channels, out_channels, mid_channels, kernel_size, padding in config:
            if in_channels > 0:
                self.transform_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), 
                    nn.ReLU(inplace=True)
                ))
            self.upsample_layers.append(nn.Sequential(
                nn.Conv2d(out_channels, mid_channels, kernel_size, padding=padding), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding), 
                nn.ReLU(inplace=True)
            ))
            self.score_layers.append(nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1))

    def forward(self, feature_maps, x_size):
        results = []
        for i, fmap in enumerate(feature_maps):
            if i < len(self.transform_layers):
                fmap = self.transform_layers[i](fmap)
            fmap = F.interpolate(fmap, size=x_size, mode='bilinear', align_corners=True)
            fmap = self.upsample_layers[i](fmap)
            results.append(self.score_layers[i](fmap))
        return results
    
       
class MergeLayer2(nn.Module):
    """
    Implements the second merging layer that processes and combines feature maps from different levels of the network.
    It applies transformations, upsamples the transformed features, and computes a score for each upsampled feature.
    """
    def __init__(self, config):
        super(MergeLayer2, self).__init__()
        self.transform_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.score_layers = nn.ModuleList()

        # Initialize transformation, upsampling, and scoring layers based on the configuration
        for base_channels in config[0]:
            transform_group, upsample_group, score_group = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
            for target_channels in config[1]:
                transform_group.append(nn.Sequential(
                    nn.Conv2d(target_channels, base_channels, kernel_size=1, stride=1, bias=False), 
                    nn.ReLU(inplace=True)
                ))
                feature_kernels = [[3, 1], [5, 2], [5, 2], [7, 3]]
                for kernel, padding in feature_kernels:
                    upsample_group.append(nn.Sequential(
                        nn.Conv2d(base_channels, base_channels, kernel, stride=1, padding=padding),
                        nn.ReLU(inplace=True)
                    ))
                score_group.append(nn.Conv2d(base_channels, 1, kernel_size=3, stride=1, padding=1))
            
            self.transform_layers.append(transform_group)
            self.upsample_layers.append(upsample_group)
            self.score_layers.append(score_group)
        
        self.final_score_layer = nn.Sequential(
            nn.Conv2d(config[0][0], config[0][0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config[0][0], 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, list_x, list_y, x_size):
        scores = []
        for x, trans_layers in zip(list_x, self.transform_layers):
            for y, up_layers in zip(reversed(list_y), self.upsample_layers):
                y_transformed = trans_layers(y)
                y_upsampled = F.interpolate(up_layers(y_transformed), size=x_size, mode='bilinear', align_corners=True)
                score = self.score_layers(x, y_upsampled)
                scores.append(score)
        
        # Combine all scores to compute the final score
        final_score = torch.cat(scores, dim=1)
        final_score = self.final_score_layer(final_score)
        return final_score

       
def extra_layer(base_model_cfg):
    """
    Configures and returns additional layers for the TUN model based on the specified base model configuration.
    """
    config = config_resnet if base_model_cfg == 'resnet' else config_vgg
    return MergeLayer1(config['merge1']), MergeLayer2(config['merge2'])


# TUN network

class TUN_bone(nn.Module):
    """
    TUN_bone integrates a base CNN model (either VGG or ResNet) with additional custom layers for advanced processing.
    """
    def __init__(self, base_model_cfg):
        super(TUN_bone, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = resnet50(pretrained=True) if base_model_cfg == 'resnet' else vgg16(pretrained=True)
        self.merge1, self.merge2 = extra_layer(base_model_cfg)

        if base_model_cfg == 'resnet':
            self.convert = ConvertLayer(config_resnet['convert'])

    def forward(self, x):
        x_size = x.size()[2:]
        features = self.base(x)
        if self.base_model_cfg == 'resnet':
            features = self.convert(features)
        merged_features = self.merge1(features, x_size)
        final_output = self.merge2(merged_features, x_size)
        return final_output

# build the whole network
def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg':
        return TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, vgg16()))
    elif base_model_cfg == 'resnet':
        return TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, resnet50()))



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()


