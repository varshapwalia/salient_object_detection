import torch
from torch import nn
import torch.nn.functional as F
from resnet import resnet50  # Assumes resnet50 is defined in a separate module

from torch.autograd import Variable


config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,512,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 256, 'edgeinfo':[[16, 16, 16, 16], 128, [16,8,4,2]],'edgeinfoc':[64,128], 'block': [[512, [16]], [256, [16]], [256, [16]], [128, [16]]], 'fuse': [[16, 16, 16, 16], True], 'fuse_ratio': [[16,1], [8,1], [4,1], [2,1]],  'merge1': [[128, 256, 128, 3,1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],[512, 0, 512, 7, 3]], 'merge2': [[128], [256, 512, 512, 512]]}


class ConvertLayer(nn.Module):
    """
    Converts feature maps from ResNet layers to specified channel sizes for further processing.
    """
    def __init__(self, channel_sizes):
        super(ConvertLayer, self).__init__()
        up = []
        for i in range(len(channel_sizes[0])):
          
            up.append(nn.Sequential(nn.Conv2d(channel_sizes[0][i], channel_sizes[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))

        self.convert0 = nn.ModuleList(up)
        
    def forward(self, feature_maps):
        # Convert each feature map using the corresponding conversion layer
        return [layer(fmap) for layer, fmap in zip(self.convert0, feature_maps)]

        
        
class MergeLayer1(nn.Module): # channel_sizes: [[64, 512, 64], [128, 512, 128], [256, 0, 256] ... ]
    """
    Initiates EGNet merging by generating preliminary edge and saliency predictions, using convolutions, upsampling, 
    and scoring to enhance spatial resolution and discrimination.
    """
    def __init__(self, channel_sizes):
        super(MergeLayer1, self).__init__()
        self.channel_sizes = channel_sizes
        trans, up, score = [], [], []
        for ik in channel_sizes:
            if ik[1] > 0:
                trans.append(nn.Sequential(nn.Conv2d(ik[1], ik[0], 1, 1, bias=False), nn.ReLU(inplace=True)))

           
            up.append(nn.Sequential(nn.Conv2d(ik[0], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True), nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True), nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True)))
            score.append(nn.Conv2d(ik[2], 1, 3, 1, 1))
        trans.append(nn.Sequential(nn.Conv2d(512, 128, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.trans, self.up, self.score = nn.ModuleList(trans), nn.ModuleList(up), nn.ModuleList(score)
        self.relu =nn.ReLU()

    def forward(self, list_x, x_size):
        up_edge, up_sal, edge_feature, sal_feature = [], [], [], []
        
        
        num_f = len(list_x)
        tmp = self.up[num_f - 1](list_x[num_f-1])
        sal_feature.append(tmp)
        U_tmp = tmp
        up_sal.append(F.interpolate(self.score[num_f - 1](tmp), x_size, mode='bilinear', align_corners=True))
        
        for j in range(2, num_f ):
            i = num_f - j
             
            if list_x[i].size()[1] < U_tmp.size()[1]:
                U_tmp = list_x[i] + F.interpolate((self.trans[i](U_tmp)), list_x[i].size()[2:], mode='bilinear', align_corners=True)
            else:
                U_tmp = list_x[i] + F.interpolate((U_tmp), list_x[i].size()[2:], mode='bilinear', align_corners=True)
            
            
            
                
            
            tmp = self.up[i](U_tmp)
            U_tmp = tmp
            sal_feature.append(tmp)
            up_sal.append(F.interpolate(self.score[i](tmp), x_size, mode='bilinear', align_corners=True))

        U_tmp = list_x[0] + F.interpolate((self.trans[-1](sal_feature[0])), list_x[0].size()[2:], mode='bilinear', align_corners=True)
        tmp = self.up[0](U_tmp)
        edge_feature.append(tmp)
       
        up_edge.append(F.interpolate(self.score[0](tmp), x_size, mode='bilinear', align_corners=True)) 
        return up_edge, edge_feature, up_sal, sal_feature
    
       
class MergeLayer2(nn.Module): 
    """
    Refines edge and saliency predictions by cross-merging two sets of feature maps, integrating diverse information 
    for high-resolution, accurate segmentation outcomes in EGNet.
    """
    def __init__(self, channel_sizes):
        super(MergeLayer2, self).__init__()
        self.channel_sizes = channel_sizes
        trans, up, score = [], [], []
        for i in channel_sizes[0]:
            tmp = []
            tmp_up = []
            tmp_score = []
            feature_k = [[3,1],[5,2], [5,2], [7,3]]
            for idx, j in enumerate(channel_sizes[1]):
                tmp.append(nn.Sequential(nn.Conv2d(j, i, 1, 1, bias=False), nn.ReLU(inplace=True)))

                tmp_up.append(nn.Sequential(nn.Conv2d(i , i, feature_k[idx][0], 1, feature_k[idx][1]), nn.ReLU(inplace=True), nn.Conv2d(i, i,  feature_k[idx][0],1 , feature_k[idx][1]), nn.ReLU(inplace=True), nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.ReLU(inplace=True)))
                tmp_score.append(nn.Conv2d(i, 1, 3, 1, 1))
            trans.append(nn.ModuleList(tmp))

            up.append(nn.ModuleList(tmp_up))
            score.append(nn.ModuleList(tmp_score))
            

        self.trans, self.up, self.score = nn.ModuleList(trans), nn.ModuleList(up), nn.ModuleList(score)       
        self.final_score = nn.Sequential(nn.Conv2d(channel_sizes[0][0], channel_sizes[0][0], 5, 1, 2), nn.ReLU(inplace=True), nn.Conv2d(channel_sizes[0][0], 1, 3, 1, 1))
        self.relu =nn.ReLU()

    def forward(self, list_x, list_y, x_size):
        up_score, tmp_feature = [], []
        list_y = list_y[::-1]

        
        for i, i_x in enumerate(list_x):
            for j, j_x in enumerate(list_y):                              
                tmp = F.interpolate(self.trans[i][j](j_x), i_x.size()[2:], mode='bilinear', align_corners=True) + i_x                
                tmp_f = self.up[i][j](tmp)             
                up_score.append(F.interpolate(self.score[i][j](tmp_f), x_size, mode='bilinear', align_corners=True))                  
                tmp_feature.append(tmp_f)
       
        tmp_fea = tmp_feature[0]
        for i_fea in range(len(tmp_feature) - 1):
            tmp_fea = self.relu(torch.add(tmp_fea, F.interpolate((tmp_feature[i_fea+1]), tmp_feature[0].size()[2:], mode='bilinear', align_corners=True)))
        up_score.append(F.interpolate(self.final_score(tmp_fea), x_size, mode='bilinear', align_corners=True))
      


        return up_score


       
def extra_layer(base_model_cfg, resnet):
    """
    Configures and returns additional layers for the TUN model based on the specified base model configuration.
    """
    config = config_resnet if base_model_cfg == 'resnet' else None
    return resnet, MergeLayer1(config['merge1']), MergeLayer2(config['merge2'])


# TUN network

class TUN_bone(nn.Module):
    """
    TUN_bone integrates a base CNN model (either VGG or ResNet) with additional custom layers for advanced processing.
    """
    def __init__(self, base_model_cfg, base, merge1_layers, merge2_layers):
        super(TUN_bone, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.convert = ConvertLayer(config_resnet['convert'])
        self.base = base
        self.merge1 = merge1_layers
        self.merge2 = merge2_layers

    def forward(self, x):
        x_size = x.size()[2:]
        features = self.base(x)
        if self.base_model_cfg == 'resnet':
            features = self.convert(features)
        up_edge, edge_feature, up_sal, sal_feature = self.merge1(features, x_size)
        up_sal_final = self.merge2(edge_feature, sal_feature, x_size)
        return up_edge, up_sal, up_sal_final

# build the whole network
def build_model(base_model_cfg='resnet'):
    """
    Constructs a ResNet-based neural network optimized for performance, including GPU acceleration, weight initialization, 
    optional pre-trained weights, and learning rate setup. Employs Adam optimizer for loss minimization.
    """
    return TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, resnet50()))



def weights_init(m):
    #Initializing weights for the convolution to train itself on
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()


