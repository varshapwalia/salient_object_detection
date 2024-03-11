import torch
from torch import nn
import torch.nn.functional as F
from resnet import resnet50  # Ensure this is correctly importing your ResNet50 model

# Only ResNet configuration is needed now
config_resnet = {
    'convert': [[64, 256, 512, 1024, 2048], [128, 256, 512, 512, 512]],
    'merge1': [[128, 256, 128, 3, 1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2], [512, 0, 512, 7, 3]],
    'merge2': [[128], [256, 512, 512, 512]]
}

class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.ReLU(inplace=True)) 
                                     for in_ch, out_ch in zip(list_k[0], list_k[1])])

    def forward(self, x):
        return [layer(feat) for layer, feat in zip(self.layers, x)]


class MergeLayer1(nn.Module): # list_k: [[64, 512, 64], [128, 512, 128], [256, 0, 256] ... ]
    def __init__(self, list_k):
        super(MergeLayer1, self).__init__()
        self.list_k = list_k
        trans, up, score = [], [], []
        for ik in list_k:
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
    def __init__(self, list_k):
        super(MergeLayer2, self).__init__()
        self.list_k = list_k
        trans, up, score = [], [], []
        for i in list_k[0]:
            tmp = []
            tmp_up = []
            tmp_score = []
            feature_k = [[3,1],[5,2], [5,2], [7,3]]
            for idx, j in enumerate(list_k[1]):
                tmp.append(nn.Sequential(nn.Conv2d(j, i, 1, 1, bias=False), nn.ReLU(inplace=True)))

                tmp_up.append(nn.Sequential(nn.Conv2d(i , i, feature_k[idx][0], 1, feature_k[idx][1]), nn.ReLU(inplace=True), nn.Conv2d(i, i,  feature_k[idx][0],1 , feature_k[idx][1]), nn.ReLU(inplace=True), nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.ReLU(inplace=True)))
                tmp_score.append(nn.Conv2d(i, 1, 3, 1, 1))
            trans.append(nn.ModuleList(tmp))

            up.append(nn.ModuleList(tmp_up))
            score.append(nn.ModuleList(tmp_score))
            

        self.trans, self.up, self.score = nn.ModuleList(trans), nn.ModuleList(up), nn.ModuleList(score)       
        self.final_score = nn.Sequential(nn.Conv2d(list_k[0][0], list_k[0][0], 5, 1, 2), nn.ReLU(inplace=True), nn.Conv2d(list_k[0][0], 1, 3, 1, 1))
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
    
class TUN_bone(nn.Module):
    def __init__(self):
        super(TUN_bone, self).__init__()
        self.base = resnet50(pretrained=True)
        self.convert = ConvertLayer(config_resnet['convert'])
        self.merge1 = MergeLayer1(config_resnet['merge1'])
        self.merge2 = MergeLayer2(config_resnet['merge2'])

    def forward(self, x):
        x_size = x.size()[2:]
        conv2merge = self.base(x)        
        conv2merge = self.convert(conv2merge)
        up_edge, edge_feature, up_sal, sal_feature = self.merge1(conv2merge, x_size)
        up_sal_final = self.merge2(edge_feature, sal_feature, x_size)
        return up_edge, up_sal, up_sal_final

def build_model():
    return TUN_bone()

# Weight initialization functions here if needed
