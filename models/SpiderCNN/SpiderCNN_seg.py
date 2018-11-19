import os,sys
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn 
import torch.nn.functional as F 

from models.SpiderCNN.BaseSpiderConvModule import _BaseSpiderConv
import models.model_utils as md_utils 
import utils.pytorch_utils as pt_utils 


class Spidercnn_seg_feature(nn.Module):
    def __init__(self, K_knn:int=16, taylor_channel:int=3, withnor=True):
        super().__init__()

        self.K_knn = K_knn
        self.taylor_channel = taylor_channel
        self.withnor = withnor
        self.inchannel = 6 if withnor else 3

        self.spiderconv1 = _BaseSpiderConv(self.inchannel, 32, self.taylor_channel, self.K_knn)
        self.spiderconv2 = _BaseSpiderConv(32, 64, self.taylor_channel, self.K_knn)
        self.spiderconv3 = _BaseSpiderConv(64, 128, self.taylor_channel, self.K_knn)
        self.spiderconv4 = _BaseSpiderConv(128, 256, self.taylor_channel, self.K_knn)


    def forward(self, pc):
        '''
        pc_withnor : B x N x 6
        or pc_withoutnor : B x N x3
        '''
        assert pc.size()[2] == self.inchannel, 'illegal input pc size:{}'.format(pc.size())
        B, N, _ = pc.size()
        pc = pc.permute(0, 2, 1)
        pc_xyz = pc[:, 0:3, :]
        idx = md_utils._knn_indices(pc_xyz, k=self.K_knn)
        grouped_xyz = md_utils._indices_group(pc_xyz, idx) #B x 3 x N x k

        grouped_pc = pc_xyz.unsqueeze(-1).expand(B, 3, N, self.K_knn)
        grouped_pc = grouped_xyz - grouped_pc

        feat_1 = self.spiderconv1(pc, idx, grouped_pc) # B x 64 x N
        feat_2 = self.spiderconv2(feat_1, idx, grouped_pc)
        feat_3 = self.spiderconv3(feat_2, idx, grouped_pc)
        feat_4 = self.spiderconv4(feat_3, idx, grouped_pc)

        cat_feat = torch.cat([feat_1, feat_2, feat_3, feat_4], dim=1) # B x 480 x N
        point_feat = cat_feat
        cat_feat = torch.topk(cat_feat, 2, dim=2)[0] # B x 480 x 2
        cat_feat = cat_feat.view(B, -1) # B x 960

        return cat_feat, point_feat

class Spidercnn_seg_classifier(nn.Module):
    def __init__(self, num_parts:int=50):
        super().__init__()
        self.num_parts = 50
        self.drop = nn.Dropout2d(p=0.2)
        self.fc1 = pt_utils.Conv2d(1456, 256, bn=True)
        self.fc2 = pt_utils.Conv2d(256, 256, bn=True)
        self.fc3 = pt_utils.Conv2d(256, 128, bn=True)
        self.fc4 = pt_utils.Conv2d(128, self.num_parts, bn=False, activation=None)

    def forward(self, feat):
        feat = self.drop(self.fc1(feat))
        feat = self.drop(self.fc2(feat))
        feat = self.drop(self.fc3(feat))
        feat = self.fc4(feat)

        return feat

class Spidercnn_seg_fullnet(nn.Module):
    def __init__(self, K_knn:int=16, taylor_channel:int=3, withnor=True, num_parts:int=50):
        super().__init__()

        self.K_knn = K_knn
        self.taylor_channel = taylor_channel
        self.withnor = withnor
        self.num_parts = num_parts

        self.feature_extractor = Spidercnn_seg_feature(self.K_knn, self.taylor_channel, self.withnor)
        self.classifier = Spidercnn_seg_classifier(self.num_parts)

    def forward(self, batch_data):
        '''
        batch_data : dict contains ['pc'], ['one_hot_labels']
        output : B x num_parts x N
        '''
        pc = batch_data['pc'] # B x N x 6/3
        one_hot_labels = batch_data['one_hot_labels'] # B x N x num_classes(16)
        one_hot_labels = one_hot_labels.permute(0, 2, 1)
        B, N, _ = pc.size()

        global_feat, point_feat = self.feature_extractor(pc) # B x 960; B x 480 x N
        global_feat = global_feat.unsqueeze(-1).expand(B, 960, N)

        global_point_feat = torch.cat([global_feat, one_hot_labels, point_feat], dim=1)
        global_point_feat = global_point_feat.unsqueeze(-1) # B x 1456 x N x 1

        scores = self.classifier(global_point_feat)

        return scores
