import os,sys
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn 
import torch.nn.functional as F 

from models.SpiderCNN.BaseSpiderConvModule import _BaseSpiderConv
import models.model_utils as md_utils 
import utils.pytorch_utils as pt_utils 

class Spidercnn_cls_feature(nn.Module):
    '''
    input
    pc(with normals) : B x N x 6 or pc(without normals) : B x N x 3
    return 
    feature : B x C
    '''
    def __init__(self, K_knn:int=20, withnor=True, taylor_channel:int=3):
        super().__init__()
        self.K_knn = K_knn
        self.withnor = withnor
        self.batch_size = batch_size
        self.num_points = num_points
        self.taylor_channel = taylor_channel
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
        cat_feat = torch.topk(cat_feat, 2, dim=2)[0] # B x 480 x 2
        cat_feat = cat_feat.view(B, -1) # B x 960

        return cat_feat

class Spidercnn_cls_classifier(nn.Module):
    def __init__(self, num_classes:int=40):
        super().__init__()
        self.num_classes = num_classes
        self.drop = nn.Dropout(p = 0.5)
        self.fc1 = pt_utils.FC(960, 512, bn=True)
        self.fc2 = pt_utils.FC(512, 256, bn=True)
        self.fc3 = pt_utils.FC(256, num_classes, bn=False, activation=None)

    def forward(self, feat):
        feat = self.drop(self.fc1(feat))
        feat = self.drop(self.fc2(feat))
        feat = self.fc3(feat)

        return feat

class Spidercnn_cls_fullnet(nn.Module):
    def __init__(self, K_knn:int=20, withnor=True, taylor_channel:int=3, num_classes:int=40):
        super().__init__()

        self.feature_extractor = Spidercnn_cls_feature(K_knn, withnor, taylor_channel)
        self.classifier = Spidercnn_cls_classifier(num_classes)

    def forward(self, pc):
        feature = self.feature_extractor(pc)
        scores = self.classifier(feature)

        return scores


if __name__ == "__main__":
    batch_size = 10
    num_points = 100
    K_knn = 20
    withnor = True
    model = Spidercnn_cls_fullnet(K_knn, withnor, batch_size, num_points)

    pc = torch.randn(batch_size, num_points, 6)

    output = model(pc)
    print(output.size())



