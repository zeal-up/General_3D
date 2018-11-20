import torch 
import torch.nn as nn  
import torch.nn.functional as F 

import utils.pytorch_utils as pt_utils 
import models.model_utils as md_utils 

from models.PointNet2.BasePointnetModule import _BasePointnetMSGModule

class PointNet2MSG_cls_feature(nn.Module):
    def __init__(self):
        super().__init__()
        inchannel = 3
        self.msg1 = _BasePointnetMSGModule(
            npoint=512,
            radius=[0.1, 0.2, 0.4],
            nsamples=[16, 32, 128],
            mlps=[[inchannel, 32, 32, 64], [inchannel, 64, 64, 128], [inchannel, 64, 96, 128]]
        )

        inchannel = 64 + 128 + 128 + 3
        self.msg2 = _BasePointnetMSGModule(
            npoint=128,
            radius=[0.2, 0.4, 0.8],
            nsamples=[32, 64, 128],
            mlps=[[inchannel, 64, 64, 128], [inchannel, 128, 128, 256], [inchannel, 128, 128, 256]]
        )

        inchannel = 128 + 256 + 256 + 3

        self.SA = pt_utils.SharedMLP([inchannel, 256, 512, 1024], bn=True)

    def forward(self, pc):
        '''
        pc : B x N x 3
        
        output : B x 1024
        '''
        assert pc.size()[2] == 3, 'illegal pc size : {}'.format(pc.size())

        pc = pc.permute(0, 2, 1)

        pc_sample, feat = self.msg1(pc, None)
        pc_sample, feat = self.msg2(pc_sample, feat) # B x 3 x npoint/ B x C x npoint

        pc_sample, feat = pc_sample.unsqueeze(-1), feat.unsqueeze(-1)

        feat = self.SA(torch.cat([pc_sample, feat], dim=1)) # B x 1024 x npoint x 1

        feat = feat.squeeze(-1)
        feat = torch.max(feat, -1)[0]

        return feat


class Pointnet2MSG_cls_classifier(nn.Module):
    def __init__(self, num_classes:int=40):
        super().__init__()

        self.num_classes = num_classes
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = pt_utils.FC(1024, 512, bn=True)
        self.fc2 = pt_utils.FC(512, 256, bn=True)
        self.fc3 = pt_utils.FC(256, self.num_classes, bn=False, activation=None)

    def forward(self, feat):
        feat = self.drop(self.fc1(feat))
        feat = self.drop(self.fc2(feat))
        feat = self.fc3(feat)

        return feat


class Pointnet2MSG_cls_fullnet(nn.Module):
    def __init__(self, num_classes:int=40):
        super().__init__()

        self.num_classes = num_classes
        self.feature_extractor = PointNet2MSG_cls_feature()
        self.classifier = Pointnet2MSG_cls_classifier(num_classes=self.num_classes)

    def forward(self, pc):
        feat = self.feature_extractor(pc)
        feat = self.classifier(feat)

        return feat