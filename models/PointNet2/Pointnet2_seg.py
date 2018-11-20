import torch 
import torch.nn as nn  
import torch.nn.functional as F 

import utils.pytorch_utils as pt_utils 
import models.model_utils as md_utils 

from models.PointNet2.BasePointnetModule import PointnetFPModule, _BasePointnetMSGModule

class Pointnet2MSG_seg_feature(nn.Module):
    def __init__(self, num_classes:int=16):
        super().__init__()
        self.num_classes = num_classes

        inchannel = 6 # with normals
        self.msg1 = _BasePointnetMSGModule(
            npoint=512,
            radius=[0.1, 0.2, 0.4],
            nsamples=[32, 64, 128],
            mlps=[[inchannel, 32, 32, 64], [inchannel, 64, 64, 128], [inchannel, 64, 96, 128]]
        )

        inchannel = 64 + 128 + 128 + 3
        self.msg2 = _BasePointnetMSGModule(
            npoint=128,
            radius=[0.4, 0.8],
            nsamples=[64, 128],
            mlps=[[inchannel, 128, 128, 256], [inchannel, 128, 196, 256]]
        )

        inchannel = 256 + 256 + 3
        self.SA = pt_utils.SharedMLP([inchannel, 256, 512, 1024], bn=True)

        inchannel = 1024 + 256 + 256
        self.fp1 = PointnetFPModule(mlp=[inchannel, 256, 256])
        
        inchannel = 256 + 64 + 128 + 128
        self.fp2 = PointnetFPModule(mlp=[inchannel, 256, 128])

        inchannel = 128 + 6 + num_classes
        self.fp3 = PointnetFPModule(mlp=[inchannel, 128, 128])


    def forward(self, batch_data):
        pc = batch_data['pc']
        assert pc.size()[2] == 6, 'illegal pc size:{}, pc must with normals'.format(pc.size())
        pc_normals = pc[:, :, 3:]
        pc = pc[:, :, :3]
        one_hot_labels = batch_data['one_hot_labels'] # B x 16
        B, N, _ = pc.size()

        
        pc = pc.permute(0, 2, 1)
        one_hot_labels = one_hot_labels.unsqueeze(-1).expand(B, one_hot_labels.size()[1], N)

        L1_pc, L1_feat = self.msg1(pc, pc_normals)
        L2_pc, L2_feat = self.msg2(L1_pc, L1_feat)

        L3_pc, L3_feat = L2_pc, self.SA(L2_feat.unsqueeze(-1)).squeeze(-1)

        up_feat1 = self.fp1(L3_pc, L2_pc, L3_feat, L2_feat)

        up_feat2 = self.fp2(L2_pc, L1_pc, up_feat1, L1_feat)

        L0_feat = torch.cat([one_hot_labels, pc, pc_normals], dim=1)
        up_feat3 = self.fp3(L1_pc, pc, up_feat2, L0_feat) # B x out_channel x N

        return up_feat3

class Pointnet2MSG_seg_classifier(nn.Module):
    def __init__(self, num_parts:int=50):
        super().__init__()

        self.num_parts = num_parts
        self.drop = nn.Dropout(p=0.5)

        self.fc1 = pt_utils.Conv1d(128, 128, bn=True)
        self.fc2 = pt_utils.Conv1d(128, num_parts, bn=False, activation=None)


    def forward(self, feat):
        feat = self.drop(self.fc1(feat))
        feat = self.fc2(feat)

        return feat # B x 50 x N


class Pointnet2MSG_seg_fullnet(nn.Module):
    def __init__(self, num_classes:int=16, num_parts:int=50):
        super().__init__()

        self.num_classes = num_classes
        self.num_parts = num_parts
        self.feature_extractor = Pointnet2MSG_seg_feature(num_classes=self.num_classes)
        self.classifier = Pointnet2MSG_seg_classifier(num_parts=self.num_parts)

    def forward(self, batch_data):
        feature = self.feature_extractor(batch_data)
        feature = self.classifier(feature)

        return feature



