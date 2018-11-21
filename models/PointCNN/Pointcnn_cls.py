import os,sys
sys.path.append(os.path.abspath('.'))

import torch 
import numpy as np
import torch.nn as nn

import utils.pytorch_utils as pt_utils 
import models.model_utils as md_utils 

from models.PointCNN.BasePointcnnModule import _Basexconv

class Pointcnn_cls_feature(nn.Module):
    def __init__(self):
        super().__init__()

        x = 3

        # x_conv1
        C_in = 0
        C_out = 16*x 
        C_delta = C_out//2 # the first is not input feature
        depth_multipiler = 4
        config ={
            'K':8, 'D':1, 'P':-1, 'C_in':C_in, 'C_out':C_out, 'C_delta':C_delta, \
            'sampling':'random', 'with_global':False, 'depth_multiplier':depth_multipiler}
        self.x_conv1 = _Basexconv(**config)

        # x_conv2
        C_in = C_out
        C_out = 32*x
        C_delta = C_in // 4
        depth_multipiler = int(np.ceil(C_out / C_in))
        config ={
            'K':12, 'D':2, 'P':384, 'C_in':C_in, 'C_out':C_out, 'C_delta':C_delta, \
            'sampling':'random', 'with_global':False, 'depth_multiplier':depth_multipiler}
        self.x_conv2 = _Basexconv(**config)

        # x_conv3
        C_in = C_out
        C_out = 64*x 
        C_delta = C_in // 4
        depth_multipiler = int(np.ceil(C_out / C_in))
        config ={
            'K':16, 'D':2, 'P':128, 'C_in':C_in, 'C_out':C_out, 'C_delta':C_delta, \
            'sampling':'random', 'with_global':False, 'depth_multiplier':depth_multipiler}
        self.x_conv3 = _Basexconv(**config)

        # x_conv4
        C_in = C_out
        C_out = 128 * x
        C_delta = C_in // 4
        depth_multipiler = int(np.ceil(C_out / C_in))
        config ={
            'K':16, 'D':3, 'P':-1, 'C_in':C_in, 'C_out':C_out, 'C_delta':C_delta, \
            'sampling':'random', 'with_global':True, 'depth_multiplier':depth_multipiler}
        # C_global = C_out // 4
        self.x_conv4 = _Basexconv(**config)

    def forward(self, pc):
        assert pc.size(2) == 3, 'illegal pc size : {}'.format(pc.size())
        pc = pc.permute(0, 2, 1).contiguous()

        sample_pc, feat = self.x_conv1(pc, None)
        sample_pc, feat = self.x_conv2(sample_pc, feat)
        sample_pc, feat = self.x_conv3(sample_pc, feat)
        sample_pc, feat = self.x_conv4(sample_pc, feat)

        return feat # B x 480 x 128


class Pointcnn_cls_classifier(nn.Module):
    def __init__(self, num_classes:int=40):
        super().__init__()

        self.num_classes = num_classes
        self.fc1 = pt_utils.Conv1d(480, 128 * 3, bn=True, activation=nn.ELU(True), act_before_bn=True)
        self.fc2 = pt_utils.Conv1d(128*3, 64*3, bn=True, activation=nn.ELU(True), act_before_bn=True)
        self.fc3 = pt_utils.Conv1d(64*3, num_classes, bn=False, activation=None)

        self.drop = nn.Dropout(p=0.8)

    def forward(self, feat):
        scores = self.fc1(feat)
        scores = self.drop(self.fc2(scores))
        scores = self.fc3(scores)

        return scores # B x 40 x 128

class Pointcnn_cls_fullnet(nn.Module):
    def __init__(self, num_classes:int=40):
        super().__init__()

        self.feature_extractor = Pointcnn_cls_feature()
        self.classifier = Pointcnn_cls_classifier(num_classes=num_classes)

    def forward(self, pc):
        feature = self.feature_extractor(pc)
        feature = self.classifier(feature)

        return feature


if __name__ == '__main__':
    model = Pointcnn_cls_fullnet(40)

    pc = torch.randn(10, 100, 3)

    output = model(pc)

    print(output.size())