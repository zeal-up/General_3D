import os, sys
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import utils.pytorch_utils as pt_utils 
from models.DGCNN.BaseEdgeConvModule import transform_net, _baseEdgeConv

class DGCNN_cls_feature(nn.Module):
    def __init__(self, k:int=20, last_bn=True):
        super().__init__()
        self.eps = 1e-3
        self.k = k
        self.last_bn = last_bn
        self.transform = transform_net(K=3)

        # EdgeConv 1
        self.edgeconv1_1 = _baseEdgeConv(3, 64, k=k)
        self.edgeconv1_2 = _baseEdgeConv(64, 64, k=k)
        self.edgeconv1_3 = _baseEdgeConv(64, 64, k=k)

        # EdgeConv2
        self.edgeconv2_1 = _baseEdgeConv(64, 128, k=k)

        # mlp1
        self.mlp = pt_utils.Conv2d(
            320,
            1024,
            bn=last_bn,
            activation=None if not last_bn else nn.ReLU(inplace=True))
        
    def forward(self, pc):
        '''
        input : B x N x 3
        output : B x 1024
        '''

        pc = pc.permute(0, 2, 1)
        assert pc.size(1)==3

        #transform
        transform = self.transform(pc)
        conv_feat = torch.bmm(pc.permute(0, 2, 1), transform).permute(0, 2, 1) # B x 3 x N

        # conv1
        conv_feat = conv_feat.unsqueeze(-1) # B x 3 x N x 1
        conv_feat = self.edgeconv1_1(conv_feat) # B x C x N x 1
        feat1 = conv_feat
        conv_feat = self.edgeconv1_2(conv_feat)
        feat2 = conv_feat
        conv_feat = self.edgeconv1_3(conv_feat)
        feat3 = conv_feat

        # conv2
        conv_feat = self.edgeconv2_1(conv_feat)
        feat4 = conv_feat

        conv_feat = torch.cat([feat1, feat2, feat3, feat4], dim=1) 
        conv_feat = self.mlp(conv_feat)

        conv_feat = conv_feat.squeeze()
        conv_feat, _ = torch.max(conv_feat, -1) # B x 1024

        # conv_feat = conv_feat / conv_feat.norm(p=2, dim=1, keepdim=True)

        return conv_feat



class DGCNN_cls_classifier(nn.Module):
    def __init__(self, num_classes:int=40):
        super().__init__()

        self.drop = nn.Dropout(p=0.5)
        self.num_classes = num_classes


        self.fc1 = pt_utils.FC(1024, 512, bn=True)
        self.fc2 = pt_utils.FC(512, 256, bn=True)

        self.fc3 = pt_utils.FC(256, num_classes, bn=False, activation=None)

    def forward(self, conv_feat):

        fc_feat = self.drop(self.fc1(conv_feat))

        fc_feat = self.drop(self.fc2(fc_feat))

        fc_feat = self.fc3(fc_feat)

        return fc_feat


class DGCNN_cls_fullnet(nn.Module):
    def __init__(self, num_classes:int=40, k:int=20):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.feature = DGCNN_cls_feature(k=self.k)
        self.classifier = DGCNN_cls_classifier(num_classes=self.num_classes)
    
    def forward(self, pc):
        conv_feat = self.feature(pc)
        scores = self.classifier(conv_feat)

        return scores



if __name__ == '__main__':
    model = DGCNN_cls_fullnet(num_classes=40)
    # print(model)
    # for m in model.children():
    #     print(m)
    # print(model.children())

    pc = torch.rand(32, 1024, 3)
    target = torch.ones(32, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    output = model(pc)
    print(output.size())
    loss = criterion(output, target)
    loss.backward()
    