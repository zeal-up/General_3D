import os, sys
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import utils.pytorch_utils as pt_utils 
from models.DGCNN.BaseEdgeConvModule import transform_net, _baseEdgeConv


class DGCNN_seg_feature(nn.Module):
    def __init__(self, num_classes=16, k:int=30):
        super().__init__()
        self.eps = 1e-3
        self.k = k
        self.transform = transform_net(K=3)
        self.num_classes = num_classes

        # EdgeConv 1
        self.edgeconv1_1 = _baseEdgeConv(3, 64, k=k, pool=False)
        self.edgeconv1_2 = pt_utils.Conv2d(64, 64, bn=True)
        # max_pool and avg_pool then send to conv1_3
        self.edgeconv1_3 = pt_utils.Conv2d(128, 64, bn=True)
        # output1

        # EdgeConv2
        self.edgeconv2_1 = _baseEdgeConv(64, 64, k=k, pool=False)
        # max_pool and avg_pool then send to conv2_2
        self.edgeconv2_2 = pt_utils.Conv2d(128, 64, bn=True)
        # output2

        self.edgeconv3_1 = _baseEdgeConv(64, 64, k=k, pool=False)
        # max_pool and avg_pool then send to conv3_3
        self.edgeconv3_2 = pt_utils.Conv2d(128, 64, bn=True)
        # output3

        # [output1, output2, output3] -> mlp(1024)
        self.mlp = pt_utils.Conv2d(64+64+64, 1024, bn=True)

        # one_hot_label conv
        self.one_hot_expand = pt_utils.Conv2d(self.num_classes, 128, bn=True)

        
    def forward(self, pc, classes_labels):
        # classes_labels must be a onthot label B x num_classes
        pc = pc.permute(0, 2, 1)
        batch_size = pc.size(0)
        num_point = pc.size(2)
        assert pc.size(1)==3
        conv_feat = pc

        #transform
        transform = self.transform(pc)
        conv_feat = torch.bmm(pc.permute(0, 2, 1), transform).permute(0, 2, 1)

        # Edgeconv1
        conv_feat = self.edgeconv1_1(conv_feat)
        conv_feat = self.edgeconv1_2(conv_feat)
        net_max_1, _ = torch.max(conv_feat, dim=-1, keepdim=True) # B x C x N x 1
        net_mean_1 = torch.mean(conv_feat, dim=-1, keepdim=True) # B x C x N x 1
        conv_feat = self.edgeconv1_3(torch.cat([net_max_1, net_mean_1], dim=1))
        output1 = conv_feat

        # Edgeconv2
        conv_feat = self.edgeconv2_1(conv_feat)
        net_max_2, _ = torch.max(conv_feat, dim=-1, keepdim=True)
        net_mean_2 = torch.mean(conv_feat, dim=-1, keepdim=True)
        conv_feat = self.edgeconv2_2(torch.cat([net_max_2, net_mean_2], dim=1))
        output2 = conv_feat

        #Edgeconv3
        conv_feat = self.edgeconv3_1(conv_feat)
        net_max_3, _ = torch.max(conv_feat, dim=-1, keepdim=True)
        net_mean_3 = torch.mean(conv_feat, dim=-1, keepdim=True)
        conv_feat = self.edgeconv3_2(torch.cat([net_max_3, net_mean_3], dim=1))
        output3 = conv_feat # B x 64 x N x 1 

        #self.mlp
        conv_feat = self.mlp(torch.cat([output1, output2, output3], dim=1)) # B x 1024 x N x 1
        output4 = conv_feat
        out_max, _ = torch.max(conv_feat, dim=-2, keepdim=True) # B x 1024 x 1 x 1
        feature = out_max

        #one_hot_label expand
        one_hot_label = classes_labels.unsqueeze(-1).unsqueeze(-1) # B x num_classes x 1 x 1
        one_hot_label_expand = self.one_hot_expand(one_hot_label) # B x 128 x 1 x 1

        out_max = torch.cat([out_max, one_hot_label_expand], dim=1)
        out_max = out_max.expand(batch_size, 128+1024, num_point, 1) # B x C x N x 1

        concat = torch.cat([
            out_max,
            net_max_1,
            net_mean_1,
            output1,
            net_max_2,
            net_mean_2,
            output2,
            net_max_3,
            net_mean_3,
            output3,
            output4
        ], dim=1)

        return feature.squeeze(), concat


class DGCNN_seg_classifier(nn.Module):
    def __init__(self, num_parts:int=50):
        super().__init__()
        self.drop = nn.Dropout2d(p=0.4)
        self.num_parts = num_parts

        self.fc1 = pt_utils.Conv2d(128+1024+64*9+1024, 256, bn=True) # has relu

        self.fc2 = pt_utils.Conv2d(256, 256, bn=True)

        self.fc3 = pt_utils.Conv2d(256, num_parts, bn=False, activation=None)

    def forward(self, feat):
        feat = self.drop(self.fc1(feat))
        feat = self.drop(self.fc2(feat))

        feat = self.fc3(feat) # B x num_parts x N x 1

        return feat.squeeze(-1) # B x num_parts x N


class DGCNN_seg_fullnet(nn.Module):
    def __init__(self, num_parts:int=50, num_classes:int=16, k:int=30):
        super().__init__()
        self.num_parts = num_parts
        self.num_classes = num_classes
        self.k = k

        self.feature_extractor = DGCNN_seg_feature(num_classes=num_classes, k=k)
        self.classifier = DGCNN_seg_classifier(num_parts=num_parts)


    def forward(self, data_dict):
        #data_dict = {'pc':points, 'one_hot_labels':one_hot}
        pc = data_dict['pc']
        one_hot_labels = data_dict['one_hot_labels']

        _, concat_feature = self.feature_extractor(pc, one_hot_labels)

        scores = self.classifier(concat_feature)

        return scores # B x num_parts x


   
if __name__ == '__main__':
    # import os
    # import sys
    # print(sys.path)
    model = DGCNN_seg_fullnet(num_parts=50, num_classes=16)
    # print(model)
    # for m in model.children():
    #     print(m)
    # print(model.children())

    pc = torch.rand(32, 1024, 3)
    classes_labels = torch.ones((32, 1), dtype=torch.long)
    one_hot_label = torch.FloatTensor(32, 16)
    one_hot_label.zero_()
    one_hot_label.scatter_(1, classes_labels, 1)
    print(one_hot_label)

    parts_labels = torch.LongTensor(32, 1024)
    parts_labels.fill_(1)
    criterion = nn.CrossEntropyLoss()
    batch_data = {}
    batch_data['pc'] = pc 
    batch_data['one_hot_labels'] = one_hot_label
    output = model(batch_data)
    print(output.size())
    loss = criterion(output, parts_labels)
    print(loss)
    loss.backward()