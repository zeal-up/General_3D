import torch 
import torch.nn as nn 
import models.model_utils as md_utils 
import utils.pytorch_utils as pt_utils


class transform_net(nn.Module):
    def __init__(self, K_channel=3, K_knn=20):
        '''
        input : B x 3 x N
        '''
        super().__init__()
        self.eps = 1e-3
        self.K_channel = K_channel
        self.K_knn = K_knn

        self.conv1 = _baseEdgeConv(3, 64, k=K_knn, pool=False) # B x 64 x N x k
        self.conv2 = pt_utils.Conv2d(64, 128, bn=True)
        self.conv3 = pt_utils.Conv2d(128, 1024, bn=True)

        self.fc1 = pt_utils.FC(1024, 512, bn=True)
        self.fc2 = pt_utils.FC(512, 256, bn=True)
        self.fc3 = pt_utils.FC(256, self.K_channel**2, bn=False, activation=None)
        self.fc3.weight.data.fill_(0)
        self.fc3.bias.data.copy_(torch.eye(self.K_channel).view(-1).float())

    def forward(self, pc):
        conv_feat = self.conv1(pc)  # B*64*N*k
        conv_feat = self.conv2(conv_feat) # B*128*N*k
        conv_feat, _ = torch.max(conv_feat, -1, keepdim=True) #B*128*N*1

        conv_feat = self.conv3(conv_feat) # B*1024*N*1
        conv_feat, _ = torch.max(conv_feat, -2, keepdim=True) # B*1024*1*1

        conv_feat = self.fc1(conv_feat.squeeze())
        conv_feat = self.fc2(conv_feat) # B*256
        conv_feat = self.fc3(conv_feat)

        B, _ = conv_feat.size()
        
        conv_feat = conv_feat.view(B, self.K_channel, self.K_channel)
        
        return conv_feat



class _baseEdgeConv(nn.Module):
    '''
    input : B x c x N
    output : B x out_channel x N
    '''
    def __init__(self, in_channel:int, out_channel:int, k:int=20, pool=True):
        super().__init__()
        self.eps = 1e-3
        self.k = k
        self.pool = pool
        self.net = pt_utils.Conv2d(in_channel*2, out_channel, bn=True)
    def forward(self, feat):
        '''
        feat : B x C x N or B x C x N x1
        '''
        if len(feat.size()) == 4:
            feat = feat.squeeze(-1)
        B, C, N = feat.size()
        topk_feat = md_utils._knn_group(feat, self.k) # B x C x N x k
        feat = feat.unsqueeze(-1).expand(B, C, N, self.k)
        edge_feat = torch.cat([feat, topk_feat - feat], dim=1)

        # send to EdgeConv
        conv_feat = self.net(edge_feat) # B x C x N x k
        if self.pool:
            return torch.max(conv_feat, -1, keepdim=True)[0] # B x C x N x 1
        else :
            return conv_feat
