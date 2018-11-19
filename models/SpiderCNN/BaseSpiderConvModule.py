import os,sys
sys.path.append(os.path.abspath('.'))
# print(sys.path)
import torch
import torch.nn as nn
import utils.pytorch_utils as pt_utils
import models.model_utils as md_utils


class _BaseSpiderConv(nn.Module):
    def __init__(self, in_channel, out_channel, taylor_channel, K_knn):
        super().__init__()
        self.K_knn = K_knn
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.taylor_channel = taylor_channel


        
        self.conv1 = pt_utils.Conv2d(19, self.taylor_channel, bn=False, activation=None)

        self.conv2 = pt_utils.Conv2d(
            in_channel*taylor_channel,
            out_channel,
            kernel_size=[1, K_knn],
            bn=True
        )
        

    def forward(self, feat, idx, group_pc):
        '''
        feat : B x in_channel x N
        idx(knn_indices) : B x N x k
        group_pc : B x 3 x N x k

        return:
        feat : B x out_channel x N
        '''
        B, in_channel, N = feat.size()
        _, _, k = idx.size()

        assert k == self.K_knn, 'illegal k'

        group_feat = md_utils._indices_group(feat, idx) # B x inchannel x N x k

        X = group_pc[:, 0, :, :].unsqueeze(1)
        Y = group_pc[:, 1, :, :].unsqueeze(1)
        Z = group_pc[:, 2, :, :].unsqueeze(1)

        XX, YY, ZZ = X**2, Y**2, Z**2
        XXX, YYY, ZZZ =  XX*X, YY*Y, ZZ*Z
        XY, XZ, YZ = X*Y, X*Z, Y*Z
        XXY, XXZ, YYZ, YYX, ZZX, ZZY, XYZ = X*XY, X*XZ, Y*YZ, Y*XY, Z*XZ, Z*YZ, XY*Z


        group_XYZ = torch.cat([
            X, Y, Z, XX, YY, ZZ, XXX, YYY, ZZZ,\
            XY, XZ, YZ, XXY, XXZ, YYZ, YYX, ZZX, ZZY, XYZ
        ], dim=1) # B x 20 x N x k

        
        taylor = self.conv1(group_XYZ) # B x taylor_channel x N x k

        group_feat = group_feat.unsqueeze(2) #B x inchannel x 1 x N x k
        taylor = taylor.unsqueeze(1) # B x 1 x taylor_channel x N x k

        group_feat = torch.mul(group_feat, taylor).view(B, self.in_channel*self.taylor_channel, N, k)

        group_feat = self.conv(group_feat) # B x out_channel x N x 1

        group_feat = group_feat.squeeze(-1)

        return group_feat



if __name__ == '__main__':
    in_channel = 3
    out_channel = 6
    taylor_channel = 9
    k = 3
    batch_size = 3
    num_points = 10
    model = _BaseSpiderConv(in_channel, out_channel, taylor_channel, batch_size, num_points, k)

    pc = torch.randn(batch_size, 3, num_points)
    feat = torch.randn(batch_size, in_channel, num_points)
    idx = md_utils._knn_indices(pc, k)
    group_pc = md_utils._indices_group(pc, idx)
    pc = pc.unsqueeze(-1).expand(batch_size, 3, num_points, k)
    group_pc = group_pc - pc

    output = model(feat, idx, group_pc)
    print(output.size())

        


