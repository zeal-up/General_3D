import torch 
import torch.nn as nn  
import torch.nn.functional as F 

import utils.pytorch_utils as pt_utils 
import models.model_utils as md_utils 

from c_lib import FarthestPointSample
from c_lib import QueryBallPoint


class _BasePointnetMSGModule(nn.Module):

    def __init__(self, npoint:int, radius:list, nsamples:list, mlps:list):
        '''
        npoint : point number for fps sampling
        nsamples : sample point numbers for each radius
        '''
        super().__init__()
        assert len(radius) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.nsamples = nsamples 
        self.radius = radius
        self.mlps = mlps

        self.fps = FarthestPointSample(npoint)
        self.mlp_layers = nn.ModuleList()
        self.query_ball_point = nn.ModuleList()
        for mlp, radiu, nsample in zip(mlps, radius, nsamples):
            self.mlp_layers.append(pt_utils.SharedMLP(mlp, bn=True))
            self.query_ball_point.append(QueryBallPoint(radiu, nsample))

    def forward(self, pc, feat):
        '''
        input
        ---------------
        pc : B x 3 x N
        feat : B x C x N

        output
        ----------------
        pc_sample : B x 3 x npoint
        feat_sample : B x outchannel x npoint 
        '''
        B, _, N = pc.size()
        idx = self.fps(pc.permute(0,2,1).contiguous()) # B x npoint
        idx = idx.unsqueeze(1).expand(B, 3, self.npoint)
        pc_sample = torch.gather(pc, 2, idx) # B x 3 x npoint
        cat_feat = []

        for i in range(len(self.mlp_layers)):
            indices, _ = self.query_ball_point[i](pc.contiguous(), pc_sample.contiguous())
            grouped_pc = md_utils._indices_group(pc, indices) # B x 3 x npoint x nsample
            grouped_pc = grouped_pc - pc_sample.unsqueeze(-1).expand_as(grouped_pc)
            out_feat = grouped_pc.contiguous()
            if feat is not None: # feat will be None in the first layer
                grouped_feat = md_utils._indices_group(feat, indices) # B x C x npoint x nsample
                out_feat = torch.cat([grouped_pc, grouped_feat], dim=1) # B x C+3 x npoint x nsample
            out_feat = self.mlp_layers[i](out_feat)
            out_feat = torch.max(out_feat, -1)[0] # B x C_out x npoint
            cat_feat.append(out_feat)

        cat_feat = torch.cat(cat_feat, dim=1) # B x sum(mlp[-1]) x npoint

        return pc_sample, cat_feat

class _BasePointnetSSGModule(_BasePointnetMSGModule):
    def __init__(self, npoint:int, radiu:float, nsample:int, mlp:list):
        super().__init__(npoint, [radiu], [nsample], [mlp])

        

class PointnetFPModule(nn.Module):
    def __init__(self, mlp:list):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=True)

    def forward(self, pc_down, pc_up, feat_down, feat_up):
        '''
        pc_down : B x 3 x N_small
        pc_up : B x 3 x N_large

        feat_down : B x C1 x N_small
        feat_up : B x C2 x N_large

        return : B x mlp[-1] x N_large

        '''
        idx, dist = md_utils._knn_indices(
            feat=pc_down,
            k=3,
            centroid=pc_up,
            dist=True
        ) # B x N_large x k

        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        grouped_feat = md_utils._indices_group(feat_down, idx) # B x C1 x N_large x k
        weight = weight.unsqueeze(1) # B x 1 x N_large x k

        interpolated_feats = grouped_feat * weight
        interpolated_feats = torch.max(interpolated_feats, dim=-1)[0] # B x C1 x N_large

        interpolated_feats = torch.cat([interpolated_feats, feat_up], dim=1) # B x C1+C2 x N_large
        interpolated_feats = interpolated_feats.unsqueeze(-1)

        interpolated_feats = self.mlp(interpolated_feats)

        return interpolated_feats.squeeze(-1) # B x out_channel x N_large




        

