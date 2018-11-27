import os,sys
sys.path.append(os.path.abspath('.'))
import torch
import torch.nn as nn    
import torch.nn.functional as F 

import models.model_utils as md_utils 
import utils.pytorch_utils as pt_utils 

from c_lib.fps.modules.farthest_point_sample import FarthestPointSample


class random_indices(nn.Module):
    '''
    pts : B x N x 3 
    return : B x P
    '''
    def __init__(self, P):
        super().__init__()
        self.P = P
    def forward(self, pts):
        B, N, _ = pts.size() 
        indice = torch.randint(low=0, high=N, size=(B, self.P), dtype=torch.long).to(pts.device)
        return indice
    

class _Basexconv(nn.Module):
    def __init__(self, K, D, P, C_in, C_out, C_delta, depth_multiplier, sampling='random', with_global=False):
        super().__init__()

        '''
        in the first layer, C_in set to be 0, and the nn_fts_input will only be the delta feature 
        pts(points) : origin points -> B x 3 x N
        fts(features) : origin features -> B x C x N
        qrs(querys) : query points -> B x 3 x P
        nn_pts_local : # B x 3 x P x K
        nn_fts_input : B x C_in+C_delta x P x K

        return : sample point and features
        '''
        self.with_global = with_global
        if sampling=='random':
            self.sample = random_indices(P) 
        elif sampling == 'fps':
            self.sample = FarthestPointSample(P)
        else :
            pass
        self.K = K
        self.D = D  
        self.P = P
        self.depth_multiplier = depth_multiplier

        #the input tot mlp_delta is nn_pts_local(B x 3 x P x K)
        self.mlp_delta = pt_utils.SharedMLP(
            [3, C_delta, C_delta],
            bn=True, activation=nn.ELU(inplace=True),
            act_before_bn=True
            )# B x C_delta x P x K


        # the input to X_transform is nn_pts_local(B x 3 x P x K)
        self.X_transform0 = pt_utils.Conv2d(
            3,
            K*K,
            kernel_size=(1, K),
            bn=True,
            activation=nn.ELU(inplace=True),
            act_before_bn=True
            ) # B x K*K x P x 1
        
        self.X_transform1 = nn.Sequential(
            nn.Conv2d(K, K*K, kernel_size=(1, K), groups=K),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(K*K)
        ) # B x K*K x P x 1
        self.X_transform2 = nn.Sequential(
            nn.Conv2d(K, K*K, kernel_size=(1, K), groups=K),
            nn.BatchNorm2d(K*K)
        ) # B x K*K x P x 1

        # depth_multiplier = torch.ceil(float(C_out)/(C_in + C_delta))
       

        self.conv = nn.Sequential(
            nn.Conv2d(C_in+C_delta, (C_in+C_delta)*depth_multiplier, kernel_size=(1, K), groups=(C_in+C_delta)),
            # nn.ELU(inplace=True),
            nn.BatchNorm2d((C_in+C_delta)*depth_multiplier),
            nn.Conv2d((C_in+C_delta)*depth_multiplier, C_out, kernel_size=1),
            nn.ELU(True),
            nn.BatchNorm2d(C_out)
        ) # equal to tf.layers.seperable_conv2d

        if self.with_global:
            self.conv_global = pt_utils.SharedMLP(
                [3, C_out // 4, C_out //4],
                bn=True,
                activation=nn.ELU(inplace=True),
                act_before_bn=True
            )

    def forward(self, pts, fts):
        assert pts.size()[1] == 3, 'illegal pointcloud size:{}'.format(pts.size())
        B, _, N = pts.size()

        if self.P == -1: # P==-1 has two situation 1),input layer 2),sample number consistent within two layer
            qrs = pts
            self.P = N
        else:
            sample_indices = self.sample(pts.permute(0, 2, 1).contiguous())
            sample_indices = sample_indices.unsqueeze(1).expand(B, 3, self.P)
            qrs = torch.gather(pts, 2, sample_indices) # B x 3 x P
        indices_dilated = md_utils._knn_indices(pts, k=self.K*self.D, centroid=qrs) #B x P x K*D
        indices = indices_dilated[:, :, ::self.D] # B x P x K
        nn_pts = md_utils._indices_group(pts, indices) # B x 3 x P x K
        nn_pts_center = qrs.unsqueeze(-1).expand_as(nn_pts)
        nn_pts_local = nn_pts - nn_pts_center # B x 3 x P x K

        nn_fts_from_pts = self.mlp_delta(nn_pts_local) # B x C_delta x P x K
        if fts is None: # in the first layer
            nn_fts_input = nn_fts_from_pts
        else:
            nn_fts_input = md_utils._indices_group(fts, indices)
            nn_fts_input = torch.cat([nn_fts_from_pts, nn_fts_input], dim=1) # B x C_delta+C_in x P x K

        X = self.X_transform0(nn_pts_local) # B x K*K x P x 1
        X = X.view(B, self.K, self.P, self.K)
        X = self.X_transform1(X)
        X = X.view(B, self.K, self.P, self.K)
        X = self.X_transform2(X)
        X = X.view(B, self.K, self.P, self.K) 
        X = X.view(B*self.P, self.K, self.K)
        fts_X = torch.bmm(nn_fts_input.permute(0,2,1,3).contiguous().view(B*self.P, -1, self.K), X)
        fts_X = fts_X.view(B, self.P, -1, self.K).permute(0, 2, 1, 3) # B x C_delta+C_in x P x K

        fts_conv = self.conv(fts_X).squeeze(-1) # B x C_out x P 

        if self.with_global:
            fts_global = self.conv_global(qrs.unsqueeze(-1)).squeeze(-1) # B x C_out//4 x P
            return qrs, torch.cat([fts_global, fts_conv], dim=1)
        else :
            return qrs, fts_conv # B x C_out x P














