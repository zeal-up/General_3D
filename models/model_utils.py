import torch 
import time



def _knn_indices(feat, k):
    '''
    feat : B x C x N
    k : int

    return 
    knn_indices : B x N x k
    '''
    r_a = torch.sum(feat.pow(2), dim=1, keepdim=True)
    dis = torch.bmm(feat.transpose(1,2), feat).mul_(-2)
    dis.add_(r_a.transpose(1,2) + r_a)

    _, indices = torch.topk(dis, k, dim=-1, largest=False, sorted=False)

    return indices

def _indices_group(feat, indices):
    '''
    input
    feat : B x C x N
    indices : B x n x k

    output
    group_feat : B x C x N x k
    '''
    B, C, N = feat.size()
    _, n, k = indices.size()

    indices = indices.unsqueeze(1).expand(B, C, n, k)
    group_feat = feat.unsqueeze(-1).expand(B, C, N, k)
    group_feat = torch.gather(group_feat, 2, indices)

    return group_feat # B x C x n x k

def _knn_group(feat, k):
    '''
    input
    feat : B x C x N
    k : int

    output :
    group_feat : B x C x N x k
    '''
    B, C, N = feat.size()
    knn_indices = _knn_indices(feat, k) # B x N x k
    group_feat = _indices_group(feat, knn_indices)


    return group_feat




if __name__ == '__main__':
    feat = torch.randn(2, 3, 3)
    k = 2

    group_feat = _knn_group(feat, k)

    print(group_feat.size())


