import torch 
import time



def _knn_indices(feat, k, centroid=None, dist=False):
    '''
    feat : B x C x N
    centroid : B x C x n
    dist : whether return dist value
    k : int

    return 
    knn_indices : B x n x k
    dist : B x n x dist
    '''
    if centroid is None:
        centroid = feat
    pow2_feat = torch.sum(feat.pow(2), dim=1, keepdim=True) # B x 1 x N
    pow2_centroid = torch.sum(centroid.pow(2), dim=1, keepdim=True) # B x 1 x n
    centroid_feat = torch.bmm(centroid.transpose(1,2), feat).mul_(-2) # B x n x N
    pow2_centroid = pow2_centroid.permute(0, 2, 1)
    distances = centroid_feat + pow2_centroid + pow2_feat
    k_dist, indices = torch.topk(distances, k, dim=-1, largest=False, sorted=False)
    if dist:
        return indices, k_dist
    else:
        return indices
    # r_a = torch.sum(feat.pow(2), dim=1, keepdim=True)
    # dis = torch.bmm(feat.transpose(1,2), feat).mul_(-2)
    # dis.add_(r_a.transpose(1,2) + r_a)

    # _, indices = torch.topk(dis, k, dim=-1, largest=False, sorted=False)

    # return indices

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


