import torch
from torch.autograd import Function
from .._ext import query_ball_point

class _query_ball_point(Function):

    def __init__(ctx, radius, nsample):
        #super(_farthest_point_sample, ctx).__init__()
        ctx.radius = radius
        ctx.nsample = nsample

    def forward(ctx, xyz1, xyz2):
        '''
        Input:
            radius: float32, ball search radius
            nsample: int32, number of points selected in each ball region
            xyz1: (batch_size, 3, ndataset) float32 array, input points
            xyz2: (batch_size, 3, npoint) float32 array, query points
        Output:
            idx: (batch_size, npoint, nsample) int32 array, indices to input points
            pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
        '''
        assert xyz1.is_cuda and xyz1.size(1) == 3
        assert xyz2.is_cuda and xyz2.size(1) == 3
        assert xyz1.size(0) == xyz2.size(0)
        assert xyz1.is_contiguous()
        assert xyz2.is_contiguous()

        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()

        batch_size = xyz1.size(0)
        n = xyz1.size(1)
        m = xyz2.size(1)
        idx = xyz1.new(batch_size, m, ctx.nsample).long()
        pts_cnt = xyz1.new(batch_size, m).int()
        query_ball_point.query_ball_point_forward_cuda(ctx.radius, ctx.nsample, xyz1, xyz2, idx, pts_cnt)
        return idx, pts_cnt
    
    def backward(ctx, grad_output):
        return None