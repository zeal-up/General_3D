import torch
from torch.autograd import Function
from .._ext import farthest_point_sample

class _farthest_point_sample(Function):

    def __init__(ctx, num_sample_points):
        #super(_farthest_point_sample, ctx).__init__()
        ctx.n_points = num_sample_points

    def forward(ctx, input):
        # input b * n * 3 
        # return b * n_points int64
        assert input.is_cuda and input.size(2) == 3
        assert input.is_contiguous()
        batch_size = input.size(0)
        num_points = input.size(1)
        temp = input.new(32, num_points)
        indices = input.new(batch_size, ctx.n_points).long()
        farthest_point_sample.fps_forward_cuda(ctx.n_points, input, temp, indices)
        return indices
    
    def backward(ctx, grad_output):
        return None